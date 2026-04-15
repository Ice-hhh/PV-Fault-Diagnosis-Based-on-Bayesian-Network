from __future__ import annotations

import argparse
import os
from pathlib import Path

Path("outputs/.matplotlib").mkdir(parents=True, exist_ok=True)
Path("outputs/.cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib")))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("outputs/.cache")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import dump
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

from pv_fault_diagnosis.data import build_submission, flatten_daily_features, prepare_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce PV power prediction experiments.")
    parser.add_argument("--data-dir", default="data", help="Directory containing the competition CSV files.")
    parser.add_argument("--output", default="outputs/submission.csv", help="Path for the generated submission CSV.")
    parser.add_argument("--model-path", default="outputs/model.joblib", help="Path for the trained model artifact.")
    parser.add_argument(
        "--plot-path",
        default="outputs/training_process.png",
        help="Path for the generated training-process plot.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip writing the training-process plot.")
    parser.add_argument(
        "--model",
        choices=["bayesian-ridge", "ridge"],
        default="bayesian-ridge",
        help="Estimator used for the reproducible command-line baseline.",
    )
    parser.add_argument("--quick", action="store_true", help="Use a small subset for a fast smoke test.")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def make_model(name: str):
    if name == "bayesian-ridge":
        return MultiOutputRegressor(BayesianRidge(compute_score=True))
    if name == "ridge":
        return Ridge(alpha=1.0)
    raise ValueError(f"Unsupported model: {name}")


def get_prediction_band(model, X_valid: np.ndarray, y_valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(model, MultiOutputRegressor) and hasattr(model.estimators_[0], "predict"):
        means = []
        stds = []
        for estimator in model.estimators_:
            try:
                mean, std = estimator.predict(X_valid, return_std=True)
            except TypeError:
                mean = estimator.predict(X_valid)
                std = np.full_like(mean, np.nan, dtype=float)
            means.append(mean)
            stds.append(std)

        mean_pred = np.column_stack(means)
        std_pred = np.column_stack(stds)
        if np.isfinite(std_pred).all():
            return np.clip(mean_pred, 0, None), std_pred

    mean_pred = np.clip(model.predict(X_valid), 0, None)
    residual_std = np.std(y_valid - mean_pred, axis=0, keepdims=True)
    return mean_pred, np.repeat(residual_std, len(X_valid), axis=0)


def get_loss_history(model, y_train: np.ndarray, y_valid: np.ndarray, valid_pred: np.ndarray) -> np.ndarray:
    if isinstance(model, MultiOutputRegressor):
        histories = [
            np.asarray(estimator.scores_, dtype=float)
            for estimator in model.estimators_
            if hasattr(estimator, "scores_") and len(estimator.scores_) > 0
        ]
        if histories:
            min_len = min(len(history) for history in histories)
            score_history = np.vstack([history[:min_len] for history in histories]).mean(axis=0)
            loss_history = score_history.max() - score_history
            return loss_history + mean_squared_error(y_valid, valid_pred)

    baseline = np.repeat(y_train.mean(axis=0, keepdims=True), len(y_valid), axis=0)
    return np.array([mean_squared_error(y_valid, baseline), mean_squared_error(y_valid, valid_pred)])


def plot_training_process(
    y_valid: np.ndarray,
    mean_pred: np.ndarray,
    std_pred: np.ndarray,
    loss_history: np.ndarray,
    plot_path: Path,
) -> None:
    y_true = y_valid.reshape(-1)
    y_mean = mean_pred.reshape(-1)
    y_std = std_pred.reshape(-1)

    max_points = min(500, len(y_true))
    x = np.linspace(0, 1, max_points)
    y_true = y_true[:max_points]
    y_mean = y_mean[:max_points]
    y_std = y_std[:max_points]

    fig, axes = plt.subplots(2, 1, figsize=(8, 9), constrained_layout=True)

    axes[0].plot(x, y_mean, color="#1f77b4", linewidth=1.2, label="Mean Posterior Predictive")
    axes[0].fill_between(
        x,
        y_mean - 1.96 * y_std,
        y_mean + 1.96 * y_std,
        color="#1f77b4",
        alpha=0.18,
        label="Epistemic uncertainty",
    )
    axes[0].scatter(x, y_true, color="#ff7f0e", s=18, alpha=0.85, label="Actual Data Points")
    axes[0].set_title("Posterior Predictive")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.15)

    loss_x = np.arange(len(loss_history))
    axes[1].plot(loss_x, loss_history, color="#1f77b4", linewidth=1.4, label="Loss Function")
    axes[1].set_title("The Value of loss")
    axes[1].set_xlabel("Iteration")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.15)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    prepared = prepare_data(args.data_dir)
    X = flatten_daily_features(prepared.X_train)
    y = prepared.y_train
    X_test = flatten_daily_features(prepared.X_test)

    if args.quick:
        sample_size = min(240, len(X))
        X = X[:sample_size]
        y = y[:sample_size]
        X_test = X_test[: min(30, len(X_test))]
        test_template = prepared.test_template.iloc[: len(X_test)].copy()
    else:
        test_template = prepared.test_template

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=args.random_state
    )

    model = make_model(args.model)
    model.fit(X_train, y_train)

    valid_pred, valid_std = get_prediction_band(model, X_valid, y_valid)
    mse = mean_squared_error(y_valid, valid_pred)
    r2 = r2_score(y_valid, valid_pred)
    print(f"Validation MSE: {mse:.6f}")
    print(f"Validation R2: {r2:.6f}")

    test_pred = np.clip(model.predict(X_test), 0, None)
    submission = build_submission(test_template, test_pred)

    output_path = Path(args.output)
    model_path = Path(args.model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False, encoding="utf-8-sig")
    dump(model, model_path)
    if not args.no_plot:
        plot_path = Path(args.plot_path)
        loss_history = get_loss_history(model, y_train, y_valid, valid_pred)
        plot_training_process(y_valid, valid_pred, valid_std, loss_history, plot_path)
        print(f"Wrote training plot: {plot_path}")

    print(f"Wrote submission: {output_path}")
    print(f"Wrote model: {model_path}")


if __name__ == "__main__":
    main()
