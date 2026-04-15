from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from pv_fault_diagnosis.data import build_submission, flatten_daily_features, prepare_data
from pv_fault_diagnosis import gpr_model, pymc3_model
from pv_fault_diagnosis.visualization import plot_training_process


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce PV power prediction experiments.")
    parser.add_argument("--data-dir", default="data", help="Directory containing the competition CSV files.")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for submissions, models, and training-process figures.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip writing the training-process plot.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["pymc3", "gpr", "all"],
        default=["all"],
        help="Models to run. `pymc3` reproduces the PyMC3 Bayesian linear notebook path.",
    )
    parser.add_argument("--max-gpr-train-samples", type=int, default=80)
    parser.add_argument("--quick", action="store_true", help="Use a small subset for a fast smoke test.")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def selected_models(model_args: list[str]) -> list[str]:
    if "all" in model_args:
        return ["pymc3", "gpr"]
    return model_args


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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name in selected_models(args.models):
        if model_name == "pymc3":
            display_name = pymc3_model.DISPLAY_NAME
            slug = pymc3_model.SLUG
            model, valid_pred, valid_std, test_pred, loss_history = pymc3_model.fit_predict(
                X_train, y_train, X_valid, y_valid, X_test
            )
        elif model_name == "gpr":
            display_name = gpr_model.DISPLAY_NAME
            slug = gpr_model.SLUG
            model, valid_pred, valid_std, test_pred, loss_history = gpr_model.fit_predict(
                X_train,
                y_train,
                X_valid,
                y_valid,
                X_test,
                random_state=args.random_state,
                max_train_samples=args.max_gpr_train_samples,
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        mse = mean_squared_error(y_valid, valid_pred)
        r2 = r2_score(y_valid, valid_pred)
        print(f"[{display_name}] Validation MSE: {mse:.6f}")
        print(f"[{display_name}] Validation R2: {r2:.6f}")

        submission = build_submission(test_template, test_pred)
        output_path = output_dir / f"{slug}_submission.csv"
        model_path = output_dir / f"{slug}_model.joblib"
        plot_path = output_dir / f"{slug}_training_process.png"

        submission.to_csv(output_path, index=False, encoding="utf-8-sig")
        dump(model, model_path)
        if not args.no_plot:
            plot_training_process(y_valid, valid_pred, valid_std, loss_history, plot_path, display_name)
            print(f"[{display_name}] Wrote training plot: {plot_path}")

        print(f"[{display_name}] Wrote submission: {output_path}")
        print(f"[{display_name}] Wrote model: {model_path}")


if __name__ == "__main__":
    main()
