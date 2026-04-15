from __future__ import annotations

import argparse
from pathlib import Path

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
        return MultiOutputRegressor(BayesianRidge())
    if name == "ridge":
        return Ridge(alpha=1.0)
    raise ValueError(f"Unsupported model: {name}")


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

    valid_pred = np.clip(model.predict(X_valid), 0, None)
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

    print(f"Wrote submission: {output_path}")
    print(f"Wrote model: {model_path}")


if __name__ == "__main__":
    main()
