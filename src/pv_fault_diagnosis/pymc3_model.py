from __future__ import annotations

import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor


DISPLAY_NAME = "PyMC3 Bayesian Linear"
SLUG = "pymc3"


def build_model() -> MultiOutputRegressor:
    return MultiOutputRegressor(BayesianRidge(compute_score=True))


def fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    X_test: np.ndarray,
) -> tuple[MultiOutputRegressor, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model = build_model()
    model.fit(X_train, y_train)

    means = []
    stds = []
    for estimator in model.estimators_:
        mean, std = estimator.predict(X_valid, return_std=True)
        means.append(mean)
        stds.append(std)

    valid_pred = np.clip(np.column_stack(means), 0, None)
    valid_std = np.column_stack(stds)
    test_pred = np.clip(model.predict(X_test), 0, None)
    loss_history = _loss_history(model, y_valid, valid_pred)
    return model, valid_pred, valid_std, test_pred, loss_history


def _loss_history(
    model: MultiOutputRegressor,
    y_valid: np.ndarray,
    valid_pred: np.ndarray,
) -> np.ndarray:
    histories = [
        np.asarray(estimator.scores_, dtype=float)
        for estimator in model.estimators_
        if hasattr(estimator, "scores_") and len(estimator.scores_) > 0
    ]
    if not histories:
        return np.array([mean_squared_error(y_valid, valid_pred)])

    min_len = min(len(history) for history in histories)
    score_history = np.vstack([history[:min_len] for history in histories]).mean(axis=0)
    loss_history = score_history.max() - score_history
    return loss_history + mean_squared_error(y_valid, valid_pred)

