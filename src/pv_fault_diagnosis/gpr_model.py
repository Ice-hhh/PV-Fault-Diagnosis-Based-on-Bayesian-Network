from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_squared_error


DISPLAY_NAME = "Gaussian Process Regression"
SLUG = "gpr"


def build_model(random_state: int) -> GaussianProcessRegressor:
    kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=0,
        random_state=random_state,
    )


def fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    X_test: np.ndarray,
    random_state: int,
    max_train_samples: int,
) -> tuple[GaussianProcessRegressor, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_fit = X_train
    y_fit = y_train
    if len(X_fit) > max_train_samples:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(X_fit), size=max_train_samples, replace=False)
        X_fit = X_fit[indices]
        y_fit = y_fit[indices]

    model = build_model(random_state)
    model.fit(X_fit, y_fit)

    valid_pred, valid_std = model.predict(X_valid, return_std=True)
    valid_pred = np.clip(valid_pred, 0, None)
    test_pred = np.clip(model.predict(X_test), 0, None)
    loss_history = _diagnostic_loss_history(X_fit, y_fit, X_valid, y_valid, valid_pred, random_state)
    return model, valid_pred, valid_std, test_pred, loss_history


def _diagnostic_loss_history(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    valid_pred: np.ndarray,
    random_state: int,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    sizes = np.linspace(max(8, len(X_fit) // 10), len(X_fit), num=min(12, len(X_fit)), dtype=int)
    losses = []
    for size in np.unique(sizes):
        indices = rng.choice(len(X_fit), size=size, replace=False)
        probe = build_model(random_state)
        probe.optimizer = None
        probe.fit(X_fit[indices], y_fit[indices])
        pred = np.clip(probe.predict(X_valid), 0, None)
        losses.append(mean_squared_error(y_valid, pred))

    losses.append(mean_squared_error(y_valid, valid_pred))
    return np.asarray(losses, dtype=float)

