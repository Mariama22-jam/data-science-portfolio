"""Metrics computed on one outer test fold."""
from __future__ import annotations

import numpy as np
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)

DAYS_PER_YEAR = 365.25
HORIZON = 5 * DAYS_PER_YEAR  # 5-year calibration / AUC horizon
TAU = 10 * DAYS_PER_YEAR  # truncation time for Uno's C


def time_grid(y_train, y_test):
    """Evaluation times inside the follow-up range of both folds (needed for IPCW)."""
    hi = min(y_test["time"].max(), y_train["time"].max(), TAU)
    return np.arange(DAYS_PER_YEAR, hi * 0.99, DAYS_PER_YEAR / 4)


def _surv_matrix(model, X, times):
    fns = model.predict_survival_function(X)
    return np.vstack([f(times) for f in fns])


def fold_metrics(model, X_test, y_train, y_test) -> tuple[dict, np.ndarray]:
    model = getattr(model, "best_estimator_", model)  # unwrap GridSearchCV
    risk = np.asarray(model.predict(X_test), dtype=float)
    times = time_grid(y_train, y_test)
    surv = _surv_matrix(model, X_test, times)
    s5 = _surv_matrix(model, X_test, np.array([HORIZON]))[:, 0]

    m = {
        "harrell_c": concordance_index_censored(y_test["event"], y_test["time"], risk)[0],
        "uno_c": concordance_index_ipcw(y_train, y_test, risk, tau=min(TAU, times.max()))[0],
        "ibs": integrated_brier_score(y_train, y_test, surv, times),
    }
    if np.ptp(risk) > 0:
        m["auc_5y"] = float(cumulative_dynamic_auc(y_train, y_test, risk, np.array([HORIZON]))[0][0])
    else:
        m["auc_5y"] = 0.5
    return m, s5
