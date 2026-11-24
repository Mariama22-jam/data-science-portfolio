"""Model zoo. Every model is a scikit-learn Pipeline: median imputation -> scaling -> estimator.

Imputation and scaling are fitted inside each training fold, so no information
from the test fold leaks into preprocessing. Penalised models tune their
penalty with an inner cross-validation on the training fold only.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import (
    ComponentwiseGradientBoostingSurvivalAnalysis,
    GradientBoostingSurvivalAnalysis,
    RandomSurvivalForest,
)
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.nonparametric import kaplan_meier_estimator


class KaplanMeierNull(BaseEstimator):
    """Covariate-free reference model: everyone gets the training Kaplan-Meier curve."""

    def fit(self, X, y):
        self.time_, self.surv_ = kaplan_meier_estimator(y["event"], y["time"])
        return self

    def predict(self, X):
        return np.zeros(len(X))

    def predict_survival_function(self, X, return_array=False):
        from sksurv.functions import StepFunction

        f = StepFunction(np.r_[0.0, self.time_], np.r_[1.0, self.surv_])
        return np.array([f] * len(X))


def _pipe(est):
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", est)])


def _tuned(est, grid, seed):
    inner = KFold(n_splits=3, shuffle=True, random_state=seed)
    return GridSearchCV(_pipe(est), {f"model__{k}": v for k, v in grid.items()}, cv=inner, n_jobs=1, error_score=np.nan)


def make_models(seed: int = 0, quick: bool = False) -> dict:
    alphas = np.logspace(-2.5, -0.5, 5 if quick else 9)
    return {
        "Kaplan-Meier (no covariates)": _pipe(KaplanMeierNull()),
        "Cox PH": _pipe(CoxPHSurvivalAnalysis(alpha=1e-4, ties="efron")),
        "Ridge Cox": _tuned(CoxPHSurvivalAnalysis(ties="efron"), {"alpha": np.logspace(-1, 3, 5 if quick else 9)}, seed),
        "LASSO Cox": _tuned(
            CoxnetSurvivalAnalysis(l1_ratio=1.0, fit_baseline_model=True, max_iter=200000),
            {"alphas": [[a] for a in alphas]},
            seed,
        ),
        "Elastic-net Cox": _tuned(
            CoxnetSurvivalAnalysis(l1_ratio=0.5, fit_baseline_model=True, max_iter=200000),
            {"alphas": [[a] for a in alphas]},
            seed,
        ),
        "Componentwise boosting (CoxBoost-like)": _tuned(
            ComponentwiseGradientBoostingSurvivalAnalysis(loss="coxph", learning_rate=0.1, random_state=seed),
            {"n_estimators": [50, 150, 400] if not quick else [50, 200]},
            seed,
        ),
        "Gradient-boosted trees": _pipe(
            GradientBoostingSurvivalAnalysis(
                n_estimators=150 if quick else 300, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=seed
            )
        ),
        "Random survival forest": _pipe(
            RandomSurvivalForest(
                n_estimators=150 if quick else 400, min_samples_leaf=10, max_features="sqrt", n_jobs=1, random_state=seed
            )
        ),
    }
