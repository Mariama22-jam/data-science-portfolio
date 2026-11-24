"""Statistical error analysis and scaling fits."""
from __future__ import annotations

import numpy as np


def blocking_error(x: np.ndarray, min_blocks: int = 16) -> float:
    """Standard error of the mean of a correlated series (Flyvbjerg-Petersen blocking).

    The series is repeatedly halved by averaging neighbours; the error estimate
    grows until blocks are longer than the autocorrelation time and then plateaus.
    The maximum over levels with at least ``min_blocks`` blocks is returned, which
    is a conservative choice.
    """
    x = np.asarray(x, dtype=float)
    best = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    while len(x) >= 2 * min_blocks:
        n = len(x) // 2
        x = 0.5 * (x[: 2 * n : 2] + x[1 : 2 * n : 2])
        best = max(best, x.std(ddof=1) / np.sqrt(len(x)))
    return float(best)


def fit_flory_with_correction(n, r2, r2_err, delta: float = 0.528):
    """Fit <R^2> = A N^(2 nu) (1 + b N^(-delta)) with the 3D correction exponent fixed.

    Short chains bias a pure power-law fit upward; the leading correction-to-scaling
    term (delta ≈ 0.528, Clisby 2010) absorbs most of that bias.
    Returns (nu, nu_err, A, b).
    """
    from scipy.optimize import curve_fit

    n = np.asarray(n, float)

    def model(x, log_a, nu, b):
        return log_a + 2 * nu * np.log(x) + np.log1p(b * x ** (-delta))

    y = np.log(np.asarray(r2, float))
    sy = np.asarray(r2_err, float) / np.asarray(r2, float)
    p, cov = curve_fit(model, n, y, p0=[0.0, 0.59, 0.0], sigma=sy, absolute_sigma=True)
    resid = (y - model(n, *p)) / sy
    chi2_red = float(resid @ resid / max(len(n) - 3, 1))
    cov = cov * max(chi2_red, 1.0)
    return float(p[1]), float(np.sqrt(cov[1, 1])), float(np.exp(p[0])), float(p[2])


def fit_flory_exponent(n: np.ndarray, r2: np.ndarray, r2_err: np.ndarray | None = None):
    """Fit <R^2> = A * N^(2 nu) by weighted least squares in log-log space.

    ``n`` is the number of bonds (monomers - 1). Returns (nu, nu_err, A).
    """
    x = np.log(np.asarray(n, float))
    y = np.log(np.asarray(r2, float))
    if r2_err is None:
        w = np.ones_like(y)
    else:
        sy = np.asarray(r2_err, float) / np.asarray(r2, float)
        w = 1.0 / np.maximum(sy, 1e-12) ** 2
    W = np.diag(w)
    X = np.column_stack([np.ones_like(x), x])
    cov = np.linalg.inv(X.T @ W @ X)
    beta = cov @ X.T @ W @ y
    resid = y - X @ beta
    dof = max(len(y) - 2, 1)
    chi2_red = float(resid @ W @ resid / dof)
    cov = cov * max(chi2_red, 1.0)  # inflate when the fit is worse than the error bars
    slope, slope_err = beta[1], np.sqrt(cov[1, 1])
    return slope / 2.0, slope_err / 2.0, float(np.exp(beta[0]))
