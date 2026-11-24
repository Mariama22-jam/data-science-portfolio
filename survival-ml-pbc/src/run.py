"""Full survival-ML pipeline on PBC.

1. Repeated stratified K-fold CV comparison of eight models (C-index, Uno's C,
   integrated Brier score, 5-year time-dependent AUC), with intervals.
2. Full-data Cox model: hazard ratios with 95% CIs and Schoenfeld-residual
   tests of proportional hazards.
3. 5-year calibration from out-of-fold predictions.
4. Figures and a markdown summary in results/.

Usage:  python src/run.py            # full run (10 x 5-fold)
        python src/run.py --quick    # CI smoke run (2 x 5-fold, smaller grids)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import BINARY, LOG_VARS, ORDINAL, NUMERIC, ROOT, features, load_processed, target  # noqa: E402
from evaluate import HORIZON, fold_metrics  # noqa: E402
from models import make_models  # noqa: E402

RES = ROOT / "results"
FIG = RES / "figures"
METRICS = {"harrell_c": "Harrell's C", "uno_c": "Uno's C", "ibs": "Integrated Brier score", "auc_5y": "5-year AUC"}


def _one_fold(name, model, X, y, tr, te, rep, fold):
    warnings.filterwarnings("ignore")
    m = clone(model).fit(X.iloc[tr], y[tr])
    met, s5 = fold_metrics(m, X.iloc[te], y[tr], y[te])
    met.update({"model": name, "repeat": rep, "fold": fold})
    return met, (name, rep, te, s5)


def cross_validate(X, y, quick, seed, n_jobs):
    n_rep = 2 if quick else 10
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep, random_state=seed)
    splits = list(cv.split(X, y["event"]))
    models = make_models(seed, quick)
    tasks = []
    for i, (tr, te) in enumerate(splits):
        for name, model in models.items():
            tasks.append(delayed(_one_fold)(name, model, X, y, tr, te, i // 5, i % 5))
    out = Parallel(n_jobs=n_jobs, verbose=0)(tasks)
    scores = pd.DataFrame([o[0] for o in out])
    oof = {}
    for name, rep, te, s5 in (o[1] for o in out):
        if rep == 0:
            arr = oof.setdefault(name, np.full(len(X), np.nan))
            arr[te] = s5
    return scores, oof, list(models)


def summarise(scores: pd.DataFrame, order) -> pd.DataFrame:
    rows = []
    cox = scores[scores.model == "Cox PH"].set_index(["repeat", "fold"])
    for name in order:
        s = scores[scores.model == name]
        row = {"model": name}
        for k in METRICS:
            rep_means = s.groupby("repeat")[k].mean()
            row[f"{k}_mean"] = s[k].mean()
            row[f"{k}_sd_folds"] = s[k].std(ddof=1)
            row[f"{k}_lo"] = np.percentile(rep_means, 2.5) if len(rep_means) > 2 else np.nan
            row[f"{k}_hi"] = np.percentile(rep_means, 97.5) if len(rep_means) > 2 else np.nan
        diff = s.set_index(["repeat", "fold"])["harrell_c"] - cox["harrell_c"]
        row["delta_c_vs_cox"] = diff.mean()
        row["share_folds_better_than_cox"] = (diff > 0).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def full_data_cox(df: pd.DataFrame):
    """Unpenalised Cox model on all 312 patients for interpretation (not for prediction)."""
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test

    X = features(df)
    X = X.fillna(X.median())
    d = X.assign(time=df.time_days.values, event=df.death.astype(int).values)
    cph = CoxPHFitter().fit(d, duration_col="time", event_col="event")
    s = cph.summary
    hr = pd.DataFrame(
        {
            "variable": [f"log({c})" if c in LOG_VARS else c for c in s.index],
            "coef": s["coef"].values,
            "HR": s["exp(coef)"].values,
            "HR_95_lo": s["exp(coef) lower 95%"].values,
            "HR_95_hi": s["exp(coef) upper 95%"].values,
            "p": s["p"].values,
        }
    )
    assert ((hr.HR >= hr.HR_95_lo) & (hr.HR <= hr.HR_95_hi)).all(), "CI must contain the estimate"
    ph = proportional_hazard_test(cph, d, time_transform="rank").summary
    ph = ph.reset_index().rename(columns={"index": "variable", "test_statistic": "chi2"})
    ph = ph[["variable", "chi2", "p"]]
    # global test: sum of per-covariate statistics is only approximate, so report the minimum p
    # (Holm-adjusted) instead
    m = len(ph)
    ph["p_holm"] = np.minimum(1.0, np.maximum.accumulate(np.sort(ph.p.values) * (m - np.arange(m))))[
        np.argsort(np.argsort(ph.p.values))
    ]
    return hr, ph, cph.concordance_index_


def lasso_path(df):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sksurv.linear_model import CoxnetSurvivalAnalysis

    X = features(df)
    Xs = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X))
    m = CoxnetSurvivalAnalysis(l1_ratio=1.0, alpha_min_ratio=0.005, max_iter=200000).fit(Xs, target(df))
    return pd.DataFrame(m.coef_, index=X.columns, columns=m.alphas_)


def calibration_table(df, s5_pred, n_groups=5):
    from sksurv.nonparametric import kaplan_meier_estimator

    risk = 1 - s5_pred
    q = pd.qcut(risk, n_groups, labels=False, duplicates="drop")
    rows = []
    for g in np.unique(q):
        idx = q == g
        t, s = kaplan_meier_estimator(df.death.values[idx], df.time_days.values[idx])
        obs = s[t <= HORIZON][-1] if (t <= HORIZON).any() else 1.0
        rows.append({"group": int(g), "n": int(idx.sum()), "predicted_risk": risk[idx].mean(), "observed_risk": 1 - obs})
    return pd.DataFrame(rows)


def make_figures(df, scores, order, oof, path, ph):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sksurv.nonparametric import kaplan_meier_estimator

    short = {n: n.replace(" (CoxBoost-like)", "\n(CoxBoost-like)").replace(" (no covariates)", "\n(null)") for n in order}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, k in zip(axes, ["harrell_c", "ibs"]):
        data = [scores[scores.model == n][k].values for n in order]
        ax.boxplot(data, orientation="horizontal", tick_labels=[short[n] for n in order])
        ax.set_xlabel(METRICS[k] + ("  (lower is better)" if k == "ibs" else "  (higher is better)"))
        ax.invert_yaxis()
    fig.suptitle("Repeated 5-fold CV on PBC (each point = one test fold)")
    fig.tight_layout()
    fig.savefig(FIG / "cv_model_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name in ["Cox PH", "LASSO Cox", "Random survival forest", "Componentwise boosting (CoxBoost-like)"]:
        if name in oof:
            c = calibration_table(df, oof[name])
            ax.plot(c.predicted_risk, c.observed_risk, "o-", label=name.replace(" (CoxBoost-like)", ""))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="Predicted 5-year risk of death", ylabel="Observed (Kaplan-Meier) 5-year risk",
           title="Calibration at 5 years (out-of-fold, quintiles)", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "calibration_5y.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for v in path.index:
        ax.plot(path.columns, path.loc[v], label=v)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set(xlabel="penalty α (log scale)", ylabel="standardised coefficient", title="LASSO Cox coefficient path")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "lasso_path.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for st in sorted(df.stage.dropna().unique()):
        sub = df[df.stage == st]
        t, s = kaplan_meier_estimator(sub.death.values, sub.time_days.values)
        ax.step(t / 365.25, s, where="post", label=f"stage {int(st)} (n={len(sub)})")
    ax.set(xlabel="years", ylabel="survival probability", title="Kaplan-Meier by histologic stage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "km_by_stage.png", dpi=150)
    plt.close(fig)


def write_markdown(summary, hr, ph, cindex_full, n_rep, quick, seconds, df):
    def fmt(r, k, digits=3):
        lo, hi = r[f"{k}_lo"], r[f"{k}_hi"]
        ci = f" [{lo:.{digits}f}, {hi:.{digits}f}]" if not np.isnan(lo) else ""
        return f"{r[f'{k}_mean']:.{digits}f}{ci}"

    lines = [
        "# Results: survival ML on PBC",
        "",
        f"*Generated automatically by `src/run.py`{' in quick mode (smoke test, not for reporting)' if quick else ''}. "
        f"Runtime {seconds/60:.1f} min.*",
        "",
        f"Data: {len(df)} randomised patients, {int(df.death.sum())} deaths, {int(df.transplant.sum())} transplants (censored).",
        f"Validation: {n_rep} × repeated stratified 5-fold CV ({n_rep*5} test folds per model). "
        "Brackets give the 2.5–97.5 percentile range of the per-repeat mean, which reflects "
        "the variability due to the random split.",
        "",
        "| Model | Harrell's C | Uno's C | Integrated Brier | 5-year AUC | ΔC vs Cox | folds better than Cox |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r.model} | {fmt(r,'harrell_c')} | {fmt(r,'uno_c')} | {fmt(r,'ibs')} | {fmt(r,'auc_5y')} | "
            f"{r.delta_c_vs_cox:+.3f} | {r.share_folds_better_than_cox:.0%} |"
        )
    lines += [
        "",
        "## Full-data Cox model (interpretation only)",
        "",
        f"Apparent (in-sample) C-index: {cindex_full:.3f}, which is optimistic compared with the cross-validated value above.",
        "",
        "| Variable | HR | 95% CI | p |",
        "|---|---|---|---|",
    ]
    for _, r in hr.iterrows():
        lines.append(f"| {r.variable} | {r.HR:.2f} | {r.HR_95_lo:.2f}–{r.HR_95_hi:.2f} | {r.p:.3g} |")
    lines += [
        "",
        "## Proportional-hazards check (Schoenfeld residuals, rank time transform)",
        "",
        "| Variable | χ² | p | Holm-adjusted p |",
        "|---|---|---|---|",
    ]
    for _, r in ph.iterrows():
        lines.append(f"| {r.variable} | {r.chi2:.2f} | {r.p:.3g} | {r.p_holm:.3g} |")
    viol = ph[ph.p < 0.05].variable.tolist()
    lines += [
        "",
        ("Variables with unadjusted p < 0.05: " + ", ".join(viol) + ". Consider stratifying on these "
         "or adding a time interaction before interpreting their hazard ratios.") if viol
        else "No covariate shows evidence against proportional hazards at the 5% level.",
        "",
    ]
    (RES / "summary.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    t0 = time.time()
    FIG.mkdir(parents=True, exist_ok=True)
    np.random.seed(a.seed)

    df = load_processed()
    X, y = features(df), target(df)
    scores, oof, order = cross_validate(X, y, a.quick, a.seed, a.jobs)
    scores.to_csv(RES / "cv_scores_per_fold.csv", index=False)
    summary = summarise(scores, order)
    summary.to_csv(RES / "cv_summary.csv", index=False)

    hr, ph, c_full = full_data_cox(df)
    hr.to_csv(RES / "cox_hazard_ratios.csv", index=False)
    ph.to_csv(RES / "ph_test.csv", index=False)
    path = lasso_path(df)
    calib = {n: calibration_table(df, oof[n]).to_dict("records") for n in oof if n != "Kaplan-Meier (no covariates)"}
    (RES / "calibration_5y.json").write_text(json.dumps(calib, indent=2))
    make_figures(df, scores, order, oof, path, ph)
    n_rep = int(scores["repeat"].nunique())
    write_markdown(summary, hr, ph, c_full, n_rep, a.quick, time.time() - t0, df)
    print((RES / "summary.md").read_text())


if __name__ == "__main__":
    main()
