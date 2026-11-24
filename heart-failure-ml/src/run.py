"""Heart-failure mortality: classical vs modern ML with honest validation.

Data: Chicco & Jurman (2020), "Machine learning can predict survival of patients
with heart failure from serum creatinine and ejection fraction alone",
BMC Medical Informatics and Decision Making 20:16. 299 patients, UCI ML
Repository dataset 519, licence CC BY 4.0.

Target: DEATH_EVENT during follow-up.

Important: the column ``time`` (follow-up days) is excluded. Patients who die
have short follow-up by construction, so using it leaks the outcome. The script
quantifies that leak in a separate "with leakage" run so the effect is visible.

Models: logistic regression (baseline), gradient boosting (HistGradientBoosting),
random forest, and a small MLP. Evaluation: repeated stratified 5-fold CV with
ROC AUC, PR AUC, Brier score, log loss and calibration slope/intercept; SHAP
values for the boosting model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "heart_failure_clinical_records_dataset.csv"
SOURCE_URL = "https://archive.ics.uci.edu/static/public/519/heart+failure+clinical+records.zip"
SHA256 = "9c73cea7468ff5d517801ec050fe9993da5912fce4b56f296f8df3b38dd75912"
RES = ROOT / "results"
FIG = RES / "figures"
TARGET = "DEATH_EVENT"
LEAKY = "time"
SKEWED = ["creatinine_phosphokinase", "platelets", "serum_creatinine"]


def load() -> pd.DataFrame:
    if not RAW.exists():
        import io
        import urllib.request
        import zipfile

        RAW.parent.mkdir(parents=True, exist_ok=True)
        blob = urllib.request.urlopen(SOURCE_URL).read()
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            z.extract(RAW.name, RAW.parent)
    if hashlib.sha256(RAW.read_bytes()).hexdigest() != SHA256:
        print("warning: dataset checksum differs from the pinned version")
    return pd.read_csv(RAW)


def _log_skewed(X):
    X = X.copy()
    for c in SKEWED:
        if c in X:
            X[c] = np.log(X[c])
    return X


def make_models(seed: int, quick: bool) -> dict:
    pre = [("log", FunctionTransformer(_log_skewed)), ("scale", StandardScaler())]
    return {
        "Logistic regression": Pipeline(pre + [("m", LogisticRegression(C=1.0, max_iter=5000))]),
        "Gradient boosting": Pipeline(
            [("m", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=100 if quick else 200,
                                                  l2_regularization=1.0, min_samples_leaf=10, random_state=seed))]
        ),
        "Random forest": Pipeline(
            [("m", RandomForestClassifier(n_estimators=200 if quick else 500, min_samples_leaf=5, random_state=seed))]
        ),
        # weight decay chosen by inner 3-fold CV on log loss; without it the net overfits badly
        "Small neural net (MLP)": GridSearchCV(
            Pipeline(pre + [("m", MLPClassifier(hidden_layer_sizes=(8,), max_iter=3000, random_state=seed))]),
            {"m__alpha": [1.0, 3.0, 10.0] if quick else [0.3, 1.0, 3.0, 10.0, 30.0]},
            scoring="neg_log_loss", cv=StratifiedKFold(3, shuffle=True, random_state=seed),
        ),
    }


def calibration_slope_intercept(y, p):
    """Logistic recalibration: logit(P(y=1)) = a + b * logit(p). Ideal a=0, b=1."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(lp, y)
    slope = float(lr.coef_[0, 0])
    # intercept with slope fixed at 1 (calibration-in-the-large)
    from scipy.optimize import minimize_scalar

    def nll(a):
        q = 1 / (1 + np.exp(-(a + lp[:, 0])))
        return -np.sum(y * np.log(q) + (1 - y) * np.log(1 - q))

    intercept = float(minimize_scalar(nll, bounds=(-5, 5), method="bounded").x)
    return slope, intercept


def _fold(name, model, X, y, tr, te, rep, fold):
    warnings.filterwarnings("ignore")
    m = clone(model).fit(X.iloc[tr], y[tr])
    p = m.predict_proba(X.iloc[te])[:, 1]
    yt = y[te]
    slope, intercept = calibration_slope_intercept(yt, p)
    return {
        "model": name, "repeat": rep, "fold": fold,
        "roc_auc": roc_auc_score(yt, p), "pr_auc": average_precision_score(yt, p),
        "brier": brier_score_loss(yt, p), "log_loss": log_loss(yt, p),
        "calib_slope": slope, "calib_intercept": intercept,
    }, (name, rep, te, p)


def cross_validate(X, y, seed, quick, jobs):
    n_rep = 3 if quick else 20
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep, random_state=seed)
    models = make_models(seed, quick)
    tasks = [delayed(_fold)(n, m, X, y, tr, te, i // 5, i % 5)
             for i, (tr, te) in enumerate(cv.split(X, y)) for n, m in models.items()]
    out = Parallel(n_jobs=jobs)(tasks)
    scores = pd.DataFrame([o[0] for o in out])
    oof = {}
    for name, rep, te, p in (o[1] for o in out):
        if rep == 0:
            oof.setdefault(name, np.full(len(X), np.nan))[te] = p
    return scores, oof, list(models)


def summarise(scores, order):
    rows = []
    for n in order:
        s = scores[scores.model == n]
        r = {"model": n}
        for k in ["roc_auc", "pr_auc", "brier", "log_loss", "calib_slope", "calib_intercept"]:
            rm = s.groupby("repeat")[k].mean()
            r[f"{k}_mean"] = s[k].mean()
            r[f"{k}_lo"] = np.percentile(rm, 2.5)
            r[f"{k}_hi"] = np.percentile(rm, 97.5)
        rows.append(r)
    return pd.DataFrame(rows)


def shap_analysis(X, y, seed):
    import shap

    model = make_models(seed, quick=False)["Gradient boosting"].fit(X, y)
    est = model.named_steps["m"]
    explainer = shap.TreeExplainer(est)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    imp = pd.Series(np.abs(sv).mean(0), index=X.columns).sort_values(ascending=False)
    return sv, imp


def figures(df, X, y, scores, order, oof, sv, imp, leak):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, k, lab in [(axes[0], "roc_auc", "ROC AUC (higher is better)"), (axes[1], "brier", "Brier score (lower is better)")]:
        ax.boxplot([scores[scores.model == n][k] for n in order], orientation="horizontal", tick_labels=order)
        ax.set_xlabel(lab)
        ax.invert_yaxis()
    fig.suptitle("Repeated 5-fold CV, heart-failure mortality (time column excluded)")
    fig.tight_layout()
    fig.savefig(FIG / "cv_model_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for n in order:
        p = oof[n]
        bins = pd.qcut(p, 5, labels=False, duplicates="drop")
        g = pd.DataFrame({"p": p, "y": y, "b": bins}).groupby("b").mean()
        ax.plot(g.p, g.y, "o-", label=n)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="predicted probability of death", ylabel="observed proportion",
           title="Calibration (out-of-fold, quintiles)", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "calibration.png", dpi=150)
    plt.close(fig)

    plt.figure()
    shap.summary_plot(sv, X, show=False, plot_size=(7, 4.5))
    plt.title("SHAP values, gradient boosting (fitted on all data)")
    plt.tight_layout()
    plt.savefig(FIG / "shap_summary.png", dpi=150)
    plt.close("all")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(["without 'time'", "with 'time' (leak)"], [leak["clean"], leak["leaky"]], color=["#4c72b0", "#c44e52"])
    for i, v in enumerate([leak["clean"], leak["leaky"]]):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center")
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("mean CV ROC AUC (logistic regression)")
    ax.set_title("Why the follow-up time column must be excluded")
    fig.tight_layout()
    fig.savefig(FIG / "leakage_demo.png", dpi=150)
    plt.close(fig)


def write_md(summary, imp, leak, n_rep, quick, seconds, df):
    f = lambda r, k: f"{r[k+'_mean']:.3f} [{r[k+'_lo']:.3f}, {r[k+'_hi']:.3f}]"  # noqa: E731
    L = [
        "# Results: heart-failure mortality",
        "",
        f"*Generated automatically by `src/run.py`{' in quick mode (smoke test, not for reporting)' if quick else ''}. "
        f"Runtime {seconds/60:.1f} min.*",
        "",
        f"Data: {len(df)} patients, {int(df[TARGET].sum())} deaths. "
        f"Validation: {n_rep} × repeated stratified 5-fold CV. Brackets are the 2.5–97.5 percentile "
        "range of the per-repeat mean (split variability).",
        "",
        "| Model | ROC AUC | PR AUC | Brier | Log loss | Calibration slope | Calibration intercept |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        L.append(f"| {r.model} | {f(r,'roc_auc')} | {f(r,'pr_auc')} | {f(r,'brier')} | {f(r,'log_loss')} | "
                 f"{f(r,'calib_slope')} | {f(r,'calib_intercept')} |")
    L += [
        "",
        f"Prevalence of death is {df[TARGET].mean():.3f}, so a PR AUC near that value means no skill.",
        "A calibration slope below 1 means predictions are too extreme (overfitting); above 1, too timid.",
        "",
        "## Leakage check",
        "",
        f"Logistic regression mean ROC AUC without `time`: **{leak['clean']:.3f}**; with `time`: "
        f"**{leak['leaky']:.3f}**. The jump comes from follow-up length, which is known only after "
        "the outcome, so models using it are not usable at the point of care.",
        "",
        "## Most important features (mean |SHAP|, gradient boosting)",
        "",
        "| Feature | mean abs SHAP |",
        "|---|---|",
    ] + [f"| {k} | {v:.3f} |" for k, v in imp.items()] + [""]
    (RES / "summary.md").write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    t0 = time.time()
    FIG.mkdir(parents=True, exist_ok=True)
    df = load()
    y = df[TARGET].values
    X = df.drop(columns=[TARGET, LEAKY])
    scores, oof, order = cross_validate(X, y, a.seed, a.quick, a.jobs)
    scores.to_csv(RES / "cv_scores_per_fold.csv", index=False)
    summary = summarise(scores, order)
    summary.to_csv(RES / "cv_summary.csv", index=False)

    Xl = df.drop(columns=[TARGET])
    lr = {"Logistic regression": make_models(a.seed, True)["Logistic regression"]}
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3 if a.quick else 10, random_state=a.seed)
    leak = {}
    for tag, XX in [("clean", X), ("leaky", Xl)]:
        aucs = [roc_auc_score(y[te], clone(lr["Logistic regression"]).fit(XX.iloc[tr], y[tr]).predict_proba(XX.iloc[te])[:, 1])
                for tr, te in cv.split(XX, y)]
        leak[tag] = float(np.mean(aucs))

    sv, imp = shap_analysis(X, y, a.seed)
    imp.to_csv(RES / "shap_importance.csv", header=["mean_abs_shap"])
    (RES / "leakage_check.json").write_text(json.dumps(leak, indent=2))
    figures(df, X, y, scores, order, oof, sv, imp, leak)
    write_md(summary, imp, leak, int(scores["repeat"].nunique()), a.quick, time.time() - t0, df)
    print((RES / "summary.md").read_text())


if __name__ == "__main__":
    main()
