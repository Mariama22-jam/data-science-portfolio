import numpy as np
from sklearn.model_selection import train_test_split

from data import features, load_processed, target
from evaluate import fold_metrics
from models import make_models


def test_analysis_set_matches_literature():
    df = load_processed()
    assert len(df) == 312 and df.id.is_unique
    assert int(df.death.sum()) == 125 and int(df.transplant.sum()) == 19
    assert not (df.death & df.transplant).any()


def test_features_exclude_outcome_columns():
    X = features(load_processed())
    assert not {"time_days", "death", "transplant", "id"} & set(X.columns)


def test_models_fit_and_null_model_is_uninformative():
    df = load_processed()
    X, y = features(df), target(df)
    tr, te = train_test_split(np.arange(len(df)), test_size=0.3, random_state=0, stratify=y["event"])
    models = make_models(seed=0, quick=True)
    null = models["Kaplan-Meier (no covariates)"].fit(X.iloc[tr], y[tr])
    m_null, _ = fold_metrics(null, X.iloc[te], y[tr], y[te])
    assert abs(m_null["harrell_c"] - 0.5) < 1e-9
    cox = models["Cox PH"].fit(X.iloc[tr], y[tr])
    m_cox, s5 = fold_metrics(cox, X.iloc[te], y[tr], y[te])
    assert m_cox["harrell_c"] > 0.75  # PBC is a strong-signal dataset
    assert m_cox["ibs"] < m_null["ibs"]
    assert np.all((s5 >= 0) & (s5 <= 1))
