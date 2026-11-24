import numpy as np

from run import LEAKY, TARGET, calibration_slope_intercept, load, make_models


def test_dataset():
    df = load()
    assert df.shape == (299, 13) and int(df[TARGET].sum()) == 96


def test_calibration_of_perfect_probabilities():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 20000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    slope, intercept = calibration_slope_intercept(y, p)
    assert abs(slope - 1) < 0.08 and abs(intercept) < 0.08


def test_models_never_see_time_column():
    df = load()
    X = df.drop(columns=[TARGET, LEAKY])
    m = make_models(0, quick=True)["Logistic regression"].fit(X, df[TARGET])
    assert LEAKY not in m.feature_names_in_
