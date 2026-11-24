import numpy as np
import pytest

from polymc import (
    SYMMETRIES,
    blocking_error,
    fit_flory_exponent,
    is_connected,
    is_self_avoiding,
    simulate_free_chain,
    simulate_grafted_chain,
    simulate_grafted_tempering,
)
from polymc.exact import exact_free, exact_grafted


def test_symmetry_group():
    assert SYMMETRIES.shape == (47, 3, 3)
    for g in SYMMETRIES:
        assert np.array_equal(g @ g.T, np.eye(3, dtype=int))  # orthogonal
    assert len({g.tobytes() for g in SYMMETRIES}) == 47


def test_exact_counts():
    # known SAW counts on the simple cubic lattice: c_1..c_4 = 6, 30, 150, 726
    assert [exact_free(n + 1)["count"] for n in range(1, 5)] == [6, 30, 150, 726]


@pytest.mark.parametrize("n", [4, 6])
def test_free_chain_matches_exact_enumeration(n):
    exact = exact_free(n)
    out = simulate_free_chain(n, n_sweeps=60000, n_equil=500, seed=1)
    err = blocking_error(out["r2"])
    assert abs(out["r2"].mean() - exact["r2"]) < 5 * err + 0.02
    assert is_self_avoiding(out["chain"]) and is_connected(out["chain"])


@pytest.mark.parametrize("eps", [0.0, 0.8])
def test_grafted_chain_matches_exact_enumeration(eps):
    n = 6
    exact = exact_grafted(n, eps)
    out = simulate_grafted_chain(n, eps, n_sweeps=60000, n_equil=500, seed=2)
    err = blocking_error(out["ns"])
    assert abs(out["ns"].mean() - exact["ns"]) < 5 * err + 0.02


def test_moves_preserve_constraints_long_chain():
    out = simulate_grafted_chain(64, 0.5, n_sweeps=300, n_equil=0, seed=3)
    c = out["chain"]
    assert is_self_avoiding(c) and is_connected(c)
    assert c[:, 2].min() >= 1 and tuple(c[0]) == (0, 0, 1)


def test_flory_fit_recovers_known_exponent():
    n = np.array([10, 20, 40, 80, 160])
    r2 = 1.2 * n ** (2 * 0.588)
    nu, nu_err, a = fit_flory_exponent(n, r2)
    assert abs(nu - 0.588) < 1e-9 and abs(a - 1.2) < 1e-9


def test_blocking_error_larger_for_correlated_series():
    rng = np.random.default_rng(0)
    white = rng.normal(size=4096)
    ar = np.zeros(4096)
    for i in range(1, 4096):
        ar[i] = 0.95 * ar[i - 1] + white[i]
    naive = ar.std(ddof=1) / np.sqrt(len(ar))
    assert blocking_error(ar) > 2 * naive


def test_replica_exchange_matches_exact_enumeration_at_every_eps():
    n, grid = 6, [0.0, 0.4, 0.8, 1.2, 1.6]
    res, swap = simulate_grafted_tempering(n, grid, n_sweeps=40000, n_equil=500, seed=4)
    assert np.all(swap > 0.05)
    for r in res:
        exact = exact_grafted(n, r["eps"])
        err = blocking_error(r["ns"])
        assert abs(r["ns"].mean() - exact["ns"]) < 5 * err + 0.02, r["eps"]


def test_replica_exchange_adsorbs_long_chain_at_strong_attraction():
    # single-eps runs from an upright rod get trapped here; tempering must not
    res, _ = simulate_grafted_tempering(100, np.linspace(0, 1.5, 16), n_sweeps=1500, n_equil=500, seed=5)
    f = np.array([r["ns"].mean() / 100 for r in res])
    assert f[-1] > 0.85
    assert np.all(np.diff(f) > -0.03)  # adsorbed fraction increases with attraction
