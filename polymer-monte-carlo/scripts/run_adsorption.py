"""Adsorption transition of a grafted SAW on an attractive wall.

All eps values for one chain length are simulated together with replica exchange
(parallel tempering), with pivot, crankshaft, kink and end moves. Each N is run
twice, from a fully desorbed (upright) and a fully adsorbed (flat) start; the
largest difference between the two is reported as a convergence diagnostic and
half of it is added to the error bars as a systematic uncertainty.

For each chain length N and attraction eps (in k_B T) we measure the adsorbed
fraction f = <n_s>/N and the specific-heat-like fluctuation C = eps^2 Var(n_s) / N.

Near the critical adsorption point f ~ N^(phi - 1) with crossover exponent
phi ≈ 0.48–0.50, so curves of N^(1-phi) f for different N cross near eps_c.
Literature estimates for the simple cubic lattice give eps_c ≈ 0.28–0.29
(e.g. Grassberger 2005; Klushin et al. 2013). Finite-size estimates from short
chains are biased, so treat the estimate here as approximate.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from polymc import blocking_error, simulate_grafted_tempering  # noqa: E402

PHI = 0.5


def crossing_points(df: pd.DataFrame) -> list[float]:
    """eps where N^(1-phi) f(eps) curves of consecutive N intersect (linear interpolation)."""
    out = []
    ns = sorted(df.monomers.unique())
    for n1, n2 in zip(ns[:-1], ns[1:]):
        a = df[df.monomers == n1].sort_values("eps")
        b = df[df.monomers == n2].sort_values("eps")
        d = (b.scaled_f.values - a.scaled_f.values)
        e = a.eps.values
        for i in range(len(d) - 1):
            if d[i] < 0 <= d[i + 1] or d[i] > 0 >= d[i + 1]:
                t = d[i] / (d[i] - d[i + 1])
                out.append(float(e[i] + t * (e[i + 1] - e[i])))
                break
    return out


def main(quick: bool, seed: int) -> dict:
    if quick:
        lengths, sweeps = [16, 32, 64], 1500
        eps_grid = np.round(np.arange(0.0, 1.21, 0.1), 3)
    else:
        lengths, sweeps = [25, 50, 100, 200, 400], 10000
        eps_grid = np.round(np.arange(0.0, 1.201, 0.04), 3)
    rows, swaps, start_gap = [], {}, {}
    for i, n in enumerate(lengths):
        t0 = time.time()
        runs = {}
        n_sw = int(sweeps * max(1.0, n / 200))  # longer chains relax more slowly
        for st in ("upright", "flat"):
            res, swap = simulate_grafted_tempering(
                n, eps_grid, n_sweeps=n_sw, n_equil=n_sw // 4, seed=seed + 100 * i + (st == "flat"), start=st
            )
            runs[st] = res
            swaps[f"{n}_{st}"] = float(swap.min())
        gaps = []
        for ru, rf in zip(runs["upright"], runs["flat"]):
            fu, ff = ru["ns"].mean() / n, rf["ns"].mean() / n
            stat = 0.5 * np.hypot(blocking_error(ru["ns"]), blocking_error(rf["ns"])) / n
            ns = np.concatenate([ru["ns"], rf["ns"]])
            gaps.append(abs(fu - ff))
            rows.append(
                {
                    "monomers": n,
                    "eps": ru["eps"],
                    "f": 0.5 * (fu + ff),
                    # statistical error combined with half the start-to-start gap (systematic)
                    "f_err": float(np.hypot(stat, 0.5 * abs(fu - ff))),
                    "f_upright_start": fu,
                    "f_flat_start": ff,
                    # average of within-run variances; pooling the two runs would add the
                    # squared gap between their means and fake a peak
                    "heat": float(ru["eps"] ** 2 * 0.5 * (ru["ns"].var() + rf["ns"].var()) / n),
                    "rg2_par": 0.5 * (ru["rg2_par"].mean() + rf["rg2_par"].mean()),
                    "rg2_perp": 0.5 * (ru["rg2_perp"].mean() + rf["rg2_perp"].mean()),
                    "samples": len(ns),
                }
            )
        start_gap[int(n)] = float(max(gaps))
        print(f"N={n:4d} done in {time.time() - t0:.1f}s, max |f_upright - f_flat| = {max(gaps):.3f}", flush=True)
    df = pd.DataFrame(rows)
    df["scaled_f"] = df.f * df.monomers ** (1 - PHI)
    res = ROOT / "results"
    df.to_csv(res / "adsorption.csv", index=False)

    crosses = crossing_points(df)
    peaks = {int(n): float(g.loc[g.heat.idxmax(), "eps"]) for n, g in df.groupby("monomers")}
    summary = {
        "eps_c_from_crossings": crosses,
        "eps_c_estimate": float(np.mean(crosses)) if crosses else None,
        "specific_heat_peak_eps": peaks,
        "eps_c_literature_range": [0.28, 0.29],
        "min_swap_acceptance": swaps,
        "convergence_max_start_gap_by_N": start_gap,
        "method": "replica exchange over eps (parallel tempering)",
        "crossover_exponent_assumed": PHI,
        "quick_mode": quick,
    }
    (res / "adsorption_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for n, g in df.groupby("monomers"):
        axes[0].errorbar(g.eps, g.f, yerr=g.f_err, marker="o", ms=3, label=f"N={n}")
        axes[1].plot(g.eps, g.scaled_f, marker="o", ms=3, label=f"N={n}")
        axes[2].plot(g.eps, g.heat, marker="o", ms=3, label=f"N={n}")
    axes[0].set(xlabel="ε / k_BT", ylabel="adsorbed fraction f", title="Adsorbed fraction")
    axes[1].set(xlabel="ε / k_BT", ylabel=f"N^(1-φ) f,  φ={PHI}", title="Finite-size scaling")
    axes[2].set(xlabel="ε / k_BT", ylabel="ε² Var(n_s)/N", title="Fluctuations (specific heat)")
    for ax in axes:
        ax.axvspan(0.28, 0.29, color="grey", alpha=0.25, lw=0)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(res / "figures" / "adsorption_transition.png", dpi=150)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    main(a.quick, a.seed)
