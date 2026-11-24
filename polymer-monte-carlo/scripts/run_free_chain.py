"""Validate the sampler on free SAWs: measure the Flory exponent nu from <R^2> ~ N^(2 nu).

The accepted 3D value is nu = 0.5876 (Clisby 2010). Short chains carry corrections to
scaling, so the fit uses the larger chain lengths and the result is reported with
that caveat.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from polymc import blocking_error, fit_flory_exponent, simulate_free_chain  # noqa: E402
from polymc.analysis import fit_flory_with_correction  # noqa: E402

NU_REFERENCE = 0.587597


def main(quick: bool, seed: int) -> dict:
    if quick:
        lengths, sweeps = [16, 32, 64, 128], 1500
    else:
        lengths, sweeps = [16, 32, 64, 128, 256, 512], 6000
    rows = []
    for i, n in enumerate(lengths):
        t0 = time.time()
        out = simulate_free_chain(n, n_sweeps=sweeps, n_equil=sweeps // 10, seed=seed + i)
        rows.append(
            {
                "monomers": n,
                "bonds": n - 1,
                "r2_mean": out["r2"].mean(),
                "r2_err": blocking_error(out["r2"]),
                "rg2_mean": out["rg2"].mean(),
                "rg2_err": blocking_error(out["rg2"]),
                "pivot_acceptance": out["pivot_acceptance"],
                "samples": len(out["r2"]),
                "seconds": round(time.time() - t0, 2),
            }
        )
        print(f"N={n:4d}  <R^2>={rows[-1]['r2_mean']:.2f} ± {rows[-1]['r2_err']:.2f}  "
              f"pivot acc={out['pivot_acceptance']:.2f}")
    df = pd.DataFrame(rows)
    res_dir = ROOT / "results"
    df.to_csv(res_dir / "free_chain.csv", index=False)

    big = df[df.bonds >= 31]
    nu, nu_err, amp = fit_flory_exponent(big.bonds, big.r2_mean, big.r2_err)
    nu_g, nu_g_err, _ = fit_flory_exponent(big.bonds, big.rg2_mean, big.rg2_err)
    nu_c, nu_c_err, _, b_c = fit_flory_with_correction(df.bonds, df.r2_mean, df.r2_err)
    summary = {
        "nu_with_correction_to_scaling": nu_c,
        "nu_with_correction_to_scaling_err": nu_c_err,
        "correction_amplitude_b": b_c,
        "nu_from_R2": nu,
        "nu_from_R2_err": nu_err,
        "nu_from_Rg2": nu_g,
        "nu_from_Rg2_err": nu_g_err,
        "nu_reference": NU_REFERENCE,
        "fit_range_bonds": [int(big.bonds.min()), int(big.bonds.max())],
        "quick_mode": quick,
    }
    (res_dir / "free_chain_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar(df.bonds, df.r2_mean, yerr=df.r2_err, fmt="o", label="MC  ⟨R²⟩")
    xs = np.geomspace(df.bonds.min(), df.bonds.max(), 50)
    ax.plot(xs, amp * xs ** (2 * nu), "-", label=f"pure power law: ν = {nu:.3f} ± {nu_err:.3f}")
    ax.plot(xs, amp * xs ** (2 * NU_REFERENCE) * (big.bonds.iloc[0] ** (2 * (nu - NU_REFERENCE))),
            "--", alpha=0.6, label=f"reference ν = {NU_REFERENCE}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("bonds N")
    ax.set_ylabel("⟨R²⟩")
    ax.set_title(f"Free SAW, cubic lattice\nwith correction to scaling: ν = {nu_c:.3f} ± {nu_c_err:.3f}", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(res_dir / "figures" / "free_chain_scaling.png", dpi=150)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="short run for CI")
    p.add_argument("--seed", type=int, default=2026)
    a = p.parse_args()
    main(a.quick, a.seed)
