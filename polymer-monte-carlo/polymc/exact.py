"""Exact enumeration of short SAWs, used to validate the Monte Carlo samplers."""
from __future__ import annotations

import numpy as np

_STEPS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def _walks(n_steps, start, half_space):
    path = [start]
    seen = {start}

    def rec(depth):
        if depth == n_steps:
            yield list(path)
            return
        x, y, z = path[-1]
        for dx, dy, dz in _STEPS:
            p = (x + dx, y + dy, z + dz)
            if p in seen or (half_space and p[2] < 1):
                continue
            path.append(p)
            seen.add(p)
            yield from rec(depth + 1)
            path.pop()
            seen.discard(p)

    yield from rec(0)


def exact_free(n_monomers: int) -> dict:
    """Exact <R^2> and <Rg^2> over all SAWs with ``n_monomers`` sites."""
    r2, rg2, count = 0.0, 0.0, 0
    for w in _walks(n_monomers - 1, (0, 0, 0), False):
        a = np.array(w, dtype=float)
        r2 += ((a[-1] - a[0]) ** 2).sum()
        rg2 += ((a - a.mean(0)) ** 2).sum(1).mean()
        count += 1
    return {"count": count, "r2": r2 / count, "rg2": rg2 / count}


def exact_grafted(n_monomers: int, eps: float) -> dict:
    """Exact Boltzmann averages for a chain grafted at (0,0,1) with wall attraction eps."""
    z_sum, ns_sum, r2_sum = 0.0, 0.0, 0.0
    for w in _walks(n_monomers - 1, (0, 0, 1), True):
        a = np.array(w, dtype=float)
        ns = int((a[:, 2] == 1).sum())
        wt = np.exp(eps * ns)
        z_sum += wt
        ns_sum += wt * ns
        r2_sum += wt * ((a[-1] - a[0]) ** 2).sum()
    return {"ns": ns_sum / z_sum, "r2": r2_sum / z_sum}
