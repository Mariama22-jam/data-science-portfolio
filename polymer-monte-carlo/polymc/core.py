"""Dynamic Monte Carlo for self-avoiding walks (SAWs) on the simple cubic lattice.

Two systems are supported:

* a free chain in bulk, sampled with pivot and reptation ("slithering snake") moves;
* a chain end-grafted to an impenetrable attractive surface, sampled with pivot moves
  about any site plus local end/kink moves, using the Metropolis rule.

Conventions
-----------
A chain of ``n`` monomers is an ``(n, 3)`` int64 array of lattice coordinates.
For the surface system the wall occupies the plane ``z = 0`` and monomers must
satisfy ``z >= 1``. Each monomer in the layer ``z = 1`` contributes energy
``-eps`` (in units of k_B T), so the Boltzmann weight of a conformation is
``exp(eps * n_surface)``.
"""
from __future__ import annotations

import numpy as np
from numba import njit

# --------------------------------------------------------------------------- #
# Lattice symmetries
# --------------------------------------------------------------------------- #


def cubic_symmetries() -> np.ndarray:
    """Return the 47 non-identity elements of the cubic point group O_h.

    They are the 48 signed 3x3 permutation matrices minus the identity.
    """
    import itertools

    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3), dtype=np.int64)
            for row, (col, s) in enumerate(zip(perm, signs)):
                m[row, col] = s
            if not np.array_equal(m, np.eye(3, dtype=np.int64)):
                mats.append(m)
    return np.array(mats, dtype=np.int64)


SYMMETRIES = cubic_symmetries()
# the 7 non-identity symmetries that leave the z axis fixed (rotations about z and
# reflections in vertical planes); pivots with these keep an adsorbed tail adsorbed
PLANAR_SYMMETRIES = np.array([g for g in SYMMETRIES if g[2, 2] == 1 and abs(g[2]).sum() == 1], dtype=np.int64)
NEIGHBOURS = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
    dtype=np.int64,
)

# --------------------------------------------------------------------------- #
# Helpers (numba)
# --------------------------------------------------------------------------- #

_OFFSET = 1 << 20  # coordinates are shifted by this before hashing
_BASE = 1 << 21


@njit(cache=True)
def _keys(chain):
    n = chain.shape[0]
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        out[i] = ((chain[i, 0] + _OFFSET) * _BASE + (chain[i, 1] + _OFFSET)) * _BASE + (
            chain[i, 2] + _OFFSET
        )
    return out


@njit(cache=True)
def is_self_avoiding(chain) -> bool:
    k = np.sort(_keys(chain))
    for i in range(1, k.shape[0]):
        if k[i] == k[i - 1]:
            return False
    return True


@njit(cache=True)
def is_connected(chain) -> bool:
    for i in range(1, chain.shape[0]):
        d = 0
        for a in range(3):
            d += abs(chain[i, a] - chain[i - 1, a])
        if d != 1:
            return False
    return True


@njit(cache=True)
def n_surface(chain) -> int:
    c = 0
    for i in range(chain.shape[0]):
        if chain[i, 2] == 1:
            c += 1
    return c


@njit(cache=True)
def r2_end_to_end(chain) -> float:
    n = chain.shape[0] - 1
    s = 0.0
    for a in range(3):
        d = chain[n, a] - chain[0, a]
        s += d * d
    return s


@njit(cache=True)
def gyration(chain):
    """Return (Rg^2, Rg_parallel^2, Rg_perp^2); perp is the z component."""
    n = chain.shape[0]
    mean = np.zeros(3)
    for i in range(n):
        for a in range(3):
            mean[a] += chain[i, a]
    mean /= n
    comp = np.zeros(3)
    for i in range(n):
        for a in range(3):
            d = chain[i, a] - mean[a]
            comp[a] += d * d
    comp /= n
    return comp[0] + comp[1] + comp[2], comp[0] + comp[1], comp[2]


# --------------------------------------------------------------------------- #
# Moves
# --------------------------------------------------------------------------- #


@njit(cache=True)
def pivot_move(chain, syms, surface, eps, planar):
    """Pivot the part of the chain after a random site k (sites k+1..n-1).

    For the surface system site 0 is grafted, so only the tail end is pivoted, and
    half of the attempts draw from the in-plane subgroup ``planar``. Choosing between
    two inverse-closed symmetry sets with fixed probability keeps the proposal
    symmetric, so detailed balance holds.
    For the free chain the head or the tail is pivoted with equal probability,
    which is equivalent by symmetry. Returns True when the move is accepted.
    """
    n = chain.shape[0]
    k = np.random.randint(0, n - 1)
    if surface and np.random.random() < 0.5:
        g = planar[np.random.randint(0, planar.shape[0])]
    else:
        g = syms[np.random.randint(0, syms.shape[0])]
    new = chain.copy()
    flip_head = (not surface) and np.random.random() < 0.5
    if flip_head:
        # rotate sites 0..k-1 around site k (k >= 1 so something moves)
        k = k + 1
        lo, hi = 0, k
    else:
        lo, hi = k + 1, n
    for i in range(lo, hi):
        for a in range(3):
            s = 0
            for b in range(3):
                s += g[a, b] * (chain[i, b] - chain[k, b])
            new[i, a] = chain[k, a] + s
    if surface:
        for i in range(lo, hi):
            if new[i, 2] < 1:
                return False
    if not is_self_avoiding(new):
        return False
    if surface and eps != 0.0:
        d_ns = n_surface(new) - n_surface(chain)
        if d_ns < 0 and np.random.random() >= np.exp(eps * d_ns):
            return False
    chain[:, :] = new
    return True


@njit(cache=True)
def reptation_move(chain, nbrs):
    """Slithering-snake move for a free chain: delete one end, grow the other."""
    n = chain.shape[0]
    d = nbrs[np.random.randint(0, 6)]
    new = np.empty_like(chain)
    if np.random.random() < 0.5:  # head grows at the tail end
        for i in range(n - 1):
            new[i] = chain[i + 1]
        for a in range(3):
            new[n - 1, a] = chain[n - 1, a] + d[a]
    else:
        for i in range(1, n):
            new[i] = chain[i - 1]
        for a in range(3):
            new[0, a] = chain[0, a] + d[a]
    if not is_self_avoiding(new):
        return False
    chain[:, :] = new
    return True


@njit(cache=True)
def crankshaft_move(chain, nbrs, eps):
    """Rotate a U-shaped pair (i, i+1) about the axis through i-1 and i+2.

    Applies when chain[i+2] - chain[i-1] is a unit vector u and both i and i+1 are
    displaced from it by the same vector v perpendicular to u. The pair moves to
    one of the other three perpendicular directions, chosen uniformly, so the
    proposal is symmetric. This creates and removes small loops at the surface.
    """
    n = chain.shape[0]
    if n < 4:
        return False
    i = np.random.randint(1, n - 2)
    u = np.empty(3, dtype=np.int64)
    v = np.empty(3, dtype=np.int64)
    lu = 0
    for a in range(3):
        u[a] = chain[i + 2, a] - chain[i - 1, a]
        v[a] = chain[i, a] - chain[i - 1, a]
        lu += abs(u[a])
        if chain[i + 1, a] - chain[i + 2, a] != v[a]:
            return False
    if lu != 1:
        return False
    # choose one of the 3 other directions perpendicular to u
    cands = np.empty((4, 3), dtype=np.int64)
    m = 0
    for d in range(6):
        dot = 0
        same = True
        for a in range(3):
            dot += nbrs[d, a] * u[a]
            if nbrs[d, a] != v[a]:
                same = False
        if dot == 0 and not same:
            cands[m] = nbrs[d]
            m += 1
    w = cands[np.random.randint(0, m)]
    p1 = np.empty(3, dtype=np.int64)
    p2 = np.empty(3, dtype=np.int64)
    for a in range(3):
        p1[a] = chain[i - 1, a] + w[a]
        p2[a] = chain[i + 2, a] + w[a]
    if p1[2] < 1 or p2[2] < 1:
        return False
    for j in range(n):
        if j == i or j == i + 1:
            continue
        if (chain[j, 0] == p1[0] and chain[j, 1] == p1[1] and chain[j, 2] == p1[2]) or (
            chain[j, 0] == p2[0] and chain[j, 1] == p2[1] and chain[j, 2] == p2[2]
        ):
            return False
    old_s = (1 if chain[i, 2] == 1 else 0) + (1 if chain[i + 1, 2] == 1 else 0)
    new_s = (1 if p1[2] == 1 else 0) + (1 if p2[2] == 1 else 0)
    d_ns = new_s - old_s
    if d_ns < 0 and np.random.random() >= np.exp(eps * d_ns):
        return False
    for a in range(3):
        chain[i, a] = p1[a]
        chain[i + 1, a] = p2[a]
    return True


@njit(cache=True)
def local_move(chain, nbrs, eps):
    """Local move for the grafted chain (site 0 fixed): crankshaft with probability
    1/2, otherwise an end rotation or kink jump at a random site."""
    if np.random.random() < 0.5:
        return crankshaft_move(chain, nbrs, eps)
    n = chain.shape[0]
    i = np.random.randint(1, n)
    trial = np.empty(3, dtype=np.int64)
    if i == n - 1:
        d = nbrs[np.random.randint(0, 6)]
        for a in range(3):
            trial[a] = chain[n - 2, a] + d[a]
    else:
        # kink jump: only possible when bonds (i-1,i) and (i,i+1) are perpendicular
        for a in range(3):
            trial[a] = chain[i - 1, a] + chain[i + 1, a] - chain[i, a]
        straight = True
        for a in range(3):
            if chain[i + 1, a] - chain[i, a] != chain[i, a] - chain[i - 1, a]:
                straight = False
        if straight:
            return False
    if trial[2] < 1:
        return False
    for j in range(n):
        if j != i and chain[j, 0] == trial[0] and chain[j, 1] == trial[1] and chain[j, 2] == trial[2]:
            return False
    d_ns = (1 if trial[2] == 1 else 0) - (1 if chain[i, 2] == 1 else 0)
    if d_ns < 0 and np.random.random() >= np.exp(eps * d_ns):
        return False
    for a in range(3):
        chain[i, a] = trial[a]
    return True


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #


@njit(cache=True)
def _seed(s):
    np.random.seed(s)


@njit(cache=True)
def _run_free(chain, syms, nbrs, n_sweeps, n_equil, sample_every, p_pivot):
    n = chain.shape[0]
    n_samples = max(0, (n_sweeps - n_equil) // sample_every)
    out = np.empty((n_samples, 2))
    acc_p = 0
    tried_p = 0
    j = 0
    for sweep in range(n_sweeps):
        for _ in range(n):
            if np.random.random() < p_pivot:
                tried_p += 1
                if pivot_move(chain, syms, False, 0.0, syms):
                    acc_p += 1
            else:
                reptation_move(chain, nbrs)
        if sweep >= n_equil and (sweep - n_equil) % sample_every == 0 and j < n_samples:
            out[j, 0] = r2_end_to_end(chain)
            out[j, 1] = gyration(chain)[0]
            j += 1
    return out[:j], acc_p / max(tried_p, 1)


@njit(cache=True)
def _run_surface(chain, syms, planar, nbrs, eps, n_sweeps, n_equil, sample_every, p_pivot):
    n = chain.shape[0]
    n_samples = max(0, (n_sweeps - n_equil) // sample_every)
    out = np.empty((n_samples, 4))
    j = 0
    for sweep in range(n_sweeps):
        for _ in range(n):
            if np.random.random() < p_pivot:
                pivot_move(chain, syms, True, eps, planar)
            else:
                local_move(chain, nbrs, eps)
        if sweep >= n_equil and (sweep - n_equil) % sample_every == 0 and j < n_samples:
            g = gyration(chain)
            out[j, 0] = n_surface(chain)
            out[j, 1] = r2_end_to_end(chain)
            out[j, 2] = g[1]
            out[j, 3] = g[2]
            j += 1
    return out[:j]


@njit(cache=True)
def _run_tempering(chains, eps, syms, planar, nbrs, n_sweeps, n_equil, sample_every, p_pivot):
    """Replica exchange over attraction strengths.

    ``chains[k]`` is the conformation currently held at ``eps[k]``. After every
    sweep, neighbouring pairs (alternating even/odd) attempt to swap
    conformations with probability min(1, exp((eps_i - eps_j)(ns_j - ns_i))),
    which preserves each replica's Boltzmann distribution exp(eps * ns).
    Strongly adsorbed replicas thereby receive conformations that were
    rearranged at weaker attraction, curing the trapping of single-eps runs.
    """
    k_rep, n = chains.shape[0], chains.shape[1]
    n_samples = max(0, (n_sweeps - n_equil) // sample_every)
    out = np.empty((k_rep, n_samples, 4))
    tried = np.zeros(k_rep - 1)
    acc = np.zeros(k_rep - 1)
    j = 0
    for sweep in range(n_sweeps):
        for r in range(k_rep):
            c = chains[r]
            for _ in range(n):
                if np.random.random() < p_pivot:
                    pivot_move(c, syms, True, eps[r], planar)
                else:
                    local_move(c, nbrs, eps[r])
        for r in range(sweep % 2, k_rep - 1, 2):
            ns_a = n_surface(chains[r])
            ns_b = n_surface(chains[r + 1])
            tried[r] += 1
            if np.random.random() < np.exp(min(0.0, (eps[r] - eps[r + 1]) * (ns_b - ns_a))):
                tmp = chains[r].copy()
                chains[r][:, :] = chains[r + 1]
                chains[r + 1][:, :] = tmp
                acc[r] += 1
        if sweep >= n_equil and (sweep - n_equil) % sample_every == 0 and j < n_samples:
            for r in range(k_rep):
                g = gyration(chains[r])
                out[r, j, 0] = n_surface(chains[r])
                out[r, j, 1] = r2_end_to_end(chains[r])
                out[r, j, 2] = g[1]
                out[r, j, 3] = g[2]
            j += 1
    return out[:, :j], acc / np.maximum(tried, 1)


def simulate_grafted_tempering(
    n: int,
    eps_grid,
    n_sweeps: int = 2000,
    n_equil: int = 200,
    sample_every: int = 1,
    p_pivot: float = 0.3,
    seed: int = 0,
    start: str = "upright",
):
    """Grafted chain at every eps in ``eps_grid`` simultaneously, with replica exchange.

    ``start`` is "upright" (rod perpendicular to the wall, fully desorbed) or "flat"
    (rod lying in the layer z = 1, fully adsorbed). Running both and comparing is a
    convergence check: equilibrium averages must not depend on the start.

    Returns a list of dicts (one per eps, same keys as ``simulate_grafted_chain``)
    and the swap acceptance rate between neighbouring eps values.
    """
    eps = np.sort(np.asarray(eps_grid, dtype=float))
    _seed(seed)
    k = len(eps)
    chains = np.zeros((k, n, 3), dtype=np.int64)
    if start == "upright":
        chains[:, :, 2] = np.arange(1, n + 1)
    elif start == "flat":
        chains[:, :, 0] = np.arange(n)
        chains[:, :, 2] = 1
    else:
        raise ValueError("start must be 'upright' or 'flat'")
    data, swap = _run_tempering(chains, eps, SYMMETRIES, PLANAR_SYMMETRIES, NEIGHBOURS, n_sweeps, n_equil, sample_every, p_pivot)
    for c in chains:
        assert is_self_avoiding(c) and is_connected(c) and c[:, 2].min() >= 1 and tuple(c[0]) == (0, 0, 1)
    res = [
        {"eps": float(e), "ns": data[i, :, 0], "r2": data[i, :, 1], "rg2_par": data[i, :, 2], "rg2_perp": data[i, :, 3]}
        for i, e in enumerate(eps)
    ]
    return res, swap


def straight_chain(n: int, surface: bool = False) -> np.ndarray:
    """Rod along x (free) or along x in the surface layer z=1 (grafted)."""
    c = np.zeros((n, 3), dtype=np.int64)
    c[:, 0] = np.arange(n)
    if surface:
        c[:, 2] = 1
    return c


def simulate_free_chain(
    n: int,
    n_sweeps: int = 2000,
    n_equil: int = 200,
    sample_every: int = 1,
    p_pivot: float = 0.5,
    seed: int = 0,
):
    """Sample a free SAW of ``n`` monomers.

    One sweep is ``n`` attempted moves. Returns a dict with arrays ``r2`` (squared
    end-to-end distance) and ``rg2`` (squared radius of gyration), plus the pivot
    acceptance rate.
    """
    _seed(seed)
    chain = straight_chain(n)
    data, acc = _run_free(chain, SYMMETRIES, NEIGHBOURS, n_sweeps, n_equil, sample_every, p_pivot)
    assert is_self_avoiding(chain) and is_connected(chain)
    return {"r2": data[:, 0], "rg2": data[:, 1], "pivot_acceptance": acc, "chain": chain}


def simulate_grafted_chain(
    n: int,
    eps: float,
    n_sweeps: int = 2000,
    n_equil: int = 200,
    sample_every: int = 1,
    p_pivot: float = 0.3,
    seed: int = 0,
):
    """Sample a chain grafted at (0,0,1) above an attractive wall with strength ``eps``.

    Returns arrays ``ns`` (monomers in the surface layer), ``r2``, ``rg2_par`` and
    ``rg2_perp``.
    """
    _seed(seed)
    chain = straight_chain(n, surface=True)
    # start from a non-adsorbed rod standing up, so equilibration is tested both ways
    chain[:, 0] = 0
    chain[:, 2] = np.arange(1, n + 1)
    data = _run_surface(chain, SYMMETRIES, PLANAR_SYMMETRIES, NEIGHBOURS, float(eps), n_sweeps, n_equil, sample_every, p_pivot)
    assert is_self_avoiding(chain) and is_connected(chain) and chain[:, 2].min() >= 1
    assert tuple(chain[0]) == (0, 0, 1)
    return {
        "ns": data[:, 0],
        "r2": data[:, 1],
        "rg2_par": data[:, 2],
        "rg2_perp": data[:, 3],
        "chain": chain,
    }
