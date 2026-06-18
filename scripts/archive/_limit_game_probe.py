"""Solve the 'all-deaths-fatal' limit game exactly by backward induction.

This is the asymptote of deep levels (survival prob -> 0): every death
(failed check or cylinder overflow) is permanently fatal. The quotient state
is just (half, cylH, cylB) with first_dropper = Hal:
  half 1: Hal drops, Baku checks (cylB at risk)
  half 2: Baku drops, Hal checks (cylH at risk)
Within-level DAG: success adds ST = max(1, c-d) >= 1 to checker cylinder,
so process cyl-sum DESCENDING (children have larger sums). One LP per state.

Outputs: per-state LP wall time (empirical per-level cost), value at
(h=1, 0, 0), value table summary.
"""
import sys, time
import numpy as np

sys.path.insert(0, r"C:\Users\owner\Desktop\stl\pstl")
from environment.cfr.exact import solve_minimax

NCYL = 300
# V[h-1, ch, cb], Hal perspective
V = np.zeros((2, NCYL, NCYL), dtype=np.float64)

t0 = time.perf_counter()
n_lp = 0
n_pure = 0

# Precompute ST for (d,c) grid, d,c in 1..60
d_idx = np.arange(1, 61)[:, None]
c_idx = np.arange(1, 61)[None, :]
success = c_idx >= d_idx
ST = np.where(success, np.maximum(1, c_idx - d_idx), 0)  # 0 marks fail

for s_sum in range(2 * (NCYL - 1), -1, -1):
    lo = max(0, s_sum - (NCYL - 1))
    hi = min(NCYL - 1, s_sum)
    for ch in range(lo, hi + 1):
        cb = s_sum - ch
        for h in (1, 2):
            cx = cb if h == 1 else ch              # checker cylinder
            die_val = 1.0 if h == 1 else -1.0      # checker dies -> Hal view
            nh = 2 if h == 1 else 1
            # next-state values per ST s = 1..59: checker cyl cx+s
            nxt = np.empty(60)
            for s in range(1, 60):
                ncx = cx + s
                if ncx >= NCYL:
                    nxt[s] = die_val               # overflow fatal
                else:
                    if h == 1:
                        nxt[s] = V[nh - 1, ch, ncx]
                    else:
                        nxt[s] = V[nh - 1, ncx, cb]
            payoff = np.where(success, nxt[np.maximum(ST, 1)], die_val)
            # rows = dropper. Hal maximizes. h=1: Hal drops -> rows. h=2: Hal checks -> cols.
            if h == 1:
                _, val = solve_minimax(payoff)
            else:
                _, val = solve_minimax(payoff.T)
            n_lp += 1
            V[h - 1, ch, cb] = val

elapsed = time.perf_counter() - t0
print(f"states solved: {n_lp:,} in {elapsed:.0f} s  -> {elapsed / n_lp * 1000:.2f} ms/state")
print(f"V_limit(h=1, 0, 0) = {V[0, 0, 0]:+.6f}   (Hal drops first, both cylinders empty)")
print(f"V_limit(h=2, 0, 0) = {V[1, 0, 0]:+.6f}   (Baku drops first)")
print(f"value range: [{V.min():+.4f}, {V.max():+.4f}]")
print(f"mean |V|: {np.abs(V).mean():.4f}; states with |V| > 0.99: {(np.abs(V) > 0.99).mean() * 100:.1f}%")
np.save(r"C:\Users\owner\Desktop\stl\pstl\_limit_game_values.npy", V)
print("saved -> _limit_game_values.npy")
