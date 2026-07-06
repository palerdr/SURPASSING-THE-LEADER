"""Feasibility probe: LP timing, invariants, state-count arithmetic."""
import sys, time, random
import numpy as np

sys.path.insert(0, r"C:\Users\owner\Desktop\stl\pstl")
from src.Game import Game
from src.Player import Player
from src.Referee import Referee
from src.Constants import PHYSICALITY_HAL, PHYSICALITY_BAKU
from environment.cfr.exact import solve_minimax

# ---------- 1. LP timing ----------
rng = np.random.default_rng(0)

def time_lp(mat, reps):
    t0 = time.perf_counter()
    for _ in range(reps):
        solve_minimax(mat)
    return (time.perf_counter() - t0) / reps

# random 60x60
m_rand = rng.uniform(-1, 1, (60, 60))
t_rand = time_lp(m_rand, 30)

# structured (real game shape): payoff(d,c) = F for c<d, W[max(1,c-d)] for c>=d
W = np.sort(rng.uniform(-1, 1, 61))[::-1]  # decreasing in ST
F = -0.3
m_struct = np.empty((60, 60))
for d in range(1, 61):
    for c in range(1, 61):
        m_struct[d-1, c-1] = F if c < d else W[max(1, c - d)]
t_struct = time_lp(m_struct, 30)

# 61x60 (leap, Baku dropper)
m_leap = np.empty((61, 60))
m_leap[:60, :] = m_struct
m_leap[60, :] = F  # drop=61 always fails vs check<=60
t_leap = time_lp(m_leap, 30)

print(f"LP 60x60 random:      {t_rand*1000:.2f} ms")
print(f"LP 60x60 structured:  {t_struct*1000:.2f} ms")
print(f"LP 61x60 leap:        {t_leap*1000:.2f} ms")

# ---------- 2. Invariant verification via random play ----------
def fresh_game():
    p1 = Player("Hal", physicality=PHYSICALITY_HAL)
    p2 = Player("Baku", physicality=PHYSICALITY_BAKU)
    g = Game(player1=p1, player2=p2, referee=Referee())
    return g

viol = []
preleap_clocks = set()
leapwin_clocks = set()
max_deaths_seen = 0
max_preleap_deaths = 0
overflow_survivals = 0
st_max = 0
r = random.Random(42)
N_GAMES = 4000
for gi in range(N_GAMES):
    g = fresh_game()
    g.seed(gi)
    while not g.game_over and g.game_clock < 30000:
        td = g.get_turn_duration()
        # decision-point invariants
        if g.game_clock != int(g.game_clock):
            viol.append(("noninteger clock", g.game_clock))
        c1, c2 = g.player1.cylinder, g.player2.cylinder
        if c1 != int(c1) or c2 != int(c2):
            viol.append(("noninteger cyl", c1, c2))
        if not (0 <= c1 <= 299 and 0 <= c2 <= 299):
            viol.append(("cyl range", c1, c2))
        t1, t2 = g.player1.ttd, g.player2.ttd
        if t1 != int(t1) or t2 != int(t2):
            viol.append(("noninteger ttd", t1, t2))
        if g.referee.cprs_performed != g.player1.deaths + g.player2.deaths:
            viol.append(("cprs mismatch", g.referee.cprs_performed,
                         g.player1.deaths, g.player2.deaths))
        if g.game_clock <= 3600:
            preleap_clocks.add((int(g.game_clock), g.current_half))
            if td == 61:
                leapwin_clocks.add((int(g.game_clock), g.current_half))
        # random legal joint action (Hal never 61 — but engine perspective: cap check at 60)
        dmax = td  # allow Baku-style drop 61
        drop = r.randint(1, dmax)
        check = r.randint(1, 60)
        rec = g.play_half_round(drop, check)
        if rec.st_gained:
            st_max = max(st_max, int(rec.st_gained))
        if rec.result.value.startswith("overflow") and rec.survived:
            overflow_survivals += 1
        d = g.player1.deaths + g.player2.deaths
        max_deaths_seen = max(max_deaths_seen, d)
        if g.game_clock <= 3600:
            max_preleap_deaths = max(max_preleap_deaths, d)

print(f"\ngames={N_GAMES} violations={viol[:5]} (count={len(viol)})")
print(f"overflow survivals (should be 0): {overflow_survivals}")
print(f"max ST seen: {st_max}")
print(f"max total survived deaths seen: {max_deaths_seen}, pre-leap: {max_preleap_deaths}")
print(f"distinct pre-leap (clock,half) decision points: {len(preleap_clocks)}")
print(f"leap-window decision points (turn=61): {sorted(leapwin_clocks)}")
h1 = sorted(c for c, h in preleap_clocks if h == 1)
print(f"half-1 pre-leap clocks: n={len(h1)} min={h1[0]} max={h1[-1]} all_minute={all(c % 60 == 0 for c in h1)}")

# turn duration at exactly 3600
g = fresh_game(); g.game_clock = 3600
print(f"turn duration at clock=3600: {g.get_turn_duration()}")

# ---------- 3. State-count arithmetic ----------
# T(d) = number of reachable ttd values after d survived deaths.
# Survivable death duration = cyl_before + 60, cyl_before in 0..239 -> dd in 60..299 (contiguous)
# sum of d values from contiguous [60,299] = contiguous [60d, 299d] -> T(d) = 239d+1
def T(d):
    return 1 if d == 0 else 239 * d + 1

PER_LEVEL = 2 * 300 * 300  # (half, cyl1, cyl2)

print("\nN  levels L(N)      states (post-leap quotient)   reach-bound P(N)")
# reach-prob bound: n-th total survived death, per-player prior deaths >= floor((n-1)/2),
# cprs prior = n-1; best-case dd=60 -> base=1-(60/300)^3=0.992; phys<=1
def reach_bound(N):
    p = 1.0
    for n in range(1, N + 1):
        cardiac = 0.85 ** ((n - 1) // 2)  # ttd_prior >= 60*floor((n-1)/2) for the dying player
        refmod = max(0.4, 0.88 ** (n - 1))
        p *= 0.992 * cardiac * refmod
    return p

for N in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
    L = sum(T(d1) * T(d2) for d1 in range(N + 1) for d2 in range(N + 1 - d1))
    states = L * PER_LEVEL
    print(f"{N:2d} {L:12,d} {states:20,d}   {reach_bound(N):.3e}")

# per-level LP cost
print(f"\nper-level cost @ {t_struct*1000:.1f} ms/LP: {PER_LEVEL * t_struct:.0f} s single-core "
      f"= {PER_LEVEL * t_struct / 60:.1f} min; /30 cores = {PER_LEVEL * t_struct / 30:.0f} s")
