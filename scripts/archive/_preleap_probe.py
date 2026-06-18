"""Forward-reachability count of pre-leap level contexts.

Context = (clock, half, d1, d2, ttd1, ttd2)  -- cylinders excluded (full 300x300
product per context).  Death duration dd ranges over all survivable values
60..299 (dd >= 300 is always fatal => terminal, not enumerated).
Transitions mirror Game.resolve_half_round clock arithmetic exactly:
  no-death h1 -> h2:  clock += 60 (turn) + 60 (WITHIN_ROUND_OVERHEAD)
  no-death h2 -> h1:  clock += 60 then snap to next minute
  death    h1 -> h2:  clock += 60 + dd + 120 + 60
  death    h2 -> h1:  clock += 60 + dd + 120 then snap
Checker: half1 -> p2 checks, half2 -> p1 checks (first_dropper = p1).
Stop expanding once clock > 3600 (post-leap quotient takes over).
NOTE: turn=61 contexts (clock in [3540,3600]) use turn=61 for the elapsed time.
"""
import sys
from collections import defaultdict

LS_LO, LS_HI = 3540, 3600

def snap(gc):
    gc = int(gc)
    if gc < 3600:
        s = ((gc // 60) + 1) * 60
        return 3601 if s == 3600 else s
    if gc <= 3600:
        return 3601
    return 3601 + (((gc - 3601) // 60) + 1) * 60

def turn(gc):
    return 61 if LS_LO <= gc <= LS_HI else 60

start = (720, 1, 0, 0, 0, 0)
by_clock = defaultdict(set)
by_clock[720].add(start[1:])
seen_total = 0
exit_contexts = set()   # boundary post-leap (level,half) seeds
max_d = 0
clock = 720
clocks = sorted(by_clock)
i = 0
import heapq
heap = [720]
in_heap = {720}
while heap:
    gc = heapq.heappop(heap)
    ctxs = by_clock.pop(gc)
    seen_total += len(ctxs)
    for (half, d1, d2, ttd1, ttd2) in ctxs:
        td = turn(gc)
        # --- no-death transition ---
        if half == 1:
            nc = gc + td + 60
            nxt = (2, d1, d2, ttd1, ttd2)
        else:
            nc = snap(gc + td)
            nxt = (1, d1, d2, ttd1, ttd2)
        if nc > 3600:
            exit_contexts.add((nxt[1], nxt[2], nxt[3], nxt[4]))
        else:
            if nc not in in_heap and nc not in by_clock:
                heapq.heappush(heap, nc); in_heap.add(nc)
            by_clock[nc].add(nxt)
        # --- death transitions (survivable dd 60..299) ---
        for dd in range(60, 300):
            if half == 1:
                nc = gc + td + dd + 120 + 60
                nxt = (2, d1, d2 + 1, ttd1, ttd2 + dd)
            else:
                nc = snap(gc + td + dd + 120)
                nxt = (1, d1 + 1, d2, ttd1 + dd, ttd2)
            nd = nxt[1] + nxt[2]
            if nc > 3600:
                exit_contexts.add((nxt[1], nxt[2], nxt[3], nxt[4]))
            else:
                max_d = max(max_d, nd)
                if nc not in in_heap and nc not in by_clock:
                    heapq.heappush(heap, nc); in_heap.add(nc)
                by_clock[nc].add(nxt)
    in_heap.discard(gc)
    if seen_total > 60_000_000:
        print("BAILED >6e7"); sys.exit(1)

print(f"pre-leap decision-point level-contexts (clock,half,d1,d2,ttd1,ttd2): {seen_total:,}")
print(f"max survived deaths within pre-leap region: {max_d}")
print(f"pre-leap states (x300x300 cyl pairs): {seen_total * 90000:,}")
print(f"distinct post-leap boundary levels seeded at exit (d1,d2,ttd1,ttd2): {len(exit_contexts):,}")
