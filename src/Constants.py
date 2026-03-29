"""
Drop The Handkerchief — Game Constants

All timing values are in seconds unless otherwise noted.
The game clock is absolute: second 0 = 8:00:00 AM.
"""

# ──────────────────────────────────────────────
# CLOCK
# ──────────────────────────────────────────────
GAME_START_HOUR = 8                 # 8:00 AM
SECONDS_PER_MINUTE = 60
MINUTES_PER_HOUR = 60
OPENING_START_CLOCK = 12 * 60       # Canonical R1 start: 8:12:00 AM

# Leap second: inserted at exactly 8:59:60 AM.
# In game clock seconds: 8:59:00 AM = 59 * 60 = 3540s after game start.
# A half-round whose start falls in [3540, 3600) spans the leap second.
LS_WINDOW_START = 59 * 60         # 3540 — start of the 8:59 minute
LS_WINDOW_END = 60 * 60               #8:59:60 AM


# ──────────────────────────────────────────────
# TURN
# ──────────────────────────────────────────────
TURN_DURATION_NORMAL = 60           # seconds per half-round (normal)
TURN_DURATION_LEAP = 61             # seconds per half-round (during LS window)
FAILED_CHECK_PENALTY = 60           # 1 minute NDD added on failed check

# ──────────────────────────────────────────────
# CYLINDER / NDD
# ──────────────────────────────────────────────
CYLINDER_MAX = 300                  # 5 minutes — at or above this, instant injection
DEATH_PROCEDURE_OVERHEAD = 120      # injection + waiting + CPR + recovery (~2 min)
# ──────────────────────────────────────────────
# CLOCK ADVANCEMENT
#
# Within a round the game clock advances continuously:
#   turn_duration (per half) + death_duration + DEATH_PROCEDURE_OVERHEAD (if death)
#   + WITHIN_ROUND_OVERHEAD
#
# After a round completes, the clock SNAPS FORWARD to the next whole-minute boundary.
# This snap absorbs imprecision and matches the manga, where every round starts on
# an exact minute (R1: 8:12, R2: 8:19, R3: 8:26, R4: 8:30, etc.).
#
# No-death round: 60 + 60 + 60 (overhead) = 180s. Snap → 4-minute gap between rounds.
# Death round (60s death): 60 + 60 + 120 + 60 + 60 = 360s. Snap → 7-minute gap.
# Manga: "1 MINUTE OF NEAR-DEATH WOULD CONSUME 5 MINUTES OF GAME TIME"
# ──────────────────────────────────────────────
WITHIN_ROUND_OVERHEAD = 60          # total procedural time within a round (settling,
                                    # injection procedure, role swap). Applied between halves.

# ──────────────────────────────────────────────
# SURVIVAL PROBABILITY
#
# P(survival) = base_curve(t) * cardiac(ttd_prior) * referee(n) * physicality
#
# base_curve(t)       = max(0, 1 - (t / CYLINDER_MAX) ^ BASE_CURVE_K)
# cardiac(ttd_prior)  = CARDIAC_DECAY ^ (ttd_prior / 60)
# referee(n)          = max(REFEREE_FLOOR, REFEREE_DECAY ^ n)
# physicality         = per-player constant
# ──────────────────────────────────────────────

# Base curve: oxygen deprivation danger for THIS death
BASE_CURVE_K = 3

# Cardiac degradation: accumulated heart damage from prior deaths
# Each minute of prior cumulative death weakens the heart by this factor
CARDIAC_DECAY = 0.85

# Referee fatigue: CPR quality degrades with each revival performed
REFEREE_DECAY = 0.88
REFEREE_FLOOR = 0.4                 # minimum effectiveness (even exhausted)

# Player physicality presets
PHYSICALITY_HAL = 1.0
PHYSICALITY_BAKU = 0.88
