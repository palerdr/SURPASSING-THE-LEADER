# Frozen unified revival model

This file is the sole repository authority for the revival-probability surface.
Every formulation in [`FORMULATION_LADDER.md`](FORMULATION_LADDER.md) — bucketed
abstractions, pure DTH, and the leap-aware STL public game — must use exactly
this function. A single frozen surface is what makes one CLI able to play every
granularity against comparable numbers.

Rule boundaries (capacity, dose composition, eligibility) are documentary and
are owned by [`CANONICAL_EXTENSIVE_FORM.md`](CANONICAL_EXTENSIVE_FORM.md) and
[`game-sources/EVIDENCE.md`](game-sources/EVIDENCE.md). The *numbers* below are
not. They are a declared solver model; see the modeling boundary at the end.

<!-- canon:C-REVIVAL-UNIFIED -->
## The model

Let `s` be the squandered time already in the current Checker's vial
immediately before a failed check, and `t` that player's accrued time-to-death
from prior survived deaths, both in seconds. The failed check adds a fixed
60-second injection, so the dose is `q = s + 60`.

```text
eligible(s, t)  <=>  q < 300  and  t + q <= 300
                <=>  s <= 239  and  s + t <= 240

P_rev(s, t) = 0.95 * (1 - s / 240) * 0.75^(t/60)      if eligible
            = 0                                       otherwise
```

Two factors and one constant:

- **`0.95` — baseline.** The probability of surviving a bare 60-second
  injection with a clean body. Absorbs STL's per-player `physicality`
  multiplier at a fixed nominal value, since reduced formulations carry no
  player identity.
- **`1 - s / 240` — dose response, linear.** Equivalently `(300 - q)/240`: it is
  linear in the *dose* and reaches zero exactly at the documentary lethal dose
  `q = 300`.
- **`0.75^(t/60)` — prior damage, geometric.** Each accrued death-minute costs a
  quarter of the remaining odds. Half-life 144.3 seconds.

`240 = 300 - 60` is the survivable ST span. It is the same constant in the
denominator of the dose factor and in the eligibility bound `s + t <= 240`; both
express the headroom before the 60-second penalty alone becomes lethal.

### Bucket form

For bucket width `B`, with `C = 300/B` load units and `F = 60/B` dose units, and
`l = s/B`, `tau = t/B`:

```text
P_rev(l, tau) = 0.95 * (1 - l / (C - F)) * 0.75^(tau/F)
```

Because `s/240 = l/(C-F)` and `t/60 = tau/F` identically, the surface is
**exactly bucket-invariant**: the same physical `(s, t)` yields bit-identical
probabilities at `B = 10`, `B = 5`, and `B = 1`. Verified to `0.00e+00` spread.

## Why this shape

### Linear in ST, not cubic

The prior repository model used `1 - (q/300)^3`. Linear is preferred because:

1. **It vanishes exactly at the lethal dose**, and only there. `1 - s/240 = 0`
   precisely when `s = 240`, i.e. `q = 300`, which
   [E-CYLINDER-CAP](game-sources/EVIDENCE.md#e-cylinder-cap) closes
   independently. The zero is therefore never assigned to a reachable survivable
   injection: the largest of those is `s = 239`, `q = 299`, where
   `P = 0.95/240 = 0.003958`. The probability meets the boundary continuously
   instead of arriving with zero slope.
2. **The cubic is flat where the game is actually played.** It assigns above
   0.97 to every dose below 90 seconds, so vial contents barely matter until the
   very end. That contradicts the premise of an accumulation battle.
3. It is the simplest monotone shape with the right endpoints, and monotonicity
   is what lets a solver derive sound bounds.

### Geometric in TTD

[E-PROXIMITY-RISK](game-sources/EVIDENCE.md#e-proximity-risk) establishes that
prior damage lowers revival odds enough to change optimal play, and
[E-REFEREE-CONDITION](game-sources/EVIDENCE.md#e-referee-condition) establishes
a second, independent degradation in the referee. Neither is a threshold effect;
both compound with repetition. A constant per-death-minute multiplier is the
minimal encoding of that, and it keeps the surface a function of exactly two
variables.

`0.75` is a rounding of `0.85 * 0.88 = 0.748`, the product of STL's
`CARDIAC_DECAY` and `REFEREE_DECAY` (`src/stl/engine/game.py:79,81`). The
rounding costs at most 1.07% relative error, at `t = 240`; since neither
constant is documentary, no accuracy that ever existed is lost.

### Why there is no referee floor

STL carries `max(0.40, 0.88^cprs)`. Reduced to two variables that floor would be
`max(0.40, 0.88^(t/60))`, which **can never bind**: it requires `t/60 > 7.17`
death-minutes, while eligibility caps `t` at 240 s = 4.0 death-minutes — a
3.17-minute margin. It is dead code inside the eligible region, so it is omitted
rather than carried.

### Two variables, not four

STL's engine takes four inputs: dose, TTD, referee CPR count, and per-player
physicality. Reduced formulations have no referee object and no player identity.
Folding physicality into the baseline and the CPR count into `t` — each death
costs at least 60 seconds of TTD, so `t/60` is a lower bound on deaths — gives a
two-variable surface faithful to STL's structure and portable to every
abstraction. This is a modeling bridge, not a claim that accrued TTD is a
literal observed CPR count.

## Table

Revival probability by vial contents and prior accrued TTD. `--` marks states
where `s + t > 240`, i.e. the injection is fatal with certainty.

| ST `s` \ TTD `t` | 0 | 60 | 120 | 180 | 240 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.9500 | 0.7125 | 0.5344 | 0.4008 | 0.3006 |
| 20 | 0.8708 | 0.6531 | 0.4898 | 0.3674 | -- |
| 40 | 0.7917 | 0.5938 | 0.4453 | 0.3340 | -- |
| 60 | 0.7125 | 0.5344 | 0.4008 | 0.3006 | -- |
| 80 | 0.6333 | 0.4750 | 0.3563 | -- | -- |
| 100 | 0.5542 | 0.4156 | 0.3117 | -- | -- |
| 120 | 0.4750 | 0.3563 | 0.2672 | -- | -- |
| 140 | 0.3958 | 0.2969 | -- | -- | -- |
| 160 | 0.3167 | 0.2375 | -- | -- | -- |
| 180 | 0.2375 | 0.1781 | -- | -- | -- |
| 200 | 0.1583 | -- | -- | -- | -- |
| 220 | 0.0792 | -- | -- | -- | -- |
| 239 | 0.0040 | -- | -- | -- | -- |

Endpoints: `P(0,0) = 0.9500`, `P(239,0) = 0.003958`, `P(0,240) = 0.300586`.

### Known asymmetry

Along the frontier `s + t = 240`, total load is always exactly 300, yet the
probability spans 76x:

| `s` | `t` | `t + q` | `P_rev` |
|---:|---:|---:|---:|
| 239 | 1 | 300 | 0.003939 |
| 180 | 60 | 300 | 0.178125 |
| 120 | 120 | 300 | 0.267187 |
| 60 | 180 | 300 | 0.300586 |
| 0 | 240 | 300 | 0.300586 |

The hard cap is on cumulative load, but the surface is dominated by the acute
dose. This is deliberate — a single massive bolus is more lethal than the same
total taken in installments — but it is a modeling choice that no evidence
forces. The lever is the TTD decay constant, not the linear dose term.

## Validation

Checked by replaying the canonical match ledger from
[`IN_DEPTH_SUMMARY.md`](game-sources/IN_DEPTH_SUMMARY.md).

| Recorded revival | `s` | `t` | dose | `t + q` | `P_rev` |
|---|---:|---:|---:|---:|---:|
| R1T1 Baku | 0 | 0 | 60 | 60 | 0.9500 |
| R2T2 Leader | 24 | 0 | 84 | 84 | 0.8550 |
| R6T1 Baku | 33 | 60 | 93 | 153 | 0.6145 |
| R8T2 Leader | 94 | 84 | 154 | 238 | 0.3863 |
| R9T2 Leader | 0 | 238 | 60 | 298 | 0.3035 |

All five recorded revivals succeeded. The model assigns them monotonically
decreasing probability in exactly the order the narrative escalates; the joint
likelihood of the observed `5/5` record is `0.0585`.

The final revival is *not* made dramatic by a low roll. It is made dramatic by
the eligibility boundary: at `t = 238`, revival requires `s <= 2`. Leader had
`s = 0`. Three more seconds of accumulation anywhere in rounds 6-8 and the
injection is fatal with certainty. The model reproduces the "2 seconds of
deviation" as slack in `s + t <= 240`, not as luck.

Other verified properties:

- **Eligibility is unchanged** from the previous repository rule. The zero-set
  matches `dth.solver.survive_injection` on all `300 x 301` pairs with zero
  mismatches, so the reachable game graph, its state counts, and every exact
  quotient are unaffected. Only values change.
- **Monotone non-increasing in both `s` and `t`** across the eligible region,
  which admits sound interval bounds.
- Range `[0, 0.95]`.

### Baseline sensitivity

Four exact `bucket6` solves (576,270 states each) under this model, sweeping the
baseline only:

| baseline | root value | root dropper policy, buckets 1-6 |
|---:|---:|---|
| 0.85 | +0.097836450 | 0.387, 0.080, 0.097, 0.118, 0.143, 0.174 |
| 0.90 | +0.096389226 | 0.339, 0.082, 0.102, 0.126, 0.156, 0.194 |
| 0.95 | +0.096931018 | 0.292, 0.083, 0.105, 0.134, 0.170, 0.216 |
| 0.99 | +0.099212059 | 0.257, 0.082, 0.107, 0.139, 0.180, 0.235 |

The root value spans 0.0028 — 0.14% of the payoff range — and is not even
monotone in the baseline. **The baseline cannot be calibrated from solved
values.** The equilibrium policy does move: drop-immediately weight swings 34%
relative across the range. So the constant is chosen on principle, frozen, and
never revisited; and a policy comparison across a baseline change is not
like-for-like.

`0.95` is chosen because both players in the source spend *planned* moves on
fresh deaths, which rules out anything that reads as a gamble; because STL's own
implied `P(0,0)` is 0.992; and because it leaves headroom for the physicality
multiplier it absorbs.

## Migration status

The DTH and abstract executable models, their Rust parity arm, and their
production tablebases now use this frozen surface. Changing the model
invalidated the prior value-bearing artifacts as intended; the DTH learned
artifacts were retired and the opening census was rerun under the new schema.
The leap-aware STL public-engine integration remains a separate pending
migration and is not used by the DTH or abstract artifacts described here.
Scope, sequencing, and measured costs are in
[`REGENERATION_PLAN.md`](REGENERATION_PLAN.md).

| Site | Current | Required |
|---|---|---|
| `src/dth/solver.py:33-39` | frozen linear/geometric surface | complete |
| `src/dth/solver.py:185-203` | source-derived rule hash | complete |
| `src/abstract/rules.py:147-177` | frozen model kind, baseline 0.95 | complete |
| `src/abstract/packed_tablebase.py:45-49` | frozen model-kind mapping | complete |
| `src/crates/abstract_solver/src/lib.rs:57-88` | frozen Rust arm | complete; parity verified |
| `src/stl/engine/game.py:222-231` | cubic, cardiac, referee, physicality | linear dose term |

`src/dth/solver.py`'s `solver_schema_hash()` now hashes the source of every
rule function plus `_FAILURE_DEAD_MIN_ST`, following the source-byte approach
used by `src/abstract/artifacts.py:31-41`. Replacing `revival_model` or changing
the failure threshold therefore changes the DTH schema hash and rejects stale
artifacts.

Per the root `AGENTS.md`, the rules change, canonical docs, evidence citations,
schemas, and tests move together in one commit.

## Modeling boundary

No numerical revival probability appears anywhere in the source material. The
evidence fixes the capacity, the dose composition, the eligibility inequalities,
and the sign of both effects. It does not fix `0.95`, `0.75`, the linear dose
shape, or the 144.3-second half-life.

These are declared solver constants. They must be cited as such in any writeup,
must be versioned together, and must never be described as documentary odds.
