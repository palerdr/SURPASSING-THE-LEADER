# Regeneration plan for the frozen revival model

Adopting [`REVIVAL_MODEL.md`](REVIVAL_MODEL.md) changes probabilities but not
structure. This file is the work list, ordered, with measured costs.

## The one fact that determines scope

**Eligibility is unchanged.** The new surface is zero on exactly the same set as
the old one — verified against `dth.solver.survive_injection` on all `300 x 301`
`(s, t)` pairs with zero mismatches. Since eligibility alone decides which
children exist, the reachable game graph is bit-identical before and after.

So every artifact splits cleanly:

| | survives the freeze | must be regenerated |
|---|---|---|
| **structural** | reachable state sets, packed index spaces, potential/rank orderings, edge counts, quotient class counts | — |
| **value-bearing** | — | values, policies, saddle gaps, win probabilities, tablebases, network checkpoints, readiness reports |

The catch: structural artifacts *embed a rules digest*, so they will be
**rejected on reopen** even though their contents are still correct. That is the
fail-closed contract working as designed, and it is why "survives" does not mean
"reusable in place".

## Prerequisite, before any regeneration

`src/dth/solver.py:185-203` `solver_schema_hash()` hashes a hand-written
description of the rules, not the rules. Swapping `revival_model` leaves it
byte-identical, so **DTH artifacts built under the old model would silently pass
validation under the new one**. Fix this in the same commit as the model change,
by hashing the source of the rule functions (`revival_model`,
`survive_injection`, `transition`, `failed_check_dose`, `overflow`, `st`,
`successful_check`) plus `_FAILURE_DEAD_MIN_ST`.

`src/abstract/artifacts.py:31-41` `digest_files()` already does this correctly.
Copy the approach; do not import across the project boundary.

Order: schema hash → model change → regeneration. Reversing the first two makes
the whole exercise unverifiable.

## abstract

Both rulesets are fully solved for a root value, so this is a clean re-solve.

### What the builder will do

`PackedTablebaseBuilder._config_digest` (`packed_tablebase.py:117-127`) includes
`rules.revival_model_metadata`. An existing `build-progress.json` therefore
raises `ValueError("checkpoint configuration does not match requested ruleset")`
— it does **not** resume, and it does not silently continue. Expect a hard stop,
not a wrong answer.

Add a new model kind rather than mutating the existing one, matching the
precedent set by `bucket*_ttd_curve95`:

1. `rules.py` — add `FROZEN_REVIVAL_MODEL = "linear_st_geometric_ttd_v1"`, a
   `ttd_decay_per_death_dose` field, the `revival_probability` branch, and the
   `revival_model_metadata` entries. The dose factor is unchanged from the
   current unified model; only the TTD term and baseline change.
2. `rules.py` — add `bucket6_frozen95` and `bucket12_frozen95` rulesets and make
   them the CLI defaults. New ids mean new output directories, so the existing
   `*_unified80` artifacts are never clobbered.
3. `packed_tablebase.py:45-49` — add the kind to `_rust_revival_model_code`.
4. `src/crates/abstract_solver/src/lib.rs:73-84` — add the `match model_kind`
   arm. **The Rust extension is not currently installed** (`import
   abstract_solver` → `ModuleNotFoundError`), so Python is the only live
   backend until `maturin develop --release` is run. A missing arm returns
   `f64::NAN`, which will surface as a build failure rather than a wrong value.
5. `tests/test_rust_parity.py` — extend to cover the new kind.

### Cost

| Phase | bucket6 | bucket12 |
|---|---|---|
| reachability | ~2 s | ~29 s |
| states | 576,270 | 8,870,160 |
| mixed LP states | 481,618 | 8,132,566 |
| backups, pure Python | ~10 min | ~3–4 h |
| backups, Rust kernel | minutes | ~1 h (per `MODEL.md`) |

Reachability timings are measured on a from-scratch BFS of the same graph; the
backup estimate for bucket6 is measured end to end (576,270 states, 570 s,
including 481,618 HiGHS solves). Reachability is negligible in both cases — the
entire cost is the LP sweep, so there is no value in trying to preserve the
reachability arrays across the freeze.

### Verification

Re-solve `bucket6_frozen95` first; it is small enough to check by hand. Two
invariants must hold against the existing `bucket6_unified80` artifact:

- reachable state count still exactly 576,270, and the packed index set
  identical;
- pure-saddle / mixed split may move, and every value, policy and gap will move.

The root value will change from `0.10400516101482249`. It is not a regression
if it does — under the baseline sweep in `REVIVAL_MODEL.md` the root value moved
only 0.0028 across a 0.14 span in the baseline, so expect a small shift, and
treat a large one as a bug.

## dth

Nothing here is worth preserving, and most of it is not worth regenerating.

### Inventory

474 files, 166.6 MB under `dth/artifacts/` and `dth/checkpoints/`:

| Category | Files | MB |
|---|---:|---:|
| census / tablebase sqlite | 84 | 118.4 |
| other json reports | 161 | 22.9 |
| readiness reports | 37 | 10.9 |
| npz datasets / tablebases | 46 | 6.5 |
| network checkpoints | 69 | 6.2 |
| other, bundles, logs | 77 | 1.6 |

**First, move them.** All 66 DTH configs and `.gitignore:35-36` point at
`src/dth/artifacts/` and `src/dth/checkpoints/`; the data is still at the
pre-src-layout paths. Until that is fixed, any run silently starts fresh instead
of resuming, and graphify scans the artifacts as source.

### The census

`exact_opening_consolidated_v1.sqlite` (24.2 MB, six hours of wall clock) is
*semantically valid* after the freeze — its `states` and `rank_layers` tables
are pure structure. It will still be rejected, because `metadata.rules_schema_hash`
will no longer match once the digest is derived from source.

Do not re-stamp it. Re-run it: the six hours were almost entirely the missing
index. `complete_game_dependencies` costs 309 µs/state, so the 309,517
expansions that took six hours are **~96 seconds of CPU** once
`CREATE INDEX ix_frontier ON states(census_status, damage_rank, state_id)` exists.
Add the index in the same commit and the census is not a fixture worth
protecting.

### The learned artifacts

The h3 tablebases, the 69 checkpoints, and the 37 readiness reports should be
**deleted, not regenerated**. Regenerating them faithfully reproduces a broken
experiment:

- `boundary_tablebase_h3_v1.npz` is 98.3% remaining-horizon-1 rows, and 71% of
  its labels are exactly 0.0 — provably `0 <=> checker_st < 240`, a threshold on
  one coordinate. 39 distinct label values across 83,194 rows.
- Every readiness metric is bit-identical across MCTS budgets 0/256/1024/4096,
  because the root warm-up sweeps all 3,600 joint cells and the branching factor
  is 3,600 per ply. No result in that ladder measured search.
- All eight final `best.pt` files are byte-identical to a checkpoint of a
  different architecture (`shutil.copyfile` at `train.py:2080-2081`).

Regenerate the learning stack only after the 61-output transition-class head is
tried; until then there is no target worth spending LP time on.

### What to actually rebuild

1. The census, to whatever rank you want, with the index in place.
2. Nothing else, until the exact track has a value to anchor against.

### Execution record

The ordered work has now been executed. The abstract production tablebases were
regenerated under `linear_st_geometric_ttd_v1`: `bucket6_frozen95` certifies
576,270 reachable states and `bucket12_frozen95` certifies 8,870,160. The DTH
learned outputs were deleted (209 files, 62,599,288 bytes including the 69
network checkpoints), and the opening census was rerun at the same bounded
configuration. Its new report records 183,647 expansions and 757,337 persisted
states, with `ix_frontier` and the source-derived rules hash present in the
new SQLite artifact.

## Ordering

```text
1  fix solver_schema_hash to derive from rule source        (blocks everything)
2  add the SQLite index to dth/tablebase.py schema
3  move dth/artifacts and dth/checkpoints under src/dth/
4  land the model in dth/solver.py and abstract/rules.py
5  add the Rust arm; maturin develop --release; run parity
6  re-solve bucket6_frozen95, verify 576,270 states
7  re-solve bucket12_frozen95
8  delete the DTH learned artifacts; re-run the census
```

Steps 1, 2 and 4 must land in a single commit — the root `AGENTS.md` requires
rules, docs, evidence, schemas and tests to move together, and steps 1 and 4
are unsound if separated.
