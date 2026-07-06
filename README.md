# Surpassing The Leader

This is a Python research/playground implementation of the *Usogui* "Drop the
Handkerchief" death game. The long-term vision is still a faithful playable duel
where the human plays Baku against a layered Hal, but the codebase currently
provides something more specific:

- a deterministic rules engine for the game clock, cylinders, death, revival,
  and leap-second timing;
- exact-second minimax/search tools for proving local positions;
- generated tablebases and value targets used to train/evaluate a small Hal
  value model;
- a CLI/playable baseline and scripted teachers for experiments.
- a playable game where a human plays Baku against a scripted/searched Hal;
- a small solved-Hal evaluator trained against an exact finite-horizon minimax
  oracle. The current local best evaluator is a low-impact Tier A auxiliary
  fine-tune that clears the strict held-out gate at **overall MSE 0.05591**
  and beat the previous default by +15 wins on a deterministic 240-game ladder.

The active continuation point is `scripts/run_tier_a.py`.

## Current Direction

The project started as a rules simulator, then grew into scripted opponents,
route-math tests, self-play experiments, exact minimax solvers, value-net
training, and finally tablebase construction. That history left several valid
but overlapping paths in the repo.

The direction now is:

1. Keep the engine exact and testable.
2. Use classical exact methods first: minimax LPs, backward induction,
   tablebases, and pinned scenario audits.
3. Use learned models only as approximation tools around those exact anchors.
4. Build Hal upward from inspectable strategy layers, not from pure end-to-end
   RL alone.

So the answer to "is this RL?" is: not primarily. RL/self-play exists in the
repo, but the strongest current path is classical exact solving plus tablebase
generation, with neural value models or MCTS used as accelerators where a full
exact solve is too expensive.

## Repository Map

| Path | Role |
|---|---|
| `src/` | Core game engine: players, referee, constants, half-round resolution, clock. |
| `environment/cfr/` | Exact/search solver core: minimax LPs, matrix-game MCTS, tablebase scenarios. |
| `training/tablebase/` | Certified interval tablebase epochs used by `run_tier_a.py`. |
| `training/` | Target generation, value-net training, calibration, audits, tournaments. |
| `hal/` | Playable Hal wrappers, value net, search adapter, legacy CanonicalHal support. |
| `environment/` | Gym wrapper, route features, legal actions, scripted opponents, abstractions. |
| `scripts/` | Human entry points. Start here for runs. |
| `tests/` | Regression suite for engine, solver, route math, AI wrappers, and training glue. |
| `docs/` | Canon sources, architecture notes, solver charter, reports, whitepaper. |
| `checkpoints/`, `models/`, `logs/`, `demos/` | Generated artifacts. Large and mostly ignored. |

## Continue From Tier A

Use Python 3.12+ and `uv` for dependency management:

```bash
uv sync --dev
uv run pytest -q
```

All commands below are run from the project root containing `src/`,
`environment/`, `training/`, and `scripts/`. Use `uv run python ...` when you do
not have the virtual environment activated.

Dependencies live in `pyproject.toml` and are locked by `uv.lock`. Imports still
use the stable top-level packages (`src`, `environment`, `training`, `hal`) so
historical scripts and checkpoints remain usable.

## Solver Shape

The system is two towers that meet at one gate.

**Ground-truth tower (the oracle).** `environment/cfr/exact.py` solves the game *exactly* over a
finite horizon: at every simultaneous move it builds the 60×60 (61 in the leap window) payoff matrix
and solves the zero-sum game with a linear program (`scipy.optimize.linprog`), recursing through
real engine-derived chance branches (survival rolls, death durations, clock advancement). Leaf
values are terminal-only (+1/−1/0 from Hal's perspective). A **structural firewall**
(`tests/test_cfr_firewall.py`) AST-walks the namespace and fails the build if any identifier hints
at bucketing/abstraction, or if shaping/observation modules are imported — so this tower can never
silently become approximate. `tablebase.py` adds hand-verified **pinned scenarios** whose exact
value is known in closed form; they are the calibration anchors.

**Approximation tower (the learned evaluator).** A small `ValueNet` (`hal/value_net.py`) learns to
predict the oracle's value from a 23-dim engine-derived feature vector. `environment/cfr/mcts.py`
is a matrix-game MCTS that can use that net as its leaf evaluator (`ValueNetEvaluator`) to search
deeper than the exact tower can afford. The net never enters the exact recursion — it sits strictly
behind the `LeafEvaluator` seam.

**The loop (AlphaZero-style).** Generate exact-LP labels → train the net → use the trained net as
the MCTS leaf for the next generation's bootstrap labels → gate against a held-out ruler → accept
only if it strictly beats the previous generation. The current best evaluator is the product of this
loop hitting its data-imposed ceiling and then being pushed past it by hardening the ruler, widening
the net, and rebalancing the training classes (full history in `docs/SOLVER_EXTENSION_GOAL.md §8`).

---

## Generated Training Data

There is nothing to download — the corpus *is* the exact solver's output. Two artifacts: a training
corpus and a disjoint held-out ruler.

The exact-LP training corpus is produced by `scripts/run_phase_g_corpus.py`,
which sweeps a stratified grid densified around the 239/240/241 overflow trigger
and the near-overflow band. This step takes hours and parallelizes across
workers.

Smoke the Tier A path without starting the long run:

```bash
uv run python scripts/run_tier_a.py --help
uv run python scripts/run_tier_a.py --stage 1 --limit 1 --workers 1
```

The real run is long and writes to `checkpoints/tablebase/tier_a/`:

```bash
uv run python scripts/run_tier_a.py --workers 20
```

`run_tier_a.py` builds:

- 480 one-death epochs: `d1_{hal|baku}_{60..299}.npz`;
- one zero-death opening epoch: `d0.npz`;
- `manifest.json` with hashes.

## Development Narrative

The project has moved through these phases:

1. **Rules engine.** The initial value was making the gamble executable:
   check/drop resolution, Squandered Time, cylinder overflow, death/revival,
   referee fatigue, and leap-second clock behavior.
2. **Route and character modeling.** The next layer added LSR/deviation math,
   scripted Baku/Hal teachers, and tests for canonical route pressure. This
   made the project more than a generic timing game.
3. **Approximate playable Hal.** `hal/` and the early self-play scripts added a
   playable/searching Hal with a small value net. Useful, but not enough as the
   foundation for "Perfect Hal."
4. **Exact solver tower.** `environment/cfr/` became the rigorous path:
   exact-second joint actions, minimax LPs, engine-derived chance branches, and
   tests that prevent bucketed or shaped logic from leaking into exact solving.
5. **Training and gates.** `training/` generated exact labels, trained small
   value models, and enforced calibration gates against held-out/pinned states.
6. **Tablebase push.** `training/tablebase/` and `scripts/run_tier_a.py` are the
   current next step: build certified post-leap epoch tables so the solver has a
   stronger exact backbone.

Once Tier A exists, the project is closer to a real "Hal" because the AI can
reason from exact arithmetic instead of only local heuristics or self-play
imitation. The eventual playable experience should combine that exact backbone
with belief, perception, persona, and memory layers.

## What To Prefer

- Prefer `src/` for engine truth.
- Prefer `environment/cfr/` and `training/tablebase/` for exact solver work.
- Prefer `training/` for generated labels, gates, and audit logic.
- Treat `hal/` self-play and older scripts as useful experiments, not the core
  proof path.
- Keep generated artifacts out of source review unless a specific artifact is
  the subject of the work.

## Quick Checks

```bash
uv run pytest tests/test_clock.py tests/test_death_mechanics.py tests/test_tablebase_epoch.py
uv run pytest --collect-only -q
```

---

# 4. Training the evaluator

## 4.1 One generation

`scripts/run_gen_iteration.py` runs a single generation end to end: MCTS-bootstrap labels from the
incoming checkpoint → merge anchors → class-rebalanced training → calibration gate vs the held-out
ruler. It exits non-zero if the gate fails.

```bash
python scripts/run_gen_iteration.py \
  --in-checkpoint   checkpoints/gen_phaseG_iter3/best.pt \
  --out-dir         checkpoints/gen_next/ \
  --out-targets     checkpoints/gen_next_targets.npz \
  --anchor-targets  checkpoints/phase_g_targets.npz \
  --held-out-targets checkpoints/ceiling_holdout_clean.npz \
  --iterations 1000 --epochs 150
```

Key flags (see `--help` for all):

- `--hidden-dim {64,128,192}` — net width. 64 = 13.7K params (original), 192 = 65.4K (current best).
- `--split-interior` — give the interior pins their own training class so the ~660 boundary ±1 pins
  don't drown them and saturate the net toward ±1.
- `--tablebase-weight`, `--interior-weight`, `--terminal-weight`, `--horizon-weight` — per-source
  loss weights.
- `--per-source-mse-threshold SOURCE:VALUE` (repeatable) — adds a per-class ceiling to the gate, so
  it catches the "overfit the tablebase, collapse h2/h3" failure mode that a tablebase-only gate misses.

## 4.2 The full ceiling run (orchestrated, resumable)

`scripts/run_ceiling.py` chains the whole pipeline — corpus regen (memory-bounded chunks) → ruler →
hidden=192 train + gate → global invariant gate — and is the recommended way to reproduce the result
from scratch. It is OOM-safe (swap watchdog, worker back-off) and resumable (any stage whose output
already exists is skipped, so a gate failure never re-runs the multi-hour corpus):

```bash
stdbuf -oL -eL python scripts/run_ceiling.py 2>&1 | tee checkpoints/ceiling_run.log
```

New self-play data should be generated through the solver-MCTS path so policy labels match deployed
`SolverAgent.policy()` semantics and save as the normal training schema with source `self_play_mcts`:

```bash
python scripts/run_solver_self_play.py \
  --hal-checkpoint checkpoints/current/best.pt \
  --opponent pattern_reader \
  --games 4 --iterations 30 \
  --out checkpoints/self_play_mcts_smoke.npz
```

The legacy self-play value-net loop (the `hal/` tower, scripted-opponent self-play rather than the
exact oracle) is retained only for old experiments:

```bash
python scripts/run_alphazero_loop.py --iterations 3 --games 500 --epochs 20 --depth 1
python scripts/run_deep_training.py            # same, with best-checkpoint selection
```

## 4.3 The calibration gate

A generation is **accepted** only when every condition holds on the held-out ruler (enforced by
`training/bootstrap_loop.py::enforce_calibration_gate`):

- overall held-out MSE strictly below the previous accepted generation's,
- every per-source MSE under its threshold (`tablebase ≤ 0.01`, `tablebase_interior ≤ 0.05`,
  `exact_horizon_2/3 ≤ their ceilings`, `terminal ≤ 1.0`),
- mean unresolved-probability per source within bounds.

Never weaken a gate to pass it — editing the firewall, lowering a threshold, or relaxing a tolerance
is an automatic fail per the charter. If the engine contradicts a construction assumption, stop and
record it.

## 4.4 Reproducing the current best evaluator

hidden=192 with the interior split + boundary-tablebase loss weight, graded on the clean grafted
ruler (the config that flipped the overall goal flag — `docs/SOLVER_EXTENSION_GOAL.md`, 2026-06-08):

```bash
python scripts/run_gen_iteration.py \
  --in-checkpoint   checkpoints/gen_phaseG_iter3/best.pt \
  --out-dir         checkpoints/gen_ceiling_tbw15_wd1e-4/ \
  --out-targets     checkpoints/gen_ceiling_tbw15_targets.npz \
  --anchor-targets  checkpoints/ceiling_corpus.npz \
  --held-out-targets checkpoints/ceiling_holdout_clean.npz \
  --hidden-dim 192 --split-interior --interior-weight 2 \
  --tablebase-weight 15 --weight-decay 1e-4 \
  --prev-gen-holdout-mse 0.06986 \
  --per-source-mse-threshold tablebase_interior:0.05
```

Result: overall **0.05683**, tablebase 0.00239, tablebase_interior 0.00246, h2 0.02031, h3 0.00290 —
all five charter success criteria met. (The bootstrap step is seeded-deterministic — a rerun with the
same seed and the same numpy/scipy versions is bit-equivalent.)

## 4.5 Tier A auxiliary fine-tune now used by default

After the Tier A frontier event, the accepted local default is a conservative fine-tune from
`checkpoints/gen_ceiling_tbw15_wd1e-4/best.pt` over a 50k low-width Tier A target file:

```bash
python scripts/run_tier_a_targets.py --out checkpoints/tier_a_targets_50k_w001.npz \
  --limit 50000 --max-width 0.01 --runtime-width 0.0 --policy-horizon 1 \
  --include-d1 --verify-manifest --seed 0
```

Create the merged generation corpus (`checkpoints/gen_tier_a_aux_50k_w001_targets.npz`) with the
Tier A frontier event's generated `run_gen_iteration.py` command, then run the accepted low-impact
fine-tune:

```bash
python scripts/train_saved_corpus_gate.py \
  --targets checkpoints/gen_tier_a_aux_50k_w001_targets.npz \
  --out-dir checkpoints/gen_tier_a_aux_50k_w001_ft_lr5e-6_tw001_pw0_iw100_e5 \
  --held-out-targets checkpoints/ceiling_holdout_clean.npz \
  --epochs 5 --hidden-dim 192 \
  --init-checkpoint checkpoints/gen_ceiling_tbw15_wd1e-4/best.pt \
  --learning-rate 5e-6 --weight-decay 1e-4 \
  --tablebase-weight 15 --interior-weight 100 \
  --tier-a-weight 0.001 --tier-a-policy-weight 0.0 \
  --prev-gen-holdout-mse 0.0568296855
```

Calibration result: overall **0.05591**, tablebase 0.00247, tablebase_interior 0.00436, h2 0.02046,
h3 0.00269. Deterministic checkpoint ladder comparison at 30 search iterations, 20 games/opponent,
seeds 0,1,2: candidate **170/240** vs previous default **155/240**. The heavier 50k Tier A run with
`tier-a-weight=0.1` was rejected despite better held-out MSE because it regressed against
`pattern_reader`.

Certified exploitability comparison with Tier A frontiers, depth 1, seeds 0,1,2: the fine-tune is
certified better on all three `postleap_230` probes, but the one-death
`postleap_d1_hal_120_230` probes still overlap and trend mixed. A focused one-death Hal target file
was generated from the supported d1-Hal artifacts:

```bash
python scripts/run_tier_a_targets.py --out checkpoints/tier_a_targets_d1_hal_20k_w001.npz \
  --source tier_a_d1_hal --limit 20000 --max-width 0.01 --runtime-width 0.0 --policy-horizon 1 \
  --include-d1 --death-filter d1_hal --ttd-min 100 --ttd-max 140 \
  --verify-manifest --seed 0
```

The filter exhausted at **8,406** accepted rows (`d1_hal_113/120/140`, no misses). Two low-impact
fine-tunes using that focused file passed calibration but were rejected by the deterministic
240-game ladder versus the current default (`-11/240` and `-15/240`). Do not promote d1-Hal append
fine-tunes without a positive checkpoint ladder and exploitability check.

For safer targeted probes, `train_saved_corpus_gate.py` supports `--trainable-parts value_head`,
which freezes the trunk and policy head. This preserves policy priors exactly, but the tested
d1-Hal value-head-only variants were still rejected by certified exploitability precheck: the
append variant introduced one fresh-postleap certified regression, and the d1-only variant introduced
two fresh-postleap certified regressions despite strong held-out MSE.

It also supports `--reference-checkpoint` + `--value-distill-weight` for trust-region value
fine-tuning. A d1-only value-head trust-region run removed certified exploitability regressions and
passed calibration (`0.05549`), but the promotion event rejected it on live strength (`-14/240`
ladder delta). Current default remains unchanged.

Promotion is now an explicit evidence-bundle event. A candidate must pass calibration, beat the
deterministic checkpoint ladder, avoid certified exploitability regressions, and supply a
policy-drift/readability report that stays within configured TV/entropy thresholds. The event also
verifies that the calibration, ladder, exploitability, policy-drift, and optional trace-policy-gate
reports all name the same candidate checkpoint, and that the comparison reports use the same
champion baseline:

```bash
python scripts/run_checkpoint_promotion_event.py \
  --calibration-report checkpoints/gen_tier_a_aux_50k_w001_ft_lr5e-6_tw001_pw0_iw100_e5/gate_report.json \
  --ladder-report checkpoints/checkpoint_ladder_compare/tier_a_50k_ft_lowimpact_seeds012_g20_seeded_report.json \
  --exploitability-report checkpoints/checkpoint_exploitability_compare/tier_a_50k_ft_seeds012_report.json \
  --policy-drift-report checkpoints/checkpoint_policy_drift/tier_a_50k_lowimpact_vs_old_seeds012_report.json
```

The post-promotion next-run event consumes the accepted promotion report, rejected d1-Hal reports,
and runtime comparison and emits the next long job as a reproducible artifact:

```bash
python scripts/run_solver_next_event.py \
  --accepted-promotion-report checkpoints/checkpoint_promotion_event/tier_a_50k_lowimpact_accept_report.json \
  --rejected-promotion-report checkpoints/checkpoint_promotion_event/tier_a_d1hal_append_reject_report.json \
  --rejected-promotion-report checkpoints/checkpoint_promotion_event/tier_a_d1hal_valuehead_trust_reject_report.json \
  --runtime-report checkpoints/tier_a_runtime_compare/current_default_exact_seeded_seeds012_g20_report.json \
  --out checkpoints/solver_next_event/tier_a_post_promotion_next_run_report.json
```

Current event verdict after executing the d0 and MCTS-refresh probes:
`needs_pattern_reader_hardening_generation`. Static Tier A append runs and a fresh 300-iteration
MCTS refresh improved the held-out ruler but failed live promotion, with pattern_reader the
recurring regression. An unbounded critical-resolve generation attempt ran for four hours without
producing a target corpus, so `run_gen_iteration.py` now supports
`--bootstrap-critical-only --bootstrap-max-states N`. The next run should use that bounded
critical-only path before spending another full overnight job.

When a pattern_reader smoke has been run for a generated candidate, pass it back with
`--canary-report`. The generation event now treats the report as a gate: by default
`--canary-min-wins-delta 0` rejects any canary regression even if held-out calibration improved.

---

# 5. Evaluation & validation

**Global invariant gate** — must be green before any solver change is called done:

```bash
python -m pytest tests/test_cfr_firewall.py tests/test_cfr_rigorous_exact.py \
                 tests/test_cfr_selective_search.py tests/test_cfr_tablebase.py \
                 tests/test_calibration.py tests/test_audit_pack.py -q
```

**Full suite** (553 tests — engine rules, route math, solver, gate, teachers):

```bash
python -m pytest tests/ -q
```

**MCTS convergence benchmark** — root value vs iteration budget on the pinned scenarios:

```bash
python scripts/benchmark_mcts_convergence.py
```

**Canonical-line walkthrough** — runs Hal through the R1T1→R9T2 canonical sequence against the LSR
teacher and prints per-turn action, MCTS root value, and principal line:

```bash
python scripts/validate_against_manga.py
```

**Scripted-opponent evaluation** of a trained policy:

```bash
python scripts/evaluate.py --model models/self_play/hal_gen3 --role hal --opponent safe --games 50
```

---

# 6. Playing the game

Two-player terminal game — you are Baku, the AI is Hal. Dropper input is hidden so the checker can't
peek:

```bash
python scripts/play_cli.py
```

`scripts/play_cli.py` defaults to `SolverAgent` (trained value net + matrix-game MCTS). If the
generated Tier A tablebase is present, you can enable its exact-only runtime lookup for
experiments (only collapsed intervals with width 0.0 are used directly by default):

```bash
python scripts/play_cli.py --use-tier-a
```

Tier A runtime is **not** the default play path yet. In the current evidence it is useful as a
frontier/diagnostic tool, but runtime leaf replacement has not produced a robust ladder win:
`width<=0.01` regressed on a 3-seed 360-game ladder (`baseline 236/360`, Tier A `222/360`), and
exact-only `width=0.0` is neutral under the promoted default on the deterministic seeded 240-game
ladder (`170/240` vs `170/240`; 60 exact hits, 62 wide hits). Re-measure before promoting it:

```bash
python scripts/compare_tier_a_runtime.py --agent-iterations 30 --games 20 --seeds 0,1,2 --tier-a-width 0.0
```

The stronger current use of Tier A is exploitability measurement: it replaces the old `[-1,+1]`
frontier bracket with certified intervals on supported post-leap states, without changing runtime
play:

```bash
python scripts/run_exploitability.py --policy agent --iterations 30 --scenario postleap_230 --depth 1 --tier-a-frontier
python scripts/run_exploitability.py --policy agent --iterations 30 --scenario postleap_d1_hal_120_230 --depth 1 --tier-a-frontier
```

For a repeatable go/no-go event that compares baseline frontiers against Tier A frontiers and emits
the next long generation command:

```bash
python scripts/run_tier_a_frontier_event.py --checkpoint checkpoints/gen_ceiling_tbw15_wd1e-4/best.pt
```

Checkpoint candidates can be compared directly with:

```bash
python scripts/compare_checkpoints_ladder.py \
  --champion-checkpoint checkpoints/gen_ceiling_tbw15_wd1e-4/best.pt \
  --candidate-checkpoint checkpoints/gen_tier_a_aux_50k_w001_ft_lr5e-6_tw001_pw0_iw100_e5/best.pt \
  --agent-iterations 30 --games 20 --seeds 0,1,2
```

For certified exploitability comparisons:

```bash
python scripts/compare_checkpoints_exploitability.py \
  --champion-checkpoint checkpoints/gen_ceiling_tbw15_wd1e-4/best.pt \
  --candidate-checkpoint checkpoints/gen_tier_a_aux_50k_w001_ft_lr5e-6_tw001_pw0_iw100_e5/best.pt \
  --iterations 30 --depth 1 --seeds 0,1,2
```

For a cheaper explanation of where search behavior changed before paying for a ladder:

```bash
python scripts/compare_checkpoints_policy_drift.py \
  --champion-checkpoint checkpoints/gen_tier_a_aux_50k_w001_ft_lr5e-6_tw001_pw0_iw100_e5/best.pt \
  --candidate-checkpoint checkpoints/gen_tier_a_aux_50k_plus_d1hal8k_ft_lr5e-6_tw001_d1w01_d1pw02_iw100_e5/best.pt \
  --iterations 200 --seeds 0,1,2
```

`run_checkpoint_promotion_event.py` requires this report by default as a policy-readability gate:
severe root-policy TV drift, mean TV drift, entropy collapse, or candidate entropy below the
champion's own floor rejects the candidate before promotion. Use the deployment-like 200-iteration
budget for this gate; cheaper 30-iteration probes can miss the policy movement that
`pattern_reader` exploits. Promotion also requires the ladder report to include the
`pattern_reader` rung by default; use `--required-ladder-opponent` for any additional mandatory
canaries. Targeted trace policy gate reports can also be supplied with
`--trace-policy-gate-report`; add `--require-trace-policy-gate-report` for candidates whose known
failure modes are trace-local.

Use `--agent legacy` for `CanonicalHal` (the older forward-search baseline). `scripts/play_vs_cfr.py`
plays against a precomputed equilibrium-strategy table (the legacy bucketed CFR baseline).

---

# 7. Game rules (quick reference)

- Two players alternate **Dropper** (drops once) and **Checker** (turns once) each half-round; turns
  are 60 s (61 s in the leap-second window).
- **Successful check** (`check_time ≥ drop_time`): `ST = check_time − drop_time` is added to the
  checker's cylinder. Cylinder ≥ 300 s → injection. **Failed check**: 60 s penalty + the whole
  cylinder is injected → death sequence.
- **Survival** on injection is a product of four independent factors (base curve × cardiac modifier ×
  referee fatigue × physicality). Survive → cylinder resets; fail → permanent death → loss.
- The clock is absolute (seconds since 8:00 AM); the leap second is at 8:59:60. **Deaths shift round
  timing, which shifts who is Dropper during the leap-second turn** — the core strategic lever the
  whole game revolves around.

Full formal rules are in the root `CLAUDE.md`; the strategic model is in `AGENTS.md` and the source
PDFs under `docs/canon/`.

---

# References

- von Neumann (1928) minimax / zero-sum matrix games — the exact tower's LP at each node.
- Silver et al., *AlphaZero* — the generate → train → re-evaluate policy-improvement loop.
- Moravčík et al., *DeepStack* / Brown & Sandholm, *Libratus*, *ReBeL* — depth-limited solving with a
  learned leaf evaluator behind a search.
- `docs/whitepaper/stl_solver_whitepaper.tex` — the project's own write-up.
- `docs/SOLVER_EXTENSION_GOAL.md` — the governing gate contract and append-only progress log.
