# Surpassing The Leader — an exact-second solver for "Drop the Handkerchief"

This repo is a perfect-information solver and AlphaZero-style policy-improvement loop for the
"Drop the Handkerchief" death game from the *Usogui* "Surpassing The Leader" arc. It targets two
products from one engine:

1. **A playable game** — a human plays Madarame Baku in the terminal against a scripted/searched Hal.
2. **A solved "Hal" evaluator** — a small value net, trained against an exact finite-horizon minimax
   oracle, that scores any game state from Hal's perspective. The current best evaluator clears the
   full strict calibration gate on a held-out ruler at **overall MSE 0.05683** (interior 0.0025,
   boundary tablebase 0.0024).

The design vision lives in [`AGENTS.md`](AGENTS.md); the solver's pass/fail charter lives in
[`docs/SOLVER_EXTENSION_GOAL.md`](docs/SOLVER_EXTENSION_GOAL.md); the formal rules are summarized in
the root `CLAUDE.md`. If you run into setup/code issues or have questions, contact `jdewitte632@gmail.com`.

> **Note on artifacts.** Everything under `checkpoints/`, `models/`, and `docs/reports/` is
> gitignored. A fresh clone has the code and the pinned scenarios but **no corpus and no trained
> net** — you regenerate them with the commands in §3–§4. There are no external datasets or
> pretrained backbones to download; all training data is *generated* by exact LP solving.

---

# 1. Setup

## 1.1 Compute

There is no GPU requirement and no cloud step. Everything runs on a single CPU workstation — the value
net is tiny (≤ 70K params, < 5 ms/state on CPU), and the expensive part is the exact LP solver, which
is CPU- and memory-bound, not compute-bound. The reference machine:

- 32 logical cores (corpus generation parallelizes across workers)
- 24 GB RAM + 16 GB swap
- Linux / WSL2

**WSL2 users — set memory limits before any long run.** The full corpus regen exhausts a 12 GB cap
and the OOM-killer cascades through tmux and the worker pool. Put this in `C:\Users\<you>\.wslconfig`
and restart WSL (`wsl --shutdown`):

```ini
[wsl2]
memory=24GB
swap=16GB
processors=32
```

The exact solver memoizes substates in a **bounded LRU cache** (`STL_SOLVE_CACHE_MAXSIZE`, default
30000) so wide-grid generation is memory-safe by construction. The orchestrated ceiling run
(§4.2) also chunks the grid and runs a `/proc/meminfo` watchdog that backs off worker count
(8→6→4→2) on swap pressure.

**Long runs should be detached.** Generation and ceiling runs take hours; run them under `tmux` with
line-buffered logging so a dropped terminal never kills the job:

```bash
tmux new -s stl
stdbuf -oL -eL python scripts/run_ceiling.py 2>&1 | tee checkpoints/ceiling_run.log
# detach with Ctrl-b d ; reattach with: tmux attach -t stl
```

## 1.2 Environment

Python 3.12. Clone, create a virtualenv, install pinned deps:

```bash
git clone <your-remote> STL && cd STL/STL
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # numpy, scipy, torch, gymnasium, sb3-contrib, pytest
```

All commands below are run from the `STL/` project directory (the inner one, containing `src/`,
`environment/`, `training/`, `scripts/`) with the venv active.

## 1.3 Repo layout & import convention

There is no `pyproject.toml`. Imports rely on `sys.path` insertion; scripts add the project root at
the top, and tests do it via `conftest.py`. Engine imports are `from src.Game import Game`, solver
imports are `from environment.cfr.exact import ...`, etc.

| Path | What lives there |
|---|---|
| `src/` | Pure-Python rules engine (clock, cylinder, death/revival, referee). Deterministic given a seed. |
| `environment/cfr/` | **Rigorous solver core** — exact-second minimax LP, matrix-game MCTS, tablebase, leaf evaluators. See [`environment/cfr/MODULES.md`](environment/cfr/MODULES.md). |
| `training/` | Corpus generation, value-net training, calibration gate, audit pack. |
| `hal/` | The tiny `ValueNet` + feature extraction, plus the legacy playable `CanonicalHal`. |
| `environment/` (top) | Gymnasium wrapper, observations, route math, opponents (scripted teachers). |
| `scripts/` | Every entry point (generation, training, evaluation, play). |
| `tests/` | 553-test pytest suite, including the firewall and gate invariants. |
| `docs/canon/` | Source PDFs (rules + strategy docs). Gitignored. |

---

# 2. The solver

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

# 3. Generating training data

There is nothing to download — the corpus *is* the exact solver's output. Two artifacts: a training
corpus and a disjoint held-out ruler.

## 3.1 The exact-LP training corpus

`scripts/run_phase_g_corpus.py` sweeps a stratified grid (densified around the
239/240/241 overflow trigger and the near-overflow band) and writes one LP-labeled record per
resolvable state. This is the long step — hours, fully parallel:

```bash
python scripts/run_phase_g_corpus.py --out checkpoints/phase_g_targets.npz --workers 14
```

~7168 grid candidates → ~4800 emitted records (the rest are deep-equilibrium states that don't
resolve within the horizon; they're serialized to `*.rejected.npz` for later reanalysis). Inspect
the result with the corpus diagnostics:

```bash
python -c "from training.value_targets import load_targets_as_records, source_breakdown; \
print(source_breakdown(load_targets_as_records('checkpoints/phase_g_targets.npz')))"
```

## 3.2 The held-out calibration ruler

The ruler must be **disjoint** from the training grid on every axis (otherwise the gate measures
memorization, not generalization), and must include the **interior-valued** pins (values strictly
inside (−1, 1)) so it constrains the middle of the range, not just the ±1 boundary:

```bash
# Regenerate the interior-aware ruler (small grid, no OOM risk)
python scripts/run_ceiling_holdout.py --out checkpoints/ceiling_holdout_interior.npz --workers 8

# Graft the trusted baseline ruler + the 3 exact interior anchors → the clean comparison ruler
python scripts/make_clean_holdout.py        # writes checkpoints/ceiling_holdout_clean.npz
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

The legacy self-play value-net loop (the `hal/` tower, scripted-opponent self-play rather than the
exact oracle) still works for quick experiments:

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
all five charter success criteria met. (The bootstrap step has no RNG, so a rerun is bit-equivalent.)

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

`scripts/play_cli.py` plays against `CanonicalHal` (forward search; optionally backed by a trained
value net via `set_nn_evaluator`). `scripts/play_vs_cfr.py` plays against a precomputed
equilibrium-strategy table (the legacy bucketed CFR baseline).

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
