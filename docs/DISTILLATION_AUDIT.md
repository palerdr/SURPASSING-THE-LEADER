# Python Solver Distillation Audit

Status: first proof-safe distillation batch applied on 2026-07-11.

Baseline archive: commit `a4b03db` on branch `rust-solver`. That commit was
pushed before any deletions in this audit, so every removed historical file is
recoverable from Git.

## 1. Scope and the meaning of "full functionality"

This audit covers the Python arm only. Nothing under `crates/` is part of the
distillation, and the Rust kernel remains a later REGEN2RL handoff.

"Full functionality" means preserving all of these currently supported
capabilities:

| ID | Capability | Required evidence |
|---|---|---|
| F1 | Engine authority and exact action legality, including the asymmetric `61 x 60` leap window | Engine regression suite and solver-firewall tests |
| F2 | Exact finite-horizon stochastic zero-sum solve with LP as oracle | Rigorous exact, transition, orientation, cache, and chance tests |
| F3 | Selective search, bounded resolve, and simultaneous matrix-game MCTS | Selective, MCTS, subgame, and P3 conformance tests |
| F4 | Tactical anchors and certified Tier A generation/runtime lookup | Tablebase, backward-map, epoch, and Tier A tests |
| F5 | Reconstructable V2 replay, Generation-Zero targets, strict bundles, and grouped training | Replay, Gen-0, target, model, and training tests |
| F6 | Solver-backed play plus the temporary legacy playable fallback | Play-agent, bucket-parity, integration, and CLI parser checks |
| F7 | Calibration, ladder, exploitability, policy-drift, audit, and promotion evidence | Gate/report tests and command-contract checks |
| F8 | Hydra entry points for every advertised workflow | Config-to-parser contract test and `--cfg job` smokes |
| F9 | Explicit compatibility readers required by the P2 migration contract | Legacy-loader and strict-rejection tests |

The following are not functionality under that definition:

- a second, disconnected copy of the repository;
- a command whose Hydra configuration cannot reach its parser;
- an unreferenced future scaffold with no test or selected configuration;
- a dead wrapper around a class that no longer exists;
- generated checkpoints or corpora that the strict V2 loader intentionally
  rejects;
- preparatory Rust code before the Python baseline is frozen.

This distinction prevents both kinds of audit error: preserving dead code just
because it once ran, and deleting compatibility behavior that is still an
explicitly tested contract.

## 2. Evidence and reproducibility

The disposition below is based on five independent views.

1. **Graphify.** The graph contained 4,284 nodes and 11,428 links before the
   deletion. `legacy/old_layout` contributed 1,390 nodes (32.4%). A Graphify
   path query found no path from `legacy/old_layout` into `stl`; the duplicate
   labels instead made live paths ambiguous.
2. **Import and command reachability.** Repository-wide exact searches found
   zero imports from `stl`, `tests`, or `configs` into `legacy/old_layout`.
   Every Hydra-selected module and every direct `__main__` surface was
   inventoried, including dynamic `importlib` use in `stl.cli` and one test.
3. **Vulture 2.16.** On `stl`, Vulture reported 135 candidates at 60%, two at
   80/90%, and zero at 100%. The two 90% imports were genuine. Every 60%
   candidate was checked for framework protocols, dataclass serialization,
   tests, dynamic imports, or a public compatibility contract before a
   deletion decision.
4. **Duplicate-body analysis.** AST-normalized top-level definitions exposed
   exact live-to-live copies. File similarity and exact definition bodies also
   measured the old-layout mirror independently of names.
5. **Executable evidence.** The pre-audit baseline collected 707 tests and
   passed `703 passed, 4 skipped`. This audit adds a parameterized Hydra-to-
   argparse contract test. Each phase below has a frozen test gate.

Useful reproduction commands are:

```powershell
graphify query "smallest live Python solver dependency spine" --budget 5000
graphify path "legacy/old_layout" "stl" --max-depth 4
uvx vulture stl --min-confidence 80
rg -n "legacy[\\/]old_layout|legacy\.old_layout" stl tests configs
uv run pytest --collect-only -q
uv run pytest -q
```

Graph absence alone is not treated as proof of deadness. A candidate is removed
only when graph reachability, exact search, dynamic-entry inspection, and the
relevant tests agree.

## 3. Quantitative answer

### 3.1 Before this audit

| Surface | Python files | Physical Python LOC | Notes |
|---|---:|---:|---|
| Active `stl/` | 85 | 20,259 | Runtime, commands, and compatibility code |
| `legacy/old_layout/` | 118 | 20,952 | 132 tracked files including Markdown |
| Tests | 58 | 10,985 | 707 collected cases before this audit |
| Hydra | 23 YAML files | 184 YAML LOC | 16 command and 5 experiment profiles plus root/RL config |

The old tree was larger than the active implementation. It contained 36
same-named active/legacy pairs at least 90% similar, 418 exact top-level
definition-body groups shared with the live tree, and 7,103 active LOC whose
definitions had an exact historical twin.

### 3.2 Smallest current full-capability closure

Without reorganizing modules, the present import closure for F1-F9 is:

```text
59 of 85 active Python files
16,306 of 20,259 active Python LOC
```

That is the rigorous answer to "how small can it be today while retaining the
whole current product surface?" The five-file computational kernel is smaller,
but it does not retain replay, training, play, evaluation, promotion, or their
audit packs.

The irreducible computational spine is:

```text
stl/engine/actions.py
stl/engine/game.py
        |
        v
stl/solver/exact.py
        |------------------|
        v                  v
stl/solver/search.py   stl/solver/tablebase.py
        |
        v
matrix-game MCTS + exact/tablebase leaf anchors
```

The operational closure adds the five V2 learning kernels
`contracts.py`, `replay.py`, `model.py`, `targets.py`, and `train.py`, then the
Gen-0, play, evaluation, promotion, Hydra, and conformance surfaces.

### 3.3 After the first applied batch

The active tree is 81 Python files and 19,580 physical LOC. In addition to the
entire old-layout mirror, this batch removes four active dead files and three
unused dependency families. Relative to the two Python implementation trees at
the baseline, 21,631 LOC are gone, a 52.5% reduction, without removing F1-F9.

After `graphify update .`, the current code graph has 2,697 nodes, 7,569 edges,
and 154 communities. Exact searches of the refreshed graph find zero
`legacy/old_layout` nodes and zero nodes for the four removed active modules.

The first batch deliberately stops short of the 59-file closure. The remaining
gap contains compatibility surfaces and core-module refactors whose proof gates
must run after the active Generation-Zero process exits.

## 4. Applied batch D1: proof-safe removal

### 4.1 Historical mirror

`legacy/old_layout/` was deleted in full. The runtime-risk classification is
low because:

- no active import points to it;
- pytest only collects `tests/`;
- all active functionality already lives under `stl/`;
- commit `a4b03db` is a pushed, exact archival snapshot.

The forensic risk is addressed by retaining `legacy/README.md` with the archive
commit and old-to-new path map.

### 4.2 Unsupported legacy training commands

These files were deleted:

| Deleted file/config | Why it is not a current AlphaZero capability |
|---|---|
| `stl/commands/alphazero_loop.py` | Ran `CanonicalHal` against scripted opponents, stored outcome-only legacy rows, and never called matrix-game MCTS |
| `stl/commands/deep_training.py` | Unreferenced duplicate of the same legacy loop with no config, test, or documentation entry |
| `stl/commands/train_hal_value_net.py` | Unreferenced short wrapper over the same legacy data/trainer |
| `configs/command/self_play.yaml` | Its keys did not match the parser and the command was explicitly documented as unsupported |
| `configs/experiment/alphazero_smoke.yaml` | Only targeted the removed unsupported command and was packaged inertly |

The tested `stl.learning.self_play` compatibility module remains until P5
replaces it with genuine two-role MCTS self-play.

### 4.3 Dead PPO opponent island and dependencies

`model_opponent.py` and its unreachable factory chain were removed. No command,
test, or config called `build_opponent_league`, `create_model_opponent`,
`parse_weighted_model_spec`, `opponent_config_label`, or `SelectorOpponent`.
`ControllerSelector`, named by the wrapper, does not exist anywhere else in the
repository.

This made three direct dependencies unnecessary:

- `sb3-contrib`;
- `stable-baselines3`;
- `matplotlib`, which had no live import at all.

Regenerating `uv.lock` removed those packages plus eight transitive packages.
Gymnasium remains because `stl.play.env.STLEnv` and its protocol attributes are
live and tested.

### 4.4 Dead symbols and future scaffolds

The following had zero live, test, config, documentation, or dynamic-string
consumer and were removed:

- `compare_ladder._copy_counts`;
- `play.search._hal_immediate_view`;
- the unused `as_completed` import in `commands/tier_a.py`;
- the unused `ROUTING_FEATURE_SIZE` observation import;
- unselected observation-V3 constants, full-history builder, and V3 builder;
- the unused routing-vector wrapper around the still-live V2 history features.

The V2 observation behavior and dimensions are unchanged.

### 4.5 Command-contract repairs

| Config | Defect | Applied correction |
|---|---|---|
| `targets` | Emitted nonexistent `--tablebase-dir` and `--workers` | Renamed to `tier_a_dir`; removed `workers` |
| `promote` | Emitted nonexistent `--candidate` and `--champion` | Removed both; retained the three actual report inputs |
| `compare_*` | Omitted required candidate input and silently used a stale champion default | Declared candidate and champion checkpoints mandatory with `???` |
| `eval` | Pointed at a stale, rejected V1 checkpoint | Declared the checkpoint mandatory |
| `train_saved_corpus` | Omitted all three required paths | Declared all three mandatory |
| `tier_a_stage1` | Composed under `cfg.experiment`, so `limit` and `workers` never reached the command | Packaged globally and nested overrides under `command` |

`tests/test_cli_contract.py` now imports every selected command, captures its
argparse contract without executing the command, rejects unknown config keys,
requires every argparse-required option to be represented, and verifies the
Tier A experiment override reaches the selected command.

## 5. Exhaustive active-file disposition

The phase labels are:

- **Keep**: required by F1-F9.
- **D2**: remove or refactor after the live Gen-0 generation exits.
- **D3**: compatibility retirement only after a reviewed V2 checkpoint is
  independently playable and its replacement tests pass.
- **P5+**: keep until the corresponding REGEN2RL phase replaces it.
- **Moved audit**: behavior stays, but can leave the runtime namespace.

### 5.1 Package, engine, and solver

| File | Disposition | Exact reason/gate |
|---|---|---|
| `stl/__init__.py` | Keep | Package marker |
| `stl/cli.py` | Keep, D2 refactor | Sole documented dispatcher; replace `sys.argv` adaptation only after direct `run(config)` entry points exist |
| `stl/engine/__init__.py` | Keep | Engine export surface |
| `stl/engine/actions.py` | Keep | Literal-second legality authority |
| `stl/engine/game.py` | Keep | State transition and chance authority |
| `stl/solver/__init__.py` | Keep | Package marker |
| `stl/solver/exact.py` | Keep, D2 deduplicate | LP oracle and exact stochastic recursion; duplicate diagnostics can be canonicalized after Gen-0 |
| `stl/solver/search.py` | Keep, D2 deduplicate | Selective solve, resolve, MCTS, and evaluator protocol |
| `stl/solver/tablebase.py` | Keep, later split | Tactical registry, backward map, epoch builder, and runtime lookup are all live; split only with full Tier A parity |
| `stl/solver/conformance.py` | Moved audit | P1 report logic is live in tests but need not ship in a future runtime-only wheel |
| `stl/solver/mcts_conformance.py` | Moved audit | P3 report logic is live in tests and promotion evidence |

Public engine conveniences such as `Game.play_round`, `Game.get_state_summary`,
`can_use_leap_second`, and legacy `solve_half_round` are not removed in D1.
They have no production inbound edge, but are public or directly tested, so
their removal is a compatibility decision rather than dead-code cleanup.

### 5.2 Learning kernels and support

| File | Disposition | Exact reason/gate |
|---|---|---|
| `learning/__init__.py` | D2 refactor | Mutates `__path__` so one physical support module can acquire two import identities |
| `learning/contracts.py` | Keep | P0 objective, role, horizon, and config contract |
| `learning/replay.py` | Keep | Reconstructable, hash-verified V2 schema |
| `learning/model.py` | Keep | Active V2 features and value/two-policy network |
| `learning/targets.py` | Keep, later isolate V1 | Exact/Tier-A target generation plus required explicit compatibility readers |
| `learning/train.py` | Keep, later isolate V1 | Grouped V2 trainer and strict checkpoint bundle |
| `learning/gates.py` | D2 remove or explicit-export | Unused wildcard facade; direct modules own the behavior |
| `support/audit.py` | Keep | Audit packs and gate evidence |
| `support/awareness.py` | D3 | Used by the Gym/legacy teacher surface |
| `support/bootstrap.py` | P6 | Reanalysis/expert-iteration support; command wrapper is historical but APIs are tested |
| `support/calibration.py` | Keep | Promotion calibration report |
| `support/corpus_diagnostics.py` | Keep, D2 schema fix | Tested audit API; hard-coded V1 index must import a named schema map |
| `support/curriculum.py` | D3 | Current Gym/teacher scenarios are tested; dead sampler factory can go in D2 |
| `support/evaluate.py` | D3 | Legacy `CanonicalHal` neural evaluator; two unused accessors can go in D2 |
| `support/legacy_train.py` | D3 | Explicit legacy playable/checkpoint path, never the V2 default |
| `support/observation.py` | D3 | V1/V2 Gym observations remain tested; unused V3 scaffold removed in D1 |
| `support/reanalysis.py` | P6 | Planned and tested exact-tier reanalysis API |
| `support/reward.py` | D3 | Gym-only shaping; forbidden from the rigorous solver |
| `support/route_math.py` | Keep | Active V2 feature and playable calculations |
| `support/route_stages.py` | D3 | Curriculum/demo surface; unused duplicate bonus map can go in D2 |
| `support/routing_features.py` | D3 | V2 legacy observation history; future/dead wrappers removed in D1 |
| `support/scenarios.py` | D3 | Gym curriculum surface |
| `support/self_play.py` | P5 replacement | Tested legacy actor, explicitly not AlphaZero self-play |
| `support/strategy_features.py` | D3 | Teacher behavior; unread snapshot fields can go in D2 |
| `support/strength/__init__.py` | Keep | Explicit promotion-facing exports |
| `support/strength/best_response.py` | Keep | Exploitability interval evidence |
| `support/strength/match_gate.py` | Keep | Deterministic ladder gate |
| `support/strength/policies.py` | Keep | Frozen agent-policy adapters |
| `support/strength/sprt.py` | Keep | Sequential match decision state |
| `support/teacher_demos.py` | D3 | Tested demonstration compatibility surface |
| `support/tournament.py` | Keep | Ladder and strength execution |

### 5.3 Commands

| File | Disposition | Exact reason/gate |
|---|---|---|
| `commands/gen0_corpus.py` | Keep | P4 reconstructable anchor entry point; active generation uses it |
| `commands/tier_a.py` | Keep | Certified Tier A build entry point |
| `commands/strength_gate.py` | Keep | Advertised evaluation command |
| `commands/promote.py` | Keep | Promotion report consumer |
| `commands/compare_ladder.py` | Keep, D2 share plumbing | Promotion ladder evidence |
| `commands/compare_exploitability.py` | Keep | Promotion exploitability evidence |
| `commands/compare_policy_drift.py` | Keep | Promotion policy-readability evidence |
| `commands/exploitability.py` | Keep as library/CLI | Shared best-response helpers and direct tests |
| `commands/compare_tier_a_runtime.py` | Moved audit | Tier A audit helper and direct tests; no Hydra config needed in a runtime package |
| `commands/tier_a_decision.py` | D2 assess | Historical decision runner, but helpers feed `tier_a_targets` |
| `commands/tier_a_targets.py` | Keep for now | Tier A-to-training bridge; config repaired in D1 |
| `commands/tier_a_frontier.py` | D2 split | Analysis helpers are live, but generated continuation commands point to deleted `scripts/` paths |
| `commands/tier_a_training_pilot.py` | D2 retire | No config, docs, test, or inbound caller; legacy-target pilot |
| `commands/train_generation.py` | D2 retire | Pre-V2 expert-iteration workflow with explicit legacy-target opt-in |
| `commands/train_saved_corpus.py` | D2 assess/retire | Legacy-target gate; retain its useful assertions only after migrating them to V2 tests |
| `commands/next_run.py` | D2 retire | Planner emits nonexistent `python scripts/...` commands |
| `commands/validate_against_manga.py` | Move to regression tooling | Dynamically imported by a regression test; not a user command |

`commands/alphazero_loop.py`, `deep_training.py`, and
`train_hal_value_net.py` were removed in D1 and therefore do not appear in the
remaining active table.

### 5.4 Play and opponents

| File/group | Disposition | Exact reason/gate |
|---|---|---|
| `play/agent.py`, `play/cli.py` | Keep | Solver-backed playable surface |
| `play/opponents/base.py` | Keep | Minimal opponent protocol |
| `play/opponents/factory.py` | Keep, now scripted-only | Ladder/play construction; dead PPO branch removed |
| `random_bot.py`, `safe_bot.py`, `league.py` | Keep | Small deterministic/random baselines and weighted scripted leagues |
| `baku_teachers.py`, `hal_teachers.py`, policy helpers, `pattern_reader.py`, `teacher_helpers.py` | D3 | Current ladder, curricula, demos, and tests use these; remove only with replacement rulers |
| `play/action_model.py`, `canonical_hal.py`, `search.py`, `state.py` | D3 | Temporary legacy playable fallback and parity suite |
| `play/env.py` | D3 | Tested Gym environment; separate from the exact solver and AlphaZero actor |
| `play/opponents/model_opponent.py` | Removed D1 | Unreachable PPO island |

The D3 compatibility block is approximately 757 implementation LOC before its
support modules and 849 LOC of dedicated legacy tests. It must move as a unit;
silently deleting the tests first is prohibited.

## 6. Symbol-level D2 queue

These are independently proven dead or duplicated, but core-file edits are
held until the running Gen-0 process exits.

### 6.1 Exact duplicates

| Canonical target | Duplicate to remove/update | Required focused gate |
|---|---|---|
| One `ExactStrategyDiagnostics` and `diagnose_exact_strategy` implementation | Identical blocks in `solver/exact.py` and `solver/search.py`; update reanalysis/tests to the chosen owner | rigorous exact, reanalysis, conformance |
| One payoff-required helper | `_require_payoff` duplicated with the diagnostics | same as above |
| Shared utility-breakdown math, if extraction reduces rather than adds coupling | `_terminal_breakdown` and `_weighted_breakdown` duplicated in exact/search | exact and selective recursion parity |
| Engine terminal utility | `_hal_terminal_value` duplicated in play search and learning evaluator | play search parity and evaluator tests |
| Shared ladder count summaries | compare-ladder and Tier-A-runtime helpers | both comparison test modules |

The diagnostics owner must be chosen explicitly. The current caller owner is
`search.py`; the conceptual owner is `exact.py`. D2 will move callers to
`exact.py`, then delete the search copy, rather than leave two dataclass types
with the same fields and different identities.

### 6.2 Proven zero-inbound candidates

| Area | Symbols | Decision |
|---|---|---|
| Search | `_prior_from_evaluator` | Remove after Gen-0; superseded by `_prior_and_value_from_evaluator` |
| Exact | `terminal_breakdown`, `solve_cache_maxsize`, `build_augmented_payoff_matrix` | Remove only after focused solver tests; public names make this a D2 compatibility cut |
| Tablebase | `EpochSpec.deaths_total_hint` | Remove; aliases `cprs` and has no reader |
| Curriculum | `make_curriculum_sampler`, `stage_is_eligible_from_start` | Remove; no caller or dynamic selection |
| Evaluation | `get_nn_device`, `evaluate_batch` | Remove; no caller |
| Route shaping | `ROUTE_STAGE_BONUSES` | Remove; duplicates the live `light` reward preset |
| Play | `get_buckets`, `medium_check`, `hal_can_check_leap` | Remove with focused legacy parity tests |
| Tournament | unused `game_idx` binding | Replace with `_` |

### 6.3 Vulture candidates explicitly retained

Vulture's 60% findings include framework and structured-data uses it cannot
see. The following categories are not dead:

- PyTorch `ValueNet.forward`, invoked through `nn.Module.__call__`;
- Gym `metadata`, `action_space`, and `observation_space` protocol attributes;
- dataclass/report fields consumed through `asdict`, JSON serialization, or
  artifact readers;
- Hydra-selected `module.main` functions imported by string;
- explicit P2 compatibility APIs such as `load_legacy_checkpoint`,
  `load_targets`, and `load_rejected_pool`;
- P6 reanalysis and P7/P8 gate APIs that are directly tested even before the
  long training phases execute.

No deletion is justified merely because Vulture assigns 60% confidence.

## 7. Hydra and entry-point audit

After D1 there are 15 command configs. Their target state is:

| Config | Status after D1 | Long/side-effect boundary |
|---|---|---|
| `tier_a` | Valid | Long tablebase generation unless limited |
| `gen0_targets` | Valid | First long P4 generation; currently active |
| `train_gen0` | Valid | Long training; must wait for reviewed manifests |
| `play` | Parser-valid through explicit `command.args` | Interactive; solver requires a V2 checkpoint |
| `eval` | Parser-valid; checkpoint mandatory | Runs games/search |
| `compare_ladder` | Parser-valid; both checkpoints mandatory | Runs paired games |
| `compare_exploitability` | Parser-valid; both checkpoints mandatory | Potentially expensive exact/search audit |
| `compare_policy_drift` | Parser-valid; both checkpoints mandatory | Search audit |
| `promote` | Parser-valid; three reports mandatory | Writes only a decision report |
| `targets` | Repaired | Tier A target generation |
| `tier_a_decision` | Parser-valid, historical | Reads historical artifacts and runs analysis |
| `tier_a_frontier` | Parser-valid, stale continuation text | D2 split/retire |
| `next_run` | Parser-valid, stale continuation text | D2 retire |
| `train_saved_corpus` | Parser-valid; three paths mandatory | D2 migration decision before training |
| `train` | Parser-valid, historical V1 workflow | D2 retire after moving useful gates |

There is no installed console script and `[tool.uv] package = false`; the public
entry point is source-tree `python -m stl.cli`. There are still direct module
guards for research helpers, but they are not separate supported products.

The versioned `cfg.rl` object is validated and hashed but not forwarded into
the selected command. It is therefore contract metadata, not live runtime
wiring. D2 must either pass it into typed `run(config)` functions or rename it
to make that limitation impossible to miss.

## 8. Tracked artifacts and non-runtime material

Four ignored-by-policy checkpoint artifacts are nevertheless tracked, totaling
6,108,732 bytes:

- `checkpoints/ceiling_corpus.npz`;
- `checkpoints/ceiling_holdout_clean.npz`;
- `checkpoints/gen_ceiling_tbw15_wd1e-4/best.pt`;
- its `log.json`.

The NPZ files use the old 23-feature/61-action schema, and the checkpoint is
intentionally rejected by the strict V2 loader. They remain only until D2
historical runners stop naming them and synthetic/V2 fixtures replace the
loader regression. They must never be confused with the active untracked
`outputs/regen2rl` generation.

Reference-paper PDFs and the rendered whitepaper are documentation
deliverables, not runtime bloat, and are outside this code-distillation metric.

## 9. Phase gates

### D1: proof-safe batch (this audit)

- [x] Push an archival baseline before deletion.
- [x] Verify zero old-layout runtime/config/test imports.
- [x] Delete `legacy/old_layout` only inside the resolved workspace path.
- [x] Remove unsupported legacy command leaves and dead PPO island.
- [x] Remove the three unused direct dependencies and regenerate `uv.lock`.
- [x] Repair known Hydra/parser mismatches.
- [x] Add the all-command parser contract test.
- [x] Run Vulture at 80%; no finding remains.
- [x] Collect the full suite: 723 cases, the baseline 707 plus 16 command-contract cases.
- [x] Exercise engine, solver, learning, play, and command tests in the full suite.
- [x] Run the full suite without interrupting Gen-0: `719 passed, 4 skipped`.
- [x] Run `graphify update .` and verify that historical/removed nodes are gone.
- [x] Commit D1 without staging any pre-existing or concurrent `crates/` changes.
- [x] Push D1 to `origin/rust-solver`.

### D2: post-generation active-tree normalization

- [ ] Confirm the Gen-0 process has exited and both shard manifests are durable.
- [ ] Canonicalize exact-strategy diagnostics into one module/type.
- [ ] Remove `_prior_from_evaluator` and the proven zero-inbound symbols.
- [ ] Replace the `learning.__path__` mutation with explicit support imports.
- [ ] Remove `learning/gates.py` or give it explicit tested exports.
- [ ] Remove manual source-tree `sys.path` injections from packaged modules.
- [ ] Retire stale planners/runners only after moving any useful report logic.
- [ ] Replace tracked V1 artifact dependencies with synthetic or V2 fixtures.
- [ ] Re-run every D1 gate plus Gen-0 target/train smokes.

### D3: reviewed-V2 compatibility retirement

- [ ] Promote and document a reviewed V2 checkpoint.
- [ ] Prove `SolverAgent` can play and evaluate without the bucket fallback.
- [ ] Replace legacy ladder/teacher rulers where promotion policy requires it.
- [ ] Move or delete the Gym/bucket/legacy-training block and its tests together.
- [ ] Remove Gymnasium only if no retained environment consumer remains.
- [ ] Recompute the import closure and dependency lock.

### P5+: genuine AlphaZero self-play

- [ ] Replace outcome-only `stl.learning.self_play` with one joint MCTS actor.
- [ ] Store both improved role policies and terminal/capped Hal-perspective
  outcomes in V2 replay.
- [ ] Make the new command pass the same Hydra/parser contract test before any
  long self-play run.
- [ ] Keep the Python baseline frozen until P0-P8 gates pass; only then define
  the narrow Rust kernel boundary.

## 10. Stop conditions

Distillation stops immediately if any batch:

- changes legal action sets, transition probabilities, matrix orientation, or
  terminal utility;
- changes V2 feature/action/replay hashes without a schema version bump;
- weakens a firewall, calibration, exploitability, or promotion threshold;
- deletes a test before the replacement capability test exists;
- touches `crates/` before the Python baseline is accepted;
- requires interrupting or mutating the active long generation.

The target is not the fewest lines at any cost. It is the smallest tree for
which every retained claim has a named implementation owner and an executable
acceptance gate.
