# Hal Lab Instructions

`src/hal_lab/` owns Hal research: the code that trains and evaluates Hal
policies, and the frozen evidence of each finished study. No project imports
it. Today the package holds the evidence verifier and its path map. The
training and study modules stay in `src/arena/policies/` until they move here,
and their tests stay with arena until then.

[`docs/HAL_RESEARCH.md`](docs/HAL_RESEARCH.md) holds the research narrative,
with the commands and results of each study.
[`docs/exploitv2.md`](docs/exploitv2.md) holds the Exploit Hal curriculum
plan.

## Boundaries

- [`docs/PROJECTS.toml`](../../docs/PROJECTS.toml) lists the modules that
  hal_lab may import: `arena`, `stl.engine`, `stl.solver.canonical`,
  `stl.reader`, `dth.agent`, and `dth.solver`.
  `tests/meta/test_layer_boundaries.py` enforces the list, and it fails any
  other project that imports hal_lab.
- hal_lab may import `torch`, `gymnasium`, `stable_baselines3`, and
  `sb3_contrib`. The torch firewall of the other projects stops at this
  directory.
- Put new research code under `src/hal_lab/`. `src/arena/` keeps the Hal
  providers that the terminal and the browser play.

## Frozen evidence

- A frozen record holds the protocol or the results of a finished Hal
  experiment, or a sealed record binds it. `RECORDS` in `provenance.py` lists
  each record. Do not edit, reformat, or re-key a frozen record, and do not
  bind it to new bytes. A new experiment writes a new versioned record.
- The records bind source files, configs, and generated outputs by SHA-256.
  The one-shot protocols are historical: translated_hal_v1, neural_pilots_v1,
  external_hal_prior_v1, selector_study_v1, the perfect_hal_bayes_v2 test
  split, and pm_hal v3. Their bound sources can change bytes or paths, and
  their runners can refuse a re-run. `hal_lab.provenance` verifies them
  against git history.
- `uv run python -m hal_lab.provenance --check` recomputes every binding and
  compares the result with `evidence/verification_baseline.json`. It reads a
  file that git tracked at the `pre-restructure` tag from the tag, and an
  ignored output from disk. It writes no file.
- A change that moves a record, a bound source, or a bound output adds the
  old and the new path to `evidence/PATH_MAP.toml`. The keys in the records
  stay as written.
- Keep `--check` passing after every change. Before you record a new baseline with
  `uv run python -m hal_lab.provenance --report > src/hal_lab/evidence/verification_baseline.json`,
  review each difference that `--check` prints, and give the reason in the
  commit message.
- The baseline holds three mismatches. Later builds replaced the DTH manifest
  at `src/dth/artifacts/complete_full_v1/tablebase.json` that two records
  bind. The neural pilots record binds the first `run_neural_pilots.py`, and
  `outputs/neural-pilots-v1/run_neural_pilots-original.py` keeps those bytes.
  The five bindings under `outputs/aggro-hal-v1/` are missing on this disk.

## Sealed records in `src/arena/config`

hal_lab owns six records that stay in `src/arena/config/`, because
`test_evaluate_pm_hal.py` and `test_aggro_adaptive_config.py` read them at
the paths that the records hold:

- `pm_hal_confirmation_v3.json`
- `pm_hal_evaluation_v3.json`
- `aggro_hal_tactical_baseline_v1.json`
- `aggro_hal_adaptive_exploitation_goal_v1.json`
- `aggro_hal_adaptive_memory_v1.yaml`
- `aggro_hal_v1.yaml`

Keep each one at its path with its bytes unchanged.

## The collector label

`src/arena/policies/train_aggro_hal.py:309` writes the string
`arena.policies.opponent_league.make_opponent` into the default session
collector binding. The string is an identity label; no code imports it. An
Aggro resume compares the binding in a checkpoint with that default, so a
changed label makes each existing Aggro checkpoint refuse to resume. Keep the
string verbatim when `opponent_league.py` moves.

## Outputs

- hal_lab owns the ignored root `outputs/` store, where the Hal research runs
  wrote their reports and checkpoints. The frozen records key its files as
  `outputs/...`, so the store stays at the root. Leave its files in place and
  unchanged.
- Write the outputs of a new run under `src/hal_lab/outputs/<study>/`, which
  git ignores.

## Working in this subtree

Run `uv run python -m pytest src/hal_lab/tests -q` and
`uv run python -m hal_lab.provenance --check`. The provenance checks that
read the tag skip in a checkout without it, such as the CI clone.
