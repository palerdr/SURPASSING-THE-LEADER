# Hal Lab Instructions

`src/hal_lab/` owns Hal research: the code that trains and evaluates Hal
policies, and the frozen evidence of each finished study. No project imports
it.

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
- Put new research code under `src/hal_lab/`. `src/arena/policies/` keeps
  the Hal providers that the terminal and the browser play, and the provider
  code that a study or a trainer shares with play.

## Layout

- `cli.py` holds the match command. `provenance.py` holds the evidence
  verifier, and `evidence/` holds its path map and its baseline.
- `harness/` holds the code that more than one study or trainer uses:
  `series.py` plays the paired-seat series and holds the SPRT, and
  `opponent_league.py` holds the deterministic public-history opponent
  families. Put new shared research code here.
- `training/` holds the trainers: `train_exploit_hal.py` with its gym
  `exploit_hal_gym.py`, and `train_aggro_hal.py` with its environment
  `aggro_env.py` and its curriculum `aggro_memory_curriculum.py`.
- `experiments/<study>/` holds one finished study: its runner, its study
  model, its configs, and its frozen records. The studies are
  `perfect_hal_bayes_v2`, `perfect_hal_ensemble_v1`, `translated_hal_v1`,
  `external_hal_prior_v1`, `neural_pilots_v1`, `selector_study_v1`,
  `pm_hal_v3`, and `aggro_hal_v1`. `exploit_hal_v2/config/` holds the three
  historical Exploit Hal v1 configs.
- A study can import the frozen module of an earlier study.
  `translated_hal_v1` imports `perfect_hal_bayes_v2` and
  `perfect_hal_ensemble_v1`, and `selector_study_v1` imports
  `neural_pilots_v1`. Do not edit a frozen runner to share its code; put the
  shared code in `harness/`.
- A runner that hashes its input modules finds an `arena.policies` provider,
  or a module in another hal_lab package, through that module's `__file__`.
  The runner uses `Path(__file__).with_name(...)` only for a sibling file in
  its own study folder. If you move a sibling out of the study folder, change
  its lookup to the moved module's `__file__`.

## Match command

`uv run python -m hal_lab match` plays a paired-seat agent-versus-agent series
and writes a JSON report:

```powershell
uv run python -m hal_lab match --candidate perfect-hal --opponent dth --pure-dth --games 50 --output src/hal_lab/outputs/perfect-hal-v1/vs-exact.json
```

- `cli.py` owns the match choices: every play agent and the research-only
  `aggro-hal`, with its three `--aggro-hal-*` flags. `cli.py` builds Aggro Hal
  through the public `arena.policies.aggro_hal.make_live_provider`, and
  `arena.policies.registry` builds every play agent. The registry's one
  pure-DTH gate also covers `aggro-hal`.
- `harness/series.py` plays each base seed in both seatings and stops when the
  predeclared SPRT decides. `arena.match.play_match_game` plays each game.
  `series.py` holds the SPRT constants and `sprt_verdict`, and
  `training/train_exploit_hal.py` reads the verdict from there.
- The report keeps the `arena-match-report-v1` schema of the earlier
  `python -m arena match` command.

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

## Records and configs in `src/arena/config`

hal_lab owns six records that stay in `src/arena/config/`, because
`tests/test_evaluate_pm_hal.py` and `tests/test_aggro_adaptive_config.py` read
them at the paths that the records hold:

- `pm_hal_confirmation_v3.json`
- `pm_hal_evaluation_v3.json`
- `aggro_hal_tactical_baseline_v1.json`
- `aggro_hal_adaptive_exploitation_goal_v1.json`
- `aggro_hal_adaptive_memory_v1.yaml`
- `aggro_hal_v1.yaml`

Keep each one at its path with its bytes unchanged.

Five configs stay there too. The Hal providers read
`translated_hal_v1_selection.json`, which is also a frozen record,
`pm_hal_controller_v3.json`, and `exploit_hal_v2.yaml`. The Exploit Hal v2
smoke configs `exploit_hal_smoke_v2.yaml` and
`exploit_hal_outcome_only_smoke_v2.yaml` stay beside `exploit_hal_v2.yaml`.
They share its v2 schema, which the live loader accepts, and
`tests/test_train_exploit_hal_contract.py` reads the three files from one
directory.

## The collector label

`training/train_aggro_hal.py:309` writes the string
`arena.policies.opponent_league.make_opponent` into the default session
collector binding. The string is an identity label; no code imports it. An
Aggro resume compares the binding in a checkpoint with that default, so a
changed label makes each existing Aggro checkpoint refuse to resume.
`opponent_league.py` now lives in `harness/`, and the label keeps its earlier
module path. Keep the string verbatim.

## Exploit Hal archives

`training/train_exploit_hal.py` lived at `arena.policies.train_exploit_hal`
before it moved here. Each `maskable-ppo.zip` archive pickles the class of the
feature extractor. `load_maskable_ppo` maps the earlier module name to the
trainer before it loads an archive, so an archive that pickled the class by
that name still loads. The archives under `outputs/exploit-hal-v1/` came from
`python -m` runs, which pickle the class by value, and load without the map.

## Outputs

- hal_lab owns the ignored root `outputs/` store, where the Hal research runs
  wrote their reports and checkpoints. The frozen records key its files as
  `outputs/...`, so the store stays at the root. Leave its files in place and
  unchanged.
- Write the outputs of a new run under `src/hal_lab/outputs/<study>/`, which
  git ignores.

## Working in this subtree

Run `uv run python -m pytest src/hal_lab/tests -q` and
`uv run python -m hal_lab.provenance --check`. `tests/test_match.py` covers the
match command, the series harness, and the SPRT. The other tests cover the
trainers, the opponent league, and the study runners and configs. The
provenance checks that read the tag skip in a checkout without it, such as the
CI clone.

The trainers and the runners are modules. Run each one with
`uv run python -m hal_lab.<package>.<module>`, for example
`uv run python -m hal_lab.training.train_aggro_hal --help`.
