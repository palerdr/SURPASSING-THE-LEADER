# `stl.commands/`

Command implementations invoked by the Hydra dispatcher in `stl.cli`. These replace the old flat `scripts/` directory while keeping reusable command logic importable for tests.

`gen0_corpus.py` is the P4 Generation-Zero anchor entry point. The scaled path
writes separate training, development, and external-ruler V3 shards plus exact
root-matrix certificate sidecars. It resumes hash-committed chunks and derives
exact/terminal rows from stored legal engine trajectories:

```bash
uv run python -m stl.cli command=gen0_targets experiment=gen0_scaled_pilot
uv run python -m stl.cli command=train_gen0 experiment=gen0_scaled_pilot_train
uv run python -m stl.cli command=gen0_targets experiment=gen0_scaled_full --cfg job
```

After full generation, train the frozen 3-by-3 width/seed ladder. Exact
per-source batch quotas and the dedicated development shard are declared in
`configs/experiment/gen0_scaled_train.yaml`. Use `command=gen0_select` on all
nine `best.pt` files; it cannot load the external ruler. Only then run the
selected checkpoint once through the ruler and MCTS audit:

```bash
uv run python -m stl.cli -m command=train_gen0 experiment=gen0_scaled_train \
  command.hidden_dim=64,128,192 command.seed=0,4,8
uv run python -m stl.cli command=gen0_select \
  'command.checkpoint=[/path/to/candidate-a.pt,/path/to/candidate-b.pt]'
uv run python -m stl.cli command=gen0_eval \
  command.checkpoint=/path/to/selected.pt \
  command.train=outputs/regen2rl/gen0_scaled_train_v3.npz \
  command.ruler=outputs/regen2rl/gen0_scaled_external_ruler_v3.npz
uv run python -m stl.cli command=gen0_mcts_audit \
  command.checkpoint=/path/to/selected.pt
```

The exact source counts, mathematical anchor derivations, gates, and current
stop point are in the scaled P4 checklist in `docs/REGEN2RL.md`.

The V5 bounded-frontier cycle is separate from the frozen P3 horizon-one pack.
Inspect every long command before running it:

```bash
uv run python -m stl.cli command=gen0_bellman_cycle experiment=gen0_v5_bellman_smoke --cfg job
uv run python -m stl.cli command=gen0_targets experiment=gen0_tablebase_v5 --cfg job
uv run python -m stl.cli command=gen0_bellman_cycle experiment=gen0_v5_bellman_full --cfg job
uv run python -m stl.cli command=train_gen0 experiment=gen0_v5_horizon_train --cfg job
uv run python -m stl.cli command=train_gen0 experiment=gen0_v5_policy_repair --cfg job
uv run python -m stl.cli command=gen0_bellman_select --cfg job
uv run python -m stl.cli command=gen0_bounded_audit --cfg job
```

Bellman generation uses `command.work_dir` for one hash-committed root at a
time. A matching rerun strict-loads those commits and the immutable plan instead
of solving again. Exact replay, certificate, and closure artifacts are published
independently of the candidate-action report; a failed report prevents holdout
sealing and downstream training but does not erase completed exact work.
`command.preflight_only=true` enumerates exact successor identities without
solving labels and proves all three complete closures are disjoint first.
`command.reuse_work_dir` lazily imports matching committed roots under a new
closure-safe split, so matrices are decompressed one at a time instead of
materializing the entire cache in memory.

`merge_gen0_evidence` publishes the single training and development shards used
by the trainer. Training combines V4 training with Bellman training.
Development combines V4 development, the consumed V4 holdout, and Bellman
development. The fresh V5 calibration and Bellman holdouts are never merge
inputs. The final bounded audit claims both concealed artifacts once, runs
static gates first, and starts the eight-root V3/V2 MCTS audit only after static
success.
