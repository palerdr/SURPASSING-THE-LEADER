# `stl.commands/`

Command implementations invoked by the Hydra dispatcher in `stl.cli`. These replace the old flat `scripts/` directory while keeping reusable command logic importable for tests.

`gen0_corpus.py` is the P4 Generation-Zero anchor entry point. Its smoke config
writes coordinated reconstructable training and external-ruler V3 shards. The
command resumes immutable component chunks and derives exact/terminal rows from
stored legal engine trajectories:

```bash
uv run python -m stl.cli command=gen0_targets experiment=gen0_smoke
uv run python -m stl.cli command=train_gen0 experiment=gen0_train_smoke
```

`command=gen0_targets` without the smoke experiment is the first long
generation run in `docs/REGEN2RL.md`; `command=train_gen0` is intentionally
sequenced after manifest review.
