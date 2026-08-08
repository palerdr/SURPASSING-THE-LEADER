# STL Project Instructions

This subtree is intentionally a clean formulation shell. It contains no
current STL solver, learned model, replay format, opponent model, training
pipeline, or gameplay agent. Git history is the archive for the removed
experiments; do not restore them under `legacy`, `old`, or version-suffixed
names.

## Retained surfaces

- `cli.py` and `config/` are a neutral Hydra experiment harness. The default
  configuration performs no command and encodes no solver or learning choice.
- `engine/` is retained only as the compatibility interface currently consumed
  by `src/arena/`. It is a behavioral reference for already-frozen rules, not
  the canonical full STL formulation and not a foundation to extend casually.
- `tests/engine/` protects that compatibility interface. `test_cli_contract.py`
  protects the generic Hydra dispatcher.
- Repository-wide canonical rules and formulation contracts belong in root
  `docs/`. Generated experiment data remains gitignored and STL-owned.

## Rebuild rule

Freeze the full game's state, observations, actions, transition/chance model,
and utility in the canonical documentation before adding implementation code.
Then add the smallest implementation that realizes that contract. Do not add
solver, learning, or play modules merely to preserve an earlier experiment.

Run `uv run python -m pytest src/stl/tests -q` for the retained surface.
