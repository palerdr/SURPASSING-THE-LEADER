# `tests/`

Regression tests for the engine, route math, solver, AI wrappers, and training
pipeline. The firewall tests are non-negotiable for solver-core changes.

Useful focused checks:

```bash
uv run pytest tests/test_clock.py tests/test_death_mechanics.py
uv run pytest tests/test_backward_map.py tests/test_tablebase_epoch.py
uv run pytest tests/test_cfr_firewall.py
```

`uv run pytest --collect-only -q` is a cheap sanity check after structural
changes.
