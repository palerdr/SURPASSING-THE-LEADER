# `tests/`

Regression tests for the engine, route math, solver, AI wrappers, and training
pipeline.

Useful focused checks:

```bash
python3 -m pytest tests/test_clock.py tests/test_death_mechanics.py
python3 -m pytest tests/test_backward_map.py tests/test_tablebase_epoch.py
python3 -m pytest tests/test_cfr_firewall.py
```

`python3 -m pytest --collect-only -q` is a cheap sanity check after structural
changes.
