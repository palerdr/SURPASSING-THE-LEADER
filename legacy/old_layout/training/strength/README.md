# `training/strength/`

Strength and promotion checks for agents.

Files:

- `match_gate.py` - ladder runs and report formatting.
- `best_response.py` - exploitability-style interval checks against fixed
  policies.
- `policies.py` - policy adapters used by the gate.
- `sprt.py` - sequential probability ratio test helpers.

This is evaluation infrastructure. It should not define engine rules or exact
solver semantics.
