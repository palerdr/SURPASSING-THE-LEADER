# `hal/`

Playable Hal adapters and the small learned value model.

Files:

- `agent.py` - solver-backed Hal/Baku agent adapters used by tests and
  `play_cli.py`.
- `hal_opponent.py` - `CanonicalHal`, the older playable scripted/search Hal.
- `search.py` - legacy bucketed forward search for `CanonicalHal`.
- `state.py` - memory/belief state helpers for the legacy Hal path.
- `action_model.py` - bucket definitions, bucket payoff averaging, and
  best-response helpers.
- `value_net.py` - compact value/policy network and feature extraction.
- `train.py`, `evaluate.py`, `self_play.py` - legacy value-net training and
  evaluation helpers.

Treat this directory as the playable/adaptor layer. Rigorous exact-second
solving lives in `environment/cfr/`; certified epoch tablebases live in
`training/tablebase/`.
