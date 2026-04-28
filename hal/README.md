# Hal Package Layout

`hal/` now has one promoted playable opponent and two clear support areas.

## Stable Playable Opponent

- `hal_opponent.py`: `CanonicalHal`, the current playable Hal wrapper.
- `search.py`: legacy bucketed forward search used by `CanonicalHal`.

## Consolidated Support Modules

- `state.py`: Hal state, memory, belief tracking, and related dataclasses.
- `action_model.py`: legacy bucket definitions, bucket payoff averaging, bucket
  legality, bucket resolution, and best-response helpers.

The old files `types.py`, `belief.py`, `memory.py`, `buckets.py`, and
`solver.py` are compatibility shims. New code should import from `state.py`,
`action_model.py`, or `environment.cfr.minimax`.

## Experimental Value Stack

- `value_net.py`: neural value model and feature extraction.
- `evaluate.py`: legacy evaluator bridge and handcrafted fallback.
- `self_play.py`: dataset generation.
- `train.py`: value-network training/checkpoint helpers.

These are not part of the rigorous exact CFR foundation. Rigorous exact-second
solver code lives under `environment/cfr/`.

