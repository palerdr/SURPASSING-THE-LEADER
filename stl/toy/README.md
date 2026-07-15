# ToySTL-v0

This package is an isolated exact-to-MCTS-to-self-play foundation. The active
`bucket12_fixed50` ruleset has twelve actions (`1..12` = 5..60 seconds),
five-second load units, a 60-unit load cap, fixed independent 50/50 revival,
simultaneous perfect information, and an eight-half-round default horizon.
Its serialized state is exactly `(hal_load, baku_load, role_phase)` plus the
explicit remaining horizon passed to search and encoding. There is no clock or
leap-second action in v0.

The later factories in `rules.py` are intentionally separate versioned
rulesets and are not mixed with v0 artifacts.

Typical commands from the repository root:

```text
uv run python -m stl.toy exact
uv run python -m stl.toy targets
uv run python -m stl.toy train --targets outputs/toy/bucket12_fixed50/targets.npz --target-manifest outputs/toy/bucket12_fixed50/targets.json
uv run python -m stl.toy mcts-audit
uv run python -m stl.toy self-play --checkpoint outputs/toy/bucket12_fixed50/checkpoint/best.pt
```
