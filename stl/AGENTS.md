# Full STL Project Instructions

This subtree owns the full leap-aware Surpassing the Leader implementation.

- `engine/` is rule authority and must not import solver, learning, or play.
- `solver/` uses literal complete action spaces and engine-derived chance.
- `learning/` may approximate exact anchors but must not alter engine truth.
- `play/` and `commands/` are adapters and user surfaces.
- `config/`, `outputs/`, and `checkpoints/` are STL-only.
- Both players always observe leap timing; only Baku-as-Dropper may use 61.
- Do not add unaware or memory-impaired leap modes to the canonical project.

Read `docs/REGEN2RL.md` only for training/promotion work and
`docs/LITERATURE_ASSESSMENT.md` only for algorithmic precedent.

Run `uv run python -m pytest stl/tests -q` for project validation.

