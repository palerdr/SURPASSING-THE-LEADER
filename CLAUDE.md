# CLAUDE.md

This repository's instructions live in `AGENTS.md` files, one per subtree. Treat
every one of them as if it were a `CLAUDE.md`: its contents are binding for work
inside that directory.

The root file below is always in force. The per-project files are imported so
they are available in context; when they conflict with the root file, the root
file's frozen global rules win.

## Root instructions

@AGENTS.md

## Project instructions

@src/stl/AGENTS.md
@src/dth/AGENTS.md
@src/abstract/AGENTS.md
@src/ocaml/AGENTS.md
@src/crates/AGENTS.md
@arena/AGENTS.md
@docs/papers/AGENTS.md

## Canonical contracts

These are documents, not instructions, but no rules question should be answered
without them. Read the relevant one before changing anything that touches game
rules, state encodings, or artifact schemas.

- `docs/FORMULATION_LADDER.md` — which games this repository claims to model,
  as rungs L0–L4. A formulation not on the ladder is not a supported claim.
- `docs/CANONICAL_EXTENSIVE_FORM.md` — state and transition boundaries.
- `docs/ACTION_TIMING.md` — action seconds and squandered time.
- `docs/REVIVAL_MODEL.md` — the single frozen revival-probability surface,
  shared by every rung. No project may carry its own revival constants.
- `docs/FOUNDATIONS.md` — shared solver mathematics.
- `docs/game-sources/EVIDENCE.md` — the evidence ledger that licenses the rules,
  and the explicit boundary between what is documentary and what is modelled.

## Maintaining this file

When a new subtree with its own `AGENTS.md` is added, add an import line for it
here. When a subtree is removed or renamed, remove or update its line — a stale
import is a silent context gap.
