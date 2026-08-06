# CLAUDE.md

The root `AGENTS.md` holds this repository's global rules. Each subtree then
carries its own `README.md`, whose "working in this subtree" guidance is
binding for work inside that directory -- treat it as if it were a `CLAUDE.md`.

The root file below is always in force. The per-project files are imported so
they are available in context; when they conflict with the root file, the root
file's frozen global rules win.

## Root instructions

@AGENTS.md

## Project instructions

@src/stl/README.md
@src/dth/README.md
@src/abstract/README.md
@src/dth_ocaml/README.md
@src/crates/README.md
@src/arena/README.md
@docs/papers/README.md

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

When a new subtree with its own binding `README.md` is added, add an import line
for it here. When a subtree is removed or renamed, remove or update its line —
a stale import is a silent context gap.
