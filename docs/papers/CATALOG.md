# Paper catalog

Primary game evidence now lives one level up in
[`../game-sources/`](../game-sources/), beside the evidence ledger it supports.
This directory holds cited literature and primary game evidence; the project
paper lives at the repository root under `paper/`.

## Primary game evidence

- `../game-sources/SURPASSING THE LEADER- HAL DOC.pdf` — game chronology and
  rule commentary used by the evidence ledger.
- `../game-sources/Leader-Deviation-Strategy.pdf` — deviation-plan analysis and
  cylinder-overflow evidence.
- [`../game-sources/IN_DEPTH_SUMMARY.md`](../game-sources/IN_DEPTH_SUMMARY.md) —
  a full round-by-round reading of the arc. Its per-turn state headers are a
  complete numeric ledger of the canonical match and carry the repository's
  sharpest quantitative rule evidence: they confirm `q = s + 60` five times, the
  inclusive ST convention twelve times, and the strict 300-second cumulative
  boundary as a worked two-second margin. Cited throughout
  [`../game-sources/EVIDENCE.md`](../game-sources/EVIDENCE.md).

The ledger records one transcription error, catalogued as
[E-LEDGER-ERRATUM](../game-sources/EVIDENCE.md#e-ledger-erratum).

## Project paper

The mathematical paper lives at the repository root:
`paper/dth_exact_solution.tex` (editable source) and
`paper/dth_exact_solution.pdf` (checked-in rendered copy). It records the
certified exact solution of the complete DTH game (2026-07-30) and retains the
matrix-value Lipschitz and saddle-gap propositions the repository cites. The
retired STL/AlphaZero whitepaper it replaced remains available in Git history
(`docs/papers/whitepaper/` before 2026-07-30).

## Solver literature

The `references/` directory contains the cited CFR, CFR+, AlphaZero, DeepStack,
ReBeL, and duel papers. STL-specific interpretation belongs in
`src/stl/docs/LITERATURE_ASSESSMENT.md`; the PDFs remain unmodified source
material.

`references/duel1712.pdf` is Alpern & Howard, *A Short Solution to the
Many-Player Silent Duel with Arbitrary Consolation Prize*, arXiv:1712.00274. It
is an n-player non-constant-sum tournament and is **not** a model for this
game's stage matrix; it is retained for the timing-game framing only.
