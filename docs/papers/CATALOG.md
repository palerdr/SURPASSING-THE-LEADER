# Paper catalog

Primary game evidence now lives one level up in
[`../game-sources/`](../game-sources/), beside the evidence ledger it supports.
This directory holds cited literature and the project whitepaper.

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

## Project whitepaper

- `whitepaper/stl_solver_whitepaper.tex` — editable source.
- `whitepaper/stl_solver_whitepaper.pdf` — checked-in rendered copy.

Note that the whitepaper predates the frozen revival model and the formulation
ladder; regenerate it before citing its rule statements.

## Solver literature

The `references/` directory contains the cited CFR, CFR+, AlphaZero, DeepStack,
ReBeL, and duel papers. STL-specific interpretation belongs in
`src/stl/docs/LITERATURE_ASSESSMENT.md`; the PDFs remain unmodified source
material.

`references/duel1712.pdf` is Alpern & Howard, *A Short Solution to the
Many-Player Silent Duel with Arbitrary Consolation Prize*, arXiv:1712.00274. It
is an n-player non-constant-sum tournament and is **not** a model for this
game's stage matrix; it is retained for the timing-game framing only.
