# Paper catalog

Primary game evidence now lives one level up in
[`../game-sources/`](../game-sources/), beside the evidence ledger it supports.
This directory holds cited literature. Primary game evidence lives in
`../game-sources/`, and the project paper lives at the repository root under
`paper/`.

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

## PM Hal opponent-modeling literature

The PM Hal design note in `paper/pm_hal.tex` cites the following primary
sources. They motivate separate mechanisms and evaluation questions; none is
treated as direct evidence about people playing this repository's DTH game.

- Johanson, Zinkevich & Bowling, *Computing Robust Counter-Strategies*
  ([NeurIPS paper](https://papers.nips.cc/paper_files/paper/2007/file/6e7b33fdea3adc80ebd648fffb665bb8-Paper.pdf),
  PDF pp. 5--6) — restricted Nash response and the exploitation/worst-case
  frontier.
- Johanson & Bowling, *Data Biased Robust Counter Strategies*
  ([PMLR](https://proceedings.mlr.press/v5/johanson09a.html), proceedings
  pp. 270--272; PDF pp. 6--8) — evidence-scaled confidence. PM Hal's additional
  change and agreement factors are project heuristics, not results from this
  paper.
- Ganzfried & Sandholm, *Safe Opponent Exploitation*
  ([author PDF](https://www.cs.cmu.edu/~sandholm/www/safeExploitation.teac15.pdf),
  article pp. 8:1--8:3 and 8:12--8:16) — expected-profit safety across
  repetitions of one game. Its cumulative theorem is not claimed for changing
  continuation-adjusted DTH stages.
- Herbster & Warmuth, *Tracking the Best Expert*
  ([DOI](https://doi.org/10.1023/A:1007424614876), pp. 151--160) — the Fixed
  Share update used by PM Hal's role-separated source and controller mixtures.
- Adams & MacKay, *Bayesian Online Changepoint Detection*
  ([arXiv](https://arxiv.org/abs/0710.3742), PDF pp. 2--3) — run-length
  inference adapted to a truncated categorical Dirichlet-multinomial model.
- Camerer & Ho, *Experience-Weighted Attraction Learning in Normal Form Games*
  ([open record](https://authors.library.caltech.edu/records/kgnbx-e2z22),
  journal pp. 827--835) — reinforcement and belief learning as competing or
  hybrid behavioral mechanisms.
- Hampton, Bossaerts & O'Doherty, *Neural correlates of mentalizing-related
  computations during strategic interactions*
  ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2373314/), journal
  pp. 6741--6744) — evidence that a model of how one's revealed actions
  influence an opponent can improve prediction in a repeated inspection game.
- Nassar et al., *An Approximately Bayesian Delta-Rule Model Explains the
  Dynamics of Belief Updating in a Changing Environment*
  ([DOI](https://doi.org/10.1523/JNEUROSCI.0822-10.2010), journal
  pp. 12366--12370) — surprise-dependent human learning rates and gradual
  post-change recovery.
- Camerer, Ho & Chong, *A Cognitive Hierarchy Model of Games*
  ([DOI](https://doi.org/10.1162/0033553041502225), journal pp. 861--866) —
  bounded recursive reasoning as a possible human archetype, not a permanent
  player label.
- Gneiting & Raftery, *Strictly Proper Scoring Rules, Prediction, and
  Estimation* ([DOI](https://doi.org/10.1198/016214506000001437), journal
  pp. 359--363) — why the PM benchmark leads with prequential log and Brier
  scores rather than accuracy alone.
