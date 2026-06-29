# Docs Layout

Design notes, canon references, reports, solver literature, and long-form solver
rationale live here. The root `README.md` is the current orientation document;
use this directory for background and audit evidence, not executable behavior.

- `papers/` - solver literature PDFs, links, and the current literature assessment.
- `whitepaper/` - LaTeX whitepaper source and generated build output.
- `reports/` - generated run reports, matrices, checklists, and JSON summaries.
- `audits/` - generated audit traces and counterfactual outputs.
- `canon/` - source PDFs and canon-facing reference material when present locally.
- `architecture/` - target architecture notes, including older rewrite plans.

Key files:

- `SOLVER_EXTENSION_GOAL.md` - governing solver-extension gate contract and progress log.
- `papers/literature_assessment.md` - audit of the current approach against solver literature.
- `phase_g_i1_validation.md` - validation notes from the earlier generation chain.
