# Formal verification in Lean

This subtree is a Lean 4 project. It machine-checks the core mathematics of
the pure-DTH solver and of the STL leap window: the rules and revival
eligibility, the quotient, the potential and backward induction, the
Toeplitz stage matrix and the equalizer recurrence, the saddle-gap
certificate, the accumulated error bound, and the value of the leap-window
stage.

The Lean kernel checks every proof. The build fails when a declaration in
`Formal` depends on `sorry`, on `native_decide`, or on any axiom other than
`propext`, `Classical.choice`, and `Quot.sound`.

[`CLAIMS.md`](CLAIMS.md) maps each claim to the Lean theorem that proves it.

## Layout

- `Formal/MatrixGame/` holds finite zero-sum games: mixed strategies,
  certificate bounds, the minimax theorem (from Mathlib's Sion theorem), and
  value transformations.
- `Formal/DTH/` holds pure DTH: rules, the revival surface, profiles, the
  potential `Φ`, the Bellman value, the quotient theorem, and the accumulated
  error bound.
- `Formal/Toeplitz/` holds the stage-matrix structure: persymmetry, row
  differences, and the equalizer recurrence.
- `Formal/Leap/Window.lean` holds the STL leap window.
- `Formal/Extended/` is a separate library, rooted at `FormalExtended.lean`,
  with deeper results: the revival surface, the 61-class stage matrix, the
  pure-saddle rung, monotone payoffs, the leap crash basis, the binary64
  shortcut certificate, and the STL leap game. The core never imports it.
- `Formal/Audit.lean` defines `#audit_axioms`; `FormalAudit.lean` runs it over
  the whole library as part of the default build.

The proofs are about real-number mathematics. Each definition mirrors the
cited code. Floating-point claims in `Formal/Extended/Float/` use an explicit
abstract rounding model, never IEEE bit-level facts.

## Working in this subtree

- Build from the repository root with `lake -d src/formal build`. The first
  build needs `lake -d src/formal exe cache get`, which downloads Mathlib's
  compiled files.
- Never add `sorry`, `admit`, `axiom`, `native_decide`, `implemented_by`, or
  `extern`. The audit rejects them, and weakening the audit is not allowed.
- Mirror the code a theorem is about. When the doc and the code disagree,
  formalize the code and state the difference in the docstring.
- Every main theorem's docstring names the claim ids from `CLAIMS.md` and the
  source `file:line` it formalizes. Update `CLAIMS.md` in the same change.
- Write each citation as a repository-relative path, or as a bare file name
  that one tracked file alone has. Write several ranges as one list with no
  spaces, as in `path:13-16,191-206`. `citations.lock` records the SHA-256 of
  each cited span, and `tests/meta/test_formal_citations.py` fails when cited
  text moves or changes. After an edit moves cited text, run
  `uv run python tests/meta/formal_citations.py --refresh`; it rewrites the
  ranges in the Lean text and in the lock. The script follows a move only
  when the text, with up to three lines of context on each side, is unique in
  its file. Otherwise it reports the span as changed. After an edit changes
  cited text, or after you add a citation, check the proof against the text
  and fix the citation's range, then run the same script with
  `--accept <path>`.
- When a solver's mathematics changes, update the matching Lean module in the
  same change, or record in `CLAIMS.md` that the claim is no longer verified.
- This project imports no other project, and no project imports it.
- Lean `v4.34.0` and Mathlib `v4.34.0` are pinned in `lean-toolchain` and
  `lakefile.toml`. Mathlib and its dependencies download under the ignored
  `build/packages/` directory, and compiled files go to the ignored `.lake/`.
  Change the pins only together, and rebuild before you commit.
- Check one file quickly with `lake env lean <file>` from `src/formal/`.
