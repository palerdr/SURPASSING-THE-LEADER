# OCaml exact solver instructions

This subtree is a standalone Dune project holding one library, `dth_solver`,
with `lib/solver/exact.ml` for states, transitions, and the revival model, and
`lib/solver/matrix_game.ml` for simultaneous matrix values. Keep it independent
of the Python and Rust peer projects; GLPK is a C build dependency, not a peer.

- Do not implement a simplex method here. `matrix_game.ml` delegates to GLPK
  and its job is to state the two linear programs and certify the answer.
- Every accepted matrix value carries a saddle gap of at most `1e-6`, matching
  the bound `src/dth/docs/GAME_AND_SOLVER.md` publishes. Never widen that gate
  or return a value that failed it.
- Matrix-game tests must include at least one asymmetric game. Symmetric games
  solved by the uniform mixture pass even for a broken solver.
- Actions are literal seconds `1..60`.
- Successful-check ST is inclusive: `check - drop + 1`.
- Revival is identity-neutral and depends only on pre-failure ST and accrued
  TTD.
- Use the repository-wide frozen revival model in `docs/REVIVAL_MODEL.md`:
  `0.95 * (1 - s / 240) * 0.75^(t / 60)` when the dose is eligible.

Validate with:

```sh
opam exec --switch=stl-dth-ocaml -- dune build --root src/ocaml
opam exec --switch=stl-dth-ocaml -- dune runtest --root src/ocaml
opam exec --switch=stl-dth-ocaml -- dune fmt --root src/ocaml
```
