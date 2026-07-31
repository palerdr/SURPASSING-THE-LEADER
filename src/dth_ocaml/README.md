# OCaml DTH solver types

This Dune project is intentionally small. `lib/solver/exact.ml` owns the
role-relative state and chance-branch types, the transition expansion, and the
frozen revival model. `lib/solver/matrix_game.ml` turns one simultaneous 60x60
payoff matrix into a certified value.

## Matrix values come from GLPK

`matrix_game.ml` does not implement a simplex method. It builds both players'
linear programs and hands them to GLPK through `lp-glpk`, then certifies the
returned pair: each policy must be a probability distribution, and the saddle
gap `max_d (Mq)_d - min_c (M^T p)_c` must not exceed `1e-6` — the same bound
`src/dth/docs/GAME_AND_SOLVER.md` publishes for the peer Python solver. A
matrix that fails either check raises `Matrix_game.Uncertified` rather than
returning a number.

This makes GLPK a build prerequisite. `conf-glpk` carries no Windows depext,
so opam cannot provision it there; see [`INSTALL.md`](INSTALL.md) for how to
get a mingw-targeted `libglpk` in place before `opam install lp-glpk`.

The OCaml solver follows the repository-wide model in
[`docs/REVIVAL_MODEL.md`](../../docs/REVIVAL_MODEL.md):

```text
P_rev(s, t) = 0.95 * (1 - s / 240) * 0.75^(t / 60)
```

The failed-check dose is `q = s + 60`. Revival is possible exactly when
`q < 300` and `t + q <= 300`; actions are literal seconds `1..60`; and a
successful check gains inclusive ST `check - drop + 1`.

See [`INSTALL.md`](INSTALL.md) for the Windows-friendly opam switch and
formatter commands, and [`RULES.md`](RULES.md) for the executable contract.

## Working in this subtree

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
opam exec --switch=stl-dth-ocaml -- dune build --root src/dth_ocaml
opam exec --switch=stl-dth-ocaml -- dune runtest --root src/dth_ocaml
opam exec --switch=stl-dth-ocaml -- dune fmt --root src/dth_ocaml
```
