# OCaml exact solver instructions

This subtree is a standalone Dune project. Its only library is
`lib/solver/exact.ml` with interface `exact.mli`; keep it independent of the
Python and Rust peer projects.

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
