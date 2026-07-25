# OCaml Engine Instructions

This subtree is a standalone Dune project imported from the authorized Cornell
course repository. Preserve the upstream architecture and authorship recorded
in `PROVENANCE.md`.

- Do not import Python or Rust peer projects.
- Keep literal action seconds, inclusive successful-check ST, and leap legality
  aligned with the repository root frozen rules.
- Revival probability is identity-neutral and depends on exactly two state
  variables: ST in the vial and accrued TTD.
- Update engine interfaces, TUI display, solver legality, documentation, and
  Alcotest coverage together for rule changes.

Validate with:

```sh
opam exec --switch=stl-dth-ocaml -- dune build --root src/ocaml
opam exec --switch=stl-dth-ocaml -- dune runtest --root src/ocaml
```
