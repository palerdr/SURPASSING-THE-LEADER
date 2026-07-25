# OCaml Drop The Handkerchief engine and solver

This Dune project contains the original CS 3110 engine/TUI and the `cfr` branch
solver foundation. It is embedded in the monorepo at `src/ocaml/`

The parity pass keeps the original architecture while enforcing the repository
rules: literal seconds begin at 1, successful ST is
`check - drop + 1`, only Baku as Dropper can use second 61, and Checker remains
capped at 60. Revival is identity-neutral and uses only vial ST and accrued TTD.

See [`INSTALL.md`](INSTALL.md) for the local toolchain and commands.
