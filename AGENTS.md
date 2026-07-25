# Agent Instructions

## Communication

- In final responses only, start with `James -`.
- Keep progress updates concise and factual.
- Treat pre-existing uncommitted changes as user work.

## Repository boundaries

- `docs/` owns repository-wide canonical game and solver contracts.
- `papers/` owns primary evidence, the whitepaper, and cited literature.
- `src/stl/`, `src/dth/`, `src/abstract/`, and `src/dth_ocaml/` are peer
  projects. They must not import one another. `arena/` is a neutral play
  surface: it may consume their public interfaces, but they must not import it
  or one another.
- `src/crates/` is a shared Rust workspace; Python remains behavioral authority
  until an explicit parity contract says otherwise.
- Each project owns its configs, docs, tests, checkpoints, and outputs.
- Generated data must remain gitignored.

Read the nearest nested `AGENTS.md` only when working in that subtree. Do not
place subsystem status, plans, or invariants in the repository root.

## Frozen global rules

- Actions are literal seconds beginning at 1; action 0 is illegal.
- A successful check uses inclusive elapsed time: `ST = check - drop + 1`.
- Normal action sets are 1..60.
- In STL's leap window only Baku as Dropper may choose 61; Checker remains
  capped at 60. Both players know the leap rule from game initialization.
- DTH and abstract do not inherit STL-only leap or information-state mechanics.
- The historical `dth_ocaml` name is retained for provenance, but that project
  implements the leap-aware STL public game and follows the STL rule above.

Any rules change must update canonical docs, evidence citations, schemas, and
tests together. Do not weaken solver firewalls, gates, tolerances, or artifact
validation to make a change pass.

## Validation

```powershell
uv run python -m pytest --collect-only -q
uv run python -m pytest -q
cargo test --workspace
opam exec --switch=stl-dth-ocaml -- dune build --root src/dth_ocaml
opam exec --switch=stl-dth-ocaml -- dune runtest --root src/dth_ocaml
```

After code changes, run `graphify update .`. Use `graphify query`, `path`, or
`explain` for architecture questions when `graphify-out/graph.json` exists.
