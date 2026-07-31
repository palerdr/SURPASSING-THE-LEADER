# Agent Instructions

## Communication

- In final responses only, start with `James -`.
- Keep progress updates concise and factual.
- Treat pre-existing uncommitted changes as user work.

## Repository boundaries

- `docs/` owns repository-wide canonical game and solver contracts.
- `paper/` owns the project's mathematical paper. Re-render and visually
  inspect its PDF after changing the TeX.
- `docs/papers/` owns primary evidence and cited literature.
- `src/stl/`, `src/dth/`, `src/abstract/`, and `src/dth_ocaml/` are peer
  projects. They must not import one another. `src/arena/` is a neutral play
  surface: it may consume their public interfaces, but they must not import it
  or one another.
- `src/crates/` is a shared Rust workspace; Python remains behavioral authority
  until an explicit parity contract says otherwise.
- Each project owns its configs, docs, tests, checkpoints, and outputs.
- Generated data must remain gitignored.

Read the nearest nested `README.md` only when working in that subtree. Do not
place subsystem status, plans, or invariants in the repository root.

`CLAUDE.md` at the repository root imports every subtree `README.md` so Claude
Code loads them all as instruction files. Adding a subtree means adding its
`README.md` and an import line there.

## Frozen global rules

- Actions are literal seconds beginning at 1; action 0 is illegal.
- A successful check uses inclusive elapsed time: `ST = check - drop + 1`.
- Normal action sets are 1..60.
- The injected dose is vial contents plus a fixed 60-second penalty, `q = s + 60`.
- Capacity is 300 seconds; `t + q > 300` is fatal and `t + q == 300` stays
  revival-eligible when `q < 300`.
- One revival-probability surface is frozen for the whole repository in
  `docs/REVIVAL_MODEL.md`. No project may carry its own revival constants, and
  none may reintroduce an explicit CPR count or per-player physicality below
  ladder rung L2.
- In STL's leap window only Baku as Dropper may choose 61; Checker remains
  capped at 60. Both players know the leap rule from game initialization.
- DTH and abstract do not inherit STL-only leap or information-state mechanics.
- The `dth_ocaml` project is an independent OCaml implementation of **pure
  DTH**, not STL: actions are literal seconds 1..60 and it has no leap window.
  It exists as a hand-written reference for the exact solver and is held to
  the same frozen rules and the same 1e-6 saddle-gap gate as its Python peer.
- `docs/FORMULATION_LADDER.md` fixes which games are claimed at all. Work that
  does not sit on a rung is not a supported claim.

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
