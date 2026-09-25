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
- `src/stl/`, `src/dth/`, `src/dth_compact/`, `src/abstract/`, and
  `src/dth_ocaml/` are peer projects. They must not import one another.
  `src/arena/` is a neutral play surface: it may consume their public
  interfaces, but they must not import it or one another.
- `src/formal/` is a Lean project that machine-checks the solvers' mathematics.
  It imports no other project, and no project imports it.
- `src/crates/` is a shared Rust workspace; Python remains behavioral authority
  until an explicit parity contract says otherwise.
- `src/arena/webclient/` is the TypeScript browser client and `src/arena/web/`
  is its server. Both are arena play surfaces, like `cli.py` and `tui.py`. The
  client renders state and collects input; it never derives game rules and
  never receives an unrevealed action.
- `src/hal_lab/` is the Hal research lab. It owns Hal training and the frozen
  evidence of the finished studies, with the ignored root `outputs/` store.
  It may import `arena`, `stl.engine`, `stl.solver.canonical`, `dth.agent`,
  and `dth.solver`, and no project imports it.
- Each project owns its configs, docs, tests, checkpoints, and outputs.
- Generated data must remain gitignored. Character sprites under `art/sprites/`
  are source art, not generated data, and are tracked.

Read the nearest nested `README.md` only when working in that subtree. Do not
place subsystem status, plans, or invariants in the repository root.

`CLAUDE.md` at the repository root imports every subtree `README.md` so Claude
Code loads them all as instruction files. Adding a subtree means adding its
`README.md` and an import line there.

## Layers

`docs/PROJECTS.toml` declares the import rules of each project.
`tests/meta/test_layer_boundaries.py` enforces them on every
`src/<id>/**/*.py` file, tests included, and it reads imports inside
functions too.

| Layer | Projects | May import |
| --- | --- | --- |
| Peers | `stl`, `dth`, `dth_compact`, `abstract`, `dth_ocaml`, `dth_cpp` | No other project |
| Shared accelerators | `crates` | No other project |
| Proofs | `formal` | Nothing |
| Play surface | `arena` | `stl.engine`, `stl.solver.canonical`, `dth.agent`, `dth.solver`, `abstract` |
| Lab | `hal_lab` | `arena`, `stl.engine`, `stl.solver.canonical`, `dth.agent`, `dth.solver` |

- A cross-project import must sit under a `may_import` entry of the importer
  and under a `public_interfaces` entry of the owner. The compiled `*_rs`
  extension modules are outside this check.
- `hal_lab` declares no public interface, so no project can import it.
- Inside `arena`, `contracts.py`, `agent.py`, `session.py`, `match.py`, and
  `variants.py` import `stl.engine` and each other alone. `dth` and
  `abstract` enter through the adapters and `arena/policies/`.
- `stl`, `dth`, `abstract`, `dth_compact`, and `arena` do not import `torch`,
  `gymnasium`, `stable_baselines3`, or `sb3_contrib`. The files in
  `forbid_imports_exempt` are today's exceptions: arena's Hal training and
  evaluation code. Add no file to that list. `arena/policies/__init__.py`
  imports nothing. `hal_lab` may import these packages.
- `tests/meta/test_root_layout.py` limits the root to the governance files,
  `docs/`, `paper/`, `src/`, `tests/`, and the ignored `outputs/` store. Its
  legacy list names the tracked root entries that a planned move removes.
- `tests/meta/test_formal_citations.py` compares every `file:line` citation in
  `src/formal` with `src/formal/citations.lock`. After an edit moves cited
  text, run `uv run python tests/meta/formal_citations.py --refresh`. After an
  edit changes cited text, review the proof, then run the same script with
  `--accept <path>`.

## Fingerprints and locks

- `src/dth/complete_tablebase.py` hashes the root `uv.lock` into the DTH
  artifact digest. A root `uv.lock` change ships with rebuilt
  `src/dth/artifacts/complete_full_v1` and `complete_fast_v1` in the same
  commit series.
- `stl.solver.leap_build.builder_hash()` hashes the builder sources with the
  root `pyproject.toml`, `uv.lock`, and `Cargo.lock`, and a resume refuses a
  changed hash. Do not start or resume a leap build across an edit to the root
  `pyproject.toml` or `uv.lock`.
- The builder hash reads `src/stl/solver/leap_*.py` by glob. Name a new STL
  builder module `src/stl/solver/leap_*.py`, and put new non-builder STL code
  outside `src/stl/solver/`.
- `uv sync` at the root removes the three maturin extensions. Run commands
  with `uv run`, and rebuild the extensions after a sync.

## Frozen global rules

- Actions are literal seconds beginning at 1; action 0 is illegal.
- A successful check uses inclusive elapsed time: `ST = check - drop + 1`.
- Normal action sets are 1..60.
- The injected dose is vial contents plus a fixed 60-second penalty, `q = s + 60`.
- Capacity is 300 seconds; `t + q > 300` is fatal and `t + q == 300` stays
  revival-eligible when `q < 300`.
- One revival-probability surface is frozen for the whole repository in
  `docs/REVIVAL_MODEL.md`. No project may carry its own revival constants or
  reintroduce an explicit CPR count or per-player physicality as a revival-odds
  input at any rung.
- In STL's leap window only Baku as Dropper may choose 61; Checker remains
  capped at 60. Both players know the leap rule from game initialization.
- DTH and abstract do not inherit STL-only leap or information-state mechanics.
- The `dth_ocaml` project is an independent OCaml implementation of **pure
  DTH**, not STL: actions are literal seconds 1..60 and it has no leap window.
  It exists as a hand-written reference for the exact solver and is held to
  the same frozen rules and the same 1e-6 saddle-gap gate as its Python peer.
- The `dth_compact` project is the paper's one-file solver of **pure DTH**. It
  shares `src/dth/`'s rules, revival surface, and reference anchors, is held
  to the same 1e-6 saddle-gap gate, and is not the arena's policy provider.
- `docs/FORMULATION_LADDER.md` fixes which games are claimed at all. Work that
  does not sit on a rung is not a supported claim.

Any rules change must update canonical docs, evidence citations, schemas, and
tests together. Do not weaken solver firewalls, gates, tolerances, or artifact
validation to make a change pass.

## Validation

```powershell
uv run python -m pytest --collect-only -q
uv run python -m pytest -q
uv run --project src/dth_compact pytest src/dth_compact/tests -q
cargo test --workspace
npm --prefix src/arena/webclient run typecheck
opam exec --switch=stl-dth-ocaml -- dune build --root src/dth_ocaml
opam exec --switch=stl-dth-ocaml -- dune runtest --root src/dth_ocaml
lake -d src/formal build
```

The typecheck needs `npm --prefix src/arena/webclient install` once. The browser
client is not covered by `pytest`, so skipping it leaves the front end
unchecked.

`pytest` runs distributed at `-n 8`; that width is deliberate, because one
worker per logical core exhausts Windows handles importing torch's
dependencies and makes the workers contend for the GPU. Add `-n 0` for a
readable serial traceback when something fails.

Everything runs by default, including the `slow` tests that build real complete
artifacts end to end — they are the only coverage of resume and backend
parity, and under xdist they cost about 19 seconds. Use `-m "not slow"` for a
tighter loop when iterating on unrelated code, never as the check before a
merge.

After code changes, run `graphify update .`. Use `graphify query`, `path`, or
`explain` for architecture questions when `graphify-out/graph.json` exists.
