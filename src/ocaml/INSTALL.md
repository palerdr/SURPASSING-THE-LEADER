# Install & Build

## Prerequisites

- **opam** 2.x
- An opam switch with **OCaml 5.3.0** (the course switch `cs3110-2026sp` works as-is).
- A terminal that supports cursor positioning. For full color, 24-bit ("truecolor") support is required; otherwise the game will fall back to a monochrome palette.

If you do not yet have a compatible switch:

```bash
opam switch create stl-dth-ocaml ocaml-base-compiler.5.3.0
eval $(opam env --switch=stl-dth-ocaml)
```

All commands below may be run from the monorepo root by passing
`--root src/ocaml`.

## Install dependencies

The `dth.opam` file declares every required library. Pull them in with:

```bash
opam install --switch=stl-dth-ocaml ./src/ocaml --deps-only --yes
```

This installs `dune` (>= 3.21), `alcotest`, and `notty`.

If `notty` fails to build, you may need its system dependency for terminal handling:
- Linux (Debian/Ubuntu/WSL): `sudo apt-get install pkg-config`
- macOS: `brew install pkg-config`

## Build

```bash
opam exec --switch=stl-dth-ocaml -- dune build --root src/ocaml
```

Or use the Makefile shortcut:

```bash
make build
```

A clean build produces no output. If `eval $(opam env)` is already loaded for this shell, you can drop the `opam exec --` prefix from every `dune` command below.

## Run

The interactive TUI:

```bash
opam exec --switch=stl-dth-ocaml -- dune exec --root src/ocaml dth-play
```

Launch flow:
1. **Splash screen** — press any key to continue.
2. **Display selection** — choose `1` (recommended for your terminal: Color on truecolor terminals, Safe / monochrome on Apple Terminal and other limited terminals) or `2` for the alternate palette.
3. **Game mode** — `1` Two Player (hot-seat, hand-off prompts between turns) or `2` Single Player vs Hal (you play Baku; Hal plays automatically from the canonical table).
4. **Match** — alternate dropping and checking until one side dies a final death.

Controls:
- Digits enter a drop time / check time, **Backspace** (or **Delete**) erases, **Enter** confirms within the valid range.
- **Enter** advances between informational screens (handoffs, NDD bar, archive feed, resolution, ending).
- **q**, **Q**, **Esc**, or **Ctrl-C** quit at any prompt.

## Tests

```bash
opam exec --switch=stl-dth-ocaml -- dune runtest --root src/ocaml
```

Or:

```bash
make test
```

The Alcotest suite covers the engine, unified revival model, Hal table, clock,
config, solver actions, policies, and transitions. A successful run currently
reports `126 tests run` and exit code `0`.

## Coverage (optional)

The library is pre-instrumented for `bisect_ppx`. To produce a coverage report:

```bash
opam install bisect_ppx --yes
opam exec -- dune runtest --force --instrument-with bisect_ppx
opam exec -- bisect-ppx-report html
```

Open `_coverage/index.html` to view per-module coverage.

## Verifying a fresh deployment

From a clean clone, the following sequence should complete without errors:

```bash
opam install . --deps-only --yes
opam exec -- dune build
opam exec -- dune runtest
opam exec -- dune exec dth-play
```

If the splash screen renders and the display/game-mode prompts respond to keystrokes, the deployment is healthy.
