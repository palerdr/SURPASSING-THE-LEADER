# Install and build

The project uses opam 2.x, OCaml 5.3.0, Dune, and OCamlformat. On Windows,
opam provides the compiler toolchain; no Unix terminal library is required.

Create the project switch once:

```text
opam switch create stl-dth-ocaml ocaml-base-compiler.5.3.0 --yes
```

## GLPK is a build prerequisite

`lib/solver/matrix_game.ml` links GLPK in process through `lp-glpk`, so a
GLPK built for the switch's target must exist before `opam install` will
succeed. On Linux and macOS the system package is enough
(`apt install libglpk-dev`, `brew install glpk`), and `conf-glpk` finds it.

On Windows there is no depext: `conf-glpk` installs but provides nothing, and
Cygwin ships only a Cygwin-native GLPK, which cannot link against the
`system-mingw` toolchain. Build a mingw-targeted static library once, from
the GLPK 5.0 release (SHA-256
`4a1013eebb50f728fc601bdd833b0b2870333c3b3e5a816eeba921d95bec6f15`):

```text
curl -sSLO https://ftp.gnu.org/gnu/glpk/glpk-5.0.tar.gz
tar xzf glpk-5.0.tar.gz && cd glpk-5.0
PATH="$(opam var --switch=stl-dth-ocaml prefix)/bin:$PATH" \
  ./configure --host=x86_64-w64-mingw32 --disable-shared --without-gmp
```

`configure` only needs to produce `config.h`; GLPK is plain C, so compile it
directly rather than depending on a `make` that Git Bash does not ship. Build
every `src/**/*.c` with `-DHAVE_CONFIG_H=1 -D__WOE__=1` and an `-I` for each
directory under `src/`, then archive the objects with
`x86_64-w64-mingw32-ar rcs libglpk.a`. `__WOE__` is required: without it GLPK
compiles its POSIX branch and the link fails on `gmtime_r` and `strerror_r`,
which mingw does not provide.

Copy `src/glpk.h` and `libglpk.a` into the switch's mingw sysroot, under
`<opam root>/.cygwin/root/usr/x86_64-w64-mingw32/sys-root/mingw/{include,lib}`.

Install the package and formatting dependencies:

```text
opam install --switch=stl-dth-ocaml .\src\ocaml --deps-only --yes

Install the language server into the same switch:

```text
opam install --switch=stl-dth-ocaml ocaml-lsp-server --yes
```
```

The switch can be used without modifying the current shell environment:

```text
opam exec --switch=stl-dth-ocaml -- dune build --root src/dth_ocaml
opam exec --switch=stl-dth-ocaml -- dune runtest --root src/dth_ocaml
opam exec --switch=stl-dth-ocaml -- dune fmt --root src/dth_ocaml
```

To verify the tools directly:

```text
opam exec --switch=stl-dth-ocaml -- ocamlc -version
opam exec --switch=stl-dth-ocaml -- ocamlformat --version
opam exec --switch=stl-dth-ocaml -- ocamllsp --version
```

`ocamlformat` must be run through this switch. Running a different global
binary can report syntax errors or use a formatter version that does not match
the project’s `.ocamlformat` file. VS Code is pinned to this switch in
`.vscode/settings.json`; restart the OCaml language server after installing it.
