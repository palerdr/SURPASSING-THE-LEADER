# Install and build

The project uses opam 2.x, OCaml 5.3.0, Dune, and OCamlformat. On Windows,
opam provides the compiler toolchain; no Unix terminal library is required.

Create the project switch once:

```text
opam switch create stl-dth-ocaml ocaml-base-compiler.5.3.0 --yes
```

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
opam exec --switch=stl-dth-ocaml -- dune build --root src/ocaml
opam exec --switch=stl-dth-ocaml -- dune runtest --root src/ocaml
opam exec --switch=stl-dth-ocaml -- dune fmt --root src/ocaml
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
