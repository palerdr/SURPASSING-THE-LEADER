# OCaml DTH solver types

This Dune project is intentionally small. The only library is
`lib/solver/exact.ml` with its interface in `lib/solver/exact.mli`; it owns the
role-relative state and chance-branch types plus the frozen revival model. It
does not contain a transition engine or matrix-game implementation.

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
