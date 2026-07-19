# Action Timing Authority

This is the sole repository authority for action seconds and Squandered Time
(ST). Engine, solver, learning, Rust, tests, and artifact schemas must agree.

## Evidence and interpretation

The sources establish that ST cannot be zero and that the narrated immediate
“0 seconds” check produces one second of ST. See
[E-ST-NONZERO](../papers/game-sources/EVIDENCE.md#L8-L12) and
[E-INSTANT-CHECK](../papers/game-sources/EVIDENCE.md#L14-L19).
The narrative zero is therefore immediate action in the first second, not a
literal solver action.

<!-- canon:C-ACTION-NORMAL -->
## Literal-second convention

For a normal half-round:

```text
drop d  in {1, ..., 60}
check c in {1, ..., 60}
success iff c >= d
successful ST = c - d + 1
failure iff c < d
failed-check penalty = 60 seconds
```

Action 1 means acting immediately. Action 0 and passing are illegal. A
same-second pair succeeds and accumulates one second.

<!-- canon:C-ACTION-LEAP -->
## STL leap window

Both players know the leap rule from initialization. Knowledge does not alter
structural legality: only Baku while acting as Dropper may additionally choose
61 in the leap window. Checker remains capped at 60, so `(d=61, c<=60)` fails.
DTH and the canonical toy example have no leap-second action.

## Dense policy indices

STL policy tensors have width 62: index 0 is illegal padding, indices 1..60 are
literal normal seconds, and index 61 is the conditional Baku drop. DTH policies
have 60 actions corresponding directly to seconds 1..60.

Any artifact produced with action 0, a pass action, exclusive `c-d` ST, or a
different leap mask is incompatible and must fail schema validation.
