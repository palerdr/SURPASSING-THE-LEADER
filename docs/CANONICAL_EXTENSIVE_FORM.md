# Canonical Extensive-Form Contract

This document freezes repository-wide state and transition boundaries. Action
timing is owned separately by [`ACTION_TIMING.md`](ACTION_TIMING.md).

## Documentary boundary

The cylinder capacity and injection cap follow
[E-CYLINDER-CAP](../papers/game-sources/EVIDENCE.md#L21-L26); the strict
cumulative boundary follows
[E-TTD-EXACT](../papers/game-sources/EVIDENCE.md#L28-L33); crossing the cylinder
limit follows [E-OVERFLOW](../papers/game-sources/EVIDENCE.md#L35-L40).

<!-- canon:C-DEATH-BOUNDARY -->
## Death and revival

Let `q` be the injected duration and `t` prior cumulative time-to-death (TTD).

1. Cylinder capacity is exactly 300 seconds.
2. Reaching capacity triggers injection; the physical dose is capped at 300.
3. A current dose `q >= 300` has zero revival probability.
4. `t + q > 300` is fatal, while `t + q == 300` remains revival-eligible when
   `q < 300`.
5. Revival uses prior TTD `t`; a surviving successor carries `t + q`, resets
   the cylinder, and increments the relevant death/referee state.

STL's eligible revival probability is:

```text
(1 - (q / 300)^3) * 0.85^(t / 60) * max(0.4, 0.88^cprs) * physicality
```

DTH deliberately removes referee fatigue and physicality and uses its locally
documented prior-TTD factor. It retains the same strict boundary inequalities.

<!-- canon:C-AUTHORITY -->
## Executable authorities

- `stl/engine/game.py` owns full-game transitions.
- `stl/engine/actions.py` owns STL action legality.
- `crates/stl_solver/src/game.rs` must match the Python engine where covered.
- `dth/solver.py` owns the no-leap DTH abstraction.
- `toy/rules.py` owns only the enumerated toy abstraction.

Solvers may use audited scalar helpers but must not reimplement divergent game
rules. Rule-bound artifacts must identify their schema and source/config digest.
