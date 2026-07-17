# Canonical extensive-form contract

This file freezes the repository-wide rules boundary used by the Python engine,
the Rust mirror, exact search, staged toy rules, and the no-leap `pure` solver.
Changing any rule below requires a coordinated code, test, documentation, and
artifact-schema change.

## Documentary basis

- [`SURPASSING THE LEADER- HAL DOC.pdf`](./SURPASSING%20THE%20LEADER-%20HAL%20DOC.pdf),
  PDF page 6, gives the cylinder a "max capacity of 5 minutes" and says it is
  injected immediately when filled.
- The same PDF, pages 12-13, explains that the cylinder holds and injects at
  most five minutes even if broader accumulated exposure is larger.
- The same PDF, page 43, treats exactly five minutes of total time dead as
  potentially revivable. This statement concerns cumulative TTD, not cylinder
  capacity or the current injection dose.
- [`Leader's Deviation Strategy.pdf`](./Leader%E2%80%99s%20Deviation%20Strategy.pdf),
  PDF page 127, treats cylinder accumulation crossing beyond five minutes as
  an instant-loss boundary.

## Frozen boundary semantics

Let `q` be the current injected death duration and `t` the player's cumulative
TTD before this death.

1. The cylinder capacity is exactly 300 seconds.
2. Reaching or crossing that capacity triggers immediate injection; the
   physical injected dose is capped at 300 seconds.
3. A current dose `q >= 300` has revival probability zero in this model.
4. Resulting cumulative TTD uses a strict fatal boundary:

   ```text
   t + q > 300   => revival probability 0
   t + q == 300  => revival remains eligible
   ```

5. The revival calculation uses prior TTD `t`. The death event then adds `q`
   to recorded TTD; only successful revival produces a live successor, resets
   the cylinder, and carries that updated TTD forward.
6. Consequently, a player may occupy a live state with TTD exactly 300. Any
   later positive-duration death is necessarily fatal because it would take
   total TTD over 300.

The canonical engine's eligible-branch probability is

```text
(1 - (q / 300)^3)
* 0.85^(t / 60)
* max(0.4, 0.88^cprs)
* physicality
```

The `pure` abstraction deliberately removes CPR fatigue and physicality and
uses `2^(-t/240)` for its prior-TTD factor. It retains the same dose and total
TTD boundary inequalities.

## Executable authorities and artifacts

- `stl/engine/game.py` is the canonical Python transition authority.
- `crates/stl_solver/src/game.rs` must preserve exact rule parity.
- `stl/solver/exact.py` may duplicate only audited scalar rule helpers.
- `pure/solver.py` is authoritative for the documented no-leap abstraction.

Exact targets, checkpoints, and replay generated under older TTD boundary
behavior are incompatible and must not be mixed with new artifacts. The pure
target schema is therefore `pure-v3-ttd-strict-overflow`; canonical artifacts
remain protected by their source/config digests.
