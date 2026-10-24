# DTH workflows

Run commands from the repository root. Generated artifacts, reports, and
checkpoints remain under ignored paths owned by `src/dth/`.

## Build or verify the complete solution

```powershell
uv run python -m dth complete
uv run python -m pytest src/dth/tests/test_complete_facade.py `
  src/dth/tests/test_complete_sweep_python.py `
  src/dth/tests/test_complete_potential.py -q
```

The build uses `src/dth/config/complete_full_v1.yaml`, resumes by descending
potential layer, and writes `src/dth/artifacts/complete_full_v1/`. It runs
the Python v2 ladder, `backend=python`. Pass `--config-name complete_fast_v1`
for the `c` recurrence backend of [`FAST_TABLEBASE.md`](FAST_TABLEBASE.md),
which builds the production artifact.

The sweep is the only DTH value-bearing production workflow. Earlier
partial-solve artifacts are not accepted by production play.

## Play the canonical game

```powershell
# DTH is the default Hal policy.
uv run python -m terminal play

# Equivalent explicit spelling and the terminal renderer.
uv run python -m terminal play --hal-agent dth --tui
```

Arena always resolves the canonical leap-aware STL game. DTH supplies an exact
mixed strategy on its shared 1..60 action/state model; arena and the STL engine
own legal action masking, the possible Baku action 61, clocks, transitions, and
the frozen revival roll.

## Retired Rust kernel

Until 2026-10-23 an optional Rust kernel, `dth_complete_rs`, swept the v2
ladder's layers and returned the LP residue to the same Python code. We
retired it with the repository's Rust workspace. On the top 200 potential
layers it took 3.4 s against 92 s for the Python v2 path, and both wrote the
same `value.npy` and `solver_kind.npy` bytes. Production builds use the `c`
backend, and `src/dth_compact/`'s numba solver ran the same layers in 0.24 s
warm. Git history keeps the kernel. See
[`DTH_COMPLETE_PARITY.md`](DTH_COMPLETE_PARITY.md) for the fail-closed
contract of the v2 ladder.
