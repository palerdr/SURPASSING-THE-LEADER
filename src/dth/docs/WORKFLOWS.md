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
potential layer, and writes `src/dth/artifacts/complete_full_v1/`. Set
`backend=python` or `backend=rust` to force a backend; `backend=auto` uses the
parity-gated Rust accelerator when installed.

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

## Cross-backend validation

```powershell
cd src/crates/dth_complete
uv run maturin develop
cd ../../..
uv run python -m pytest src/dth/tests/test_complete_rust_parity.py -q
```

See [`DTH_COMPLETE_PARITY.md`](DTH_COMPLETE_PARITY.md) for the fail-closed
contract.
