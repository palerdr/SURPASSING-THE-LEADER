# DTH workflows

Run commands from the repository root. Generated artifacts, reports, datasets,
and checkpoints remain under ignored paths owned by `src/dth/`.

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
uv run python -m arena play

# Equivalent explicit spelling and the terminal renderer.
uv run python -m arena play --hal-agent dth --tui
```

Arena always resolves the canonical leap-aware STL game. DTH supplies an exact
mixed strategy on its shared 1..60 action/state model; arena and the STL engine
own legal action masking, the possible Baku action 61, clocks, transitions, and
the frozen revival roll.

## Optional research workflows

```powershell
uv run python -m dth dataset --help
uv run python -m dth train --help
uv run python -m dth self-play --help
uv run python -m dth mcts-audit --help
```

These commands remain useful for policy compression, learned evaluation,
search comparisons, and empirical analysis. Configured outputs must state
whether a target is finite-horizon exact or an approximate play estimate. No
research output is a fallback for `CompleteDTHAgent` or a substitute for the
complete artifact. [`RESEARCH_CONFIGS.md`](RESEARCH_CONFIGS.md) classifies the
tracked presets and separates current entry configurations from reproducible
research lineages.

## Cross-backend validation

```powershell
cd src/crates/dth_complete
uv run maturin develop
cd ../../..
uv run python -m pytest src/dth/tests/test_complete_rust_parity.py -q
```

See [`DTH_COMPLETE_PARITY.md`](DTH_COMPLETE_PARITY.md) for the fail-closed
contract.
