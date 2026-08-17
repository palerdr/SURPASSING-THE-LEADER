# Generated-artifact regeneration

Generated tablebases, checkpoints, reports, and paper figure data are never
source authority and remain gitignored. A generated artifact is usable only
when its owning project can verify the artifact schema, the exact required
array set, file digests, canonical rule/config fingerprint, solver gate, and
the source-derived build fingerprint recorded when construction began.

## Compatibility rule

- Compute one canonical build fingerprint before creating or resuming output.
- Derive it from repository-relative source labels and file contents, frozen
  rules/configuration, dependency locks where relevant, array encodings, and
  solver tolerances.
- Store that same fingerprint in every progress checkpoint and final manifest.
  Do not recompute a new identity only at finalization.
- Reject missing, extra, self-described, or incompatible arrays and metadata.
- Source, schema, encoding, routing, or tolerance changes orphan an incomplete
  build. Rebuild rather than silently migrating exact values.

## Owning commands

Run commands from the repository root. Each project writes only beneath its
own ignored artifact/output roots.

```powershell
# Complete pure-DTH quotient artifact and independent audit.
uv run python -m dth complete
uv run python -m dth complete-audit

# Certified abstract tablebase.
uv run python -m abstract exact

# Paper figures, after the DTH artifact passes its production reader.
uv run python paper/generate_figure_data.py
uv run --with matplotlib --with seaborn --with pandas python paper/make_figures.py
```

The OCaml project retains its independent direct and packed exact-solver paths;
the Python DTH manifest remains the repository's canonical complete artifact
contract. Rust crates are accelerators behind their Python owners' artifact
formats and do not publish independent tablebases. The C++ artifact workflow
remains subtree-owned while that implementation and its build order are in
progress.

After regeneration, run the owning project tests and the repository-wide
validation commands in `AGENTS.md`. Exact artifacts must fail closed if copied
between incompatible source revisions or configurations.
