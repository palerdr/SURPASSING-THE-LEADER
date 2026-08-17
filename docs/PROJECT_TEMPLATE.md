# Project structure contract

Every entry in `PROJECTS.toml` is an independently understandable project or
shared implementation surface. Languages may use their native conventions;
the required template is a contract envelope, not a demand for identical
folder names.

## Required project information

Each project must make these facts discoverable through its `PROJECTS.toml`
entry and its project-level `README.md` together:

1. formulation rung and supported claim;
2. maturity (`solved`, `implemented-unsolved`, `reference`, `in-progress`,
   `opt-in`, or `play-surface`);
3. behavioral authority and explicit non-goals;
4. public library and CLI interfaces;
5. allowed dependency direction;
6. generated artifact/output ownership;
7. focused validation commands.

The machine-readable catalog is the required common envelope and owns the
path, rung, maturity, public interfaces, languages, and root validation
commands. A README may link to that catalog rather than repeat those fields;
it owns project-specific authority, non-goals, dependency details, artifact
paths, and operational guidance. Repository meta-tests own catalog
completeness; game mechanics remain authoritative only in the other canonical
documents under `docs/`.

## Code responsibilities

Use language-idiomatic modules for the following responsibilities and keep the
boundaries explicit:

- **domain/rules:** validated state and action types plus the sole local
  transition authority;
- **solver:** matrix construction, exact/approximate algorithms, and immutable
  certification gates;
- **tablebase:** schema, reader, and builder as separable responsibilities;
- **artifact I/O:** canonical fingerprints, strict manifests, atomic writes,
  and resume compatibility;
- **CLI:** argument parsing and orchestration, with testable return semantics;
- **tests:** public-boundary rejection, golden conformance vectors, solver
  certificates, artifact incompatibility, and interruption/resume behavior.

Compatibility facades may preserve existing imports while large modules are
split. Do not create a generic cross-project rules utility: peer projects stay
independent and consume shared conformance fixtures only through tests.

## Generated paths

- `artifacts/` contains certified exact data and its manifest.
- `checkpoints/` contains learned or resumable model state.
- `outputs/` contains reports, transcripts, figures, and other reproducible
  generated results.

All three are project-owned and gitignored. Source art and primary evidence are
not generated data.

## Configuration lifecycle

Tracked configurations are reproducibility inputs. Projects with a large
history should catalog configurations as current, archival, or dependency-only
without silently moving or deleting paths referenced by older experiments.
Inheritance graphs must be cycle-free and every referenced local preset or
checkpoint producer must be explicit.
