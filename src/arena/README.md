# Arena Project Instructions

`arena/` is the neutral executable surface for matches between peer projects.
It may import public interfaces from `stl`, `dth`, and `abstract`; peer projects
must not import one another in return.

- The STL engine remains the only canonical live-game referee.
- Providers return policy distributions; `PolicyDrivenAgent` alone masks and
  samples literal legal seconds.
- Projection adapters must document their approximation and may not alter the
  canonical game state or transitions.
- Keep generated artifacts in the owning project, never under `arena/`.
