# Abstract Project Instructions

This subtree owns the exact, role-relative 10-second and 5-second bucket
abstractions. It is a standalone project, not an STL engine. Both formulations
are solved by exhaustively enumerating their full reachable acyclic graphs and
certifying every simultaneous matrix before any approximate method is
considered.

- Never import `stl` or `dth`.
- Keep actions as positive ordinal buckets; action zero is illegal and
  successful squandered time is inclusive in bucket units.
- Do not add leap seconds, route stages, private information, neural leaf
  values, MCTS, CFR, or training code without an explicit new contract.
- Generated files belong under `abstract/outputs`.
- Python is behavioral authority. A packed Rust hot loop must satisfy
  `docs/PACKED_TABLEBASE_PARITY.md` before it can become a default backend.

Run `uv run python -m pytest abstract/tests -q`.
