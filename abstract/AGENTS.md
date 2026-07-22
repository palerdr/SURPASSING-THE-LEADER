# Abstract Project Instructions

This subtree is the exact, role-relative 10-second abstraction. It is a
standalone project, not an STL engine. The canonical model is solved by
exhaustively enumerating its full reachable acyclic graph and certifying every
simultaneous matrix with LP before any approximate method is considered.

- Never import `stl` or `dth`.
- Keep actions as positive ordinal buckets; action zero is illegal and
  successful squandered time is inclusive in bucket units.
- Do not add leap seconds, route stages, private information, neural leaf
  values, MCTS, CFR, or training code without an explicit new contract.
- Generated files belong under `abstract/outputs`.

Run `uv run python -m pytest abstract/tests -q`.
