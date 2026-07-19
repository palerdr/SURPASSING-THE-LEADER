# Canonical enumerated toy

The toy project demonstrates the complete exact-solving pipeline on the
`bucket12_fixed50` abstraction. Actions are ordinal buckets 1..12; each bucket
represents five seconds, and successful squandered time is inclusive in bucket
units. The model has no leap second, route stages, hidden information, or
production checkpoint compatibility.

`toy.rules` owns transitions, `toy.exact` recursively builds complete stage
matrices, `toy.tablebase` enumerates reachable states, and `toy.matrix` certifies
the zero-sum LP solution. Artifacts include source/config digests and must not be
mixed with STL or DTH artifacts.

Run the deterministic example with:

```powershell
uv run python -m toy exact --ruleset bucket12_fixed50
uv run python -m toy targets --ruleset bucket12_fixed50
```

The acceptance boundary is complete state/action enumeration, normalized legal
role policies, finite values, deterministic manifests, and a saddle gap within
the configured LP tolerance.

