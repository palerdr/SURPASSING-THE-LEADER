# Exact 10-second role-relative abstraction

`abstract` is a standalone exact solver for one finite stochastic game. It has
no neural evaluator, MCTS, self-play, CFR, private information, leap rule, or
identity-specific parameters.

All solver quantities are ordinal ten-second buckets. The action set is
`1..6`; action `a` maps to `10*a` seconds only in metadata. Action zero is
illegal. A check succeeds iff `check >= drop`, and successful squandered time
is inclusive in bucket units:

```text
ST = check - drop + 1
```

Each live public state is role-relative:

```text
(checker_load, checker_ttd, dropper_load, dropper_ttd)
```

Vial loads range from `0..29`; TTD ranges from `0..30`. On every live
transition, the current roles swap. Values are always from the current
Dropper's perspective, so a live child's value is negated during backup.

Failed checks apply a six-bucket dose:

```text
dose = checker_load + 6
```

For eligible deaths, revival uses a shared no-CPR, no-physicality curve.  In
bucket units its 120-second TTD half-life is 12 and its tail exponent is 1.3:

```text
revive = 0.95 * (1 - (dose / 30)^3) * 2^(-((ttd / 12)^1.3))
```

The current dose of 30 buckets or more is fatal; `ttd + dose > 30` is fatal;
and equality at 30 remains eligible. A survival resets the old Checker's vial
and adds the dose to their TTD.

The potential

```text
checker_load + checker_ttd + dropper_load + dropper_ttd
```

strictly increases on every nonterminal edge. The reachable game graph is
therefore finite and acyclic. `abstract.exact` enumerates its complete closure
from the initial state and solves every `6 x 6` simultaneous matrix with LP to
terminal outcomes. No depth horizon or approximate leaf evaluation is used.

Run the complete tablebase build with:

```powershell
uv run python -m abstract exact --ruleset bucket6_ttd_curve95
```
