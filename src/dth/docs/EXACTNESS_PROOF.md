# DTH finite-horizon exactness proof

## Claim

For every represented DTH state and finite horizon `h`, `dth.solver.solve`
returns the value and role marginals of the declared simultaneous stochastic
zero-sum game, up to the numerical tolerance of the matrix LP.

## Base case

At horizon zero, terminal states return their fixed utility and live states
return the declared cutoff value. No action is silently selected.

## Inductive step

Assume every successor at horizon `h-1` is solved exactly. DTH has finite legal
action sets `{1, ..., 60}` for both roles. For every joint action, the transition
function enumerates the complete success/failure outcome and every revival
chance branch. Weighting the exact successor values constructs the exact stage
matrix. The zero-sum LP returns that matrix's minimax value and equilibrium
marginals. Therefore the result at horizon `h` is exact.

## Boundary of the proof

The proof covers the no-leap rules and state described in
`GAME_AND_SOLVER.md`. It does not establish an infinite-horizon solution, STL
private-information behavior, or correctness of learned leaf values. LP
residual and saddle-gap tests cover numerical certification.

