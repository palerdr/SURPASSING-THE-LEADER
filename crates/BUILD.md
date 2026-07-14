# Rust Solver Build Notes

This Rust solver port is a learning exercise first. The Python solver already
exists as the reference implementation, so agents should avoid racing ahead and
filling in large pieces of Rust solver code unless explicitly asked.

## Collaboration Preferences

- Prefer explanations, pseudocode, probability primitives, and math over full
  implementation.
- When proposing a next step, name the smallest game-theoretic concept being
  built and why it is next.
- Do not implement whole modules, solver towers, tablebases, MCTS, or training
  bridges without direct instruction.
- If code is requested, keep changes narrow and explain the invariant each test
  is protecting.
- Preserve the user's existing Rust representations unless they are incorrect
  or block the stated task.
- Treat uncommitted files as user work unless they were changed in the current
  turn.

## Concept Order

Build the Rust solver from small, inspectable game-theoretic pieces:

1. Matrix payoff representation:
   A dense row-major matrix `A`, where rows are one player's actions, columns
   are the opponent's actions, and `A[i,j]` is utility for the row player.

2. Probability vector primitives:
   Nonnegative vectors that sum to one. These represent mixed strategies.

3. Expected value of mixed strategies:
   For row strategy `p` and column strategy `q`,
   `EV(p, q) = p^T A q = sum_i sum_j p_i A[i,j] q_j`.

4. Best response:
   Given opponent strategy `q`, each row action has value `(A q)_i`.
   A row best response chooses an action with maximum value. Given row strategy
   `p`, each column action has value `(p^T A)_j`; the minimizing column player
   chooses a minimum.

5. Joint-action transition primitive:
   Given a public `Game` state and a `JointAction { drop, check }`,
   produce the outcome branches caused by that single cell of the matrix. This
   is the Rust mirror of Python's `evaluate_joint_action` path, but it should
   be built in smaller pieces:
   - apply the half-round with an explicit survival outcome,
   - identify terminal versus nonterminal child states,
   - expose death/survival as probability-weighted branches,
   - restore or clone state so probing one matrix cell cannot mutate siblings.

6. Joint-action value:
   Convert transition branches into one Hal-perspective expected value:
   `V(action) = sum_k P(branch_k | action) * V(child_k)`.
   At horizon 1, `V(child_k)` is terminal value or unresolved value. At deeper
   horizons, it comes from recursive solve.

7. Matrix-game minimax:
   Solve for a saddle point:
   `max_p min_j (p^T A)_j`, subject to `p_i >= 0` and `sum_i p_i = 1`.
   This can be developed as a standalone matrix primitive before or after
   joint-action evaluation, but it is not enough to mirror the recursive solver
   until matrix cells can be evaluated from real engine transitions.

8. Recursive finite-horizon solve:
   Build the current half-round matrix by evaluating each joint action. For
   nonterminal child states, recurse to the remaining horizon. For stochastic
   death/survival, combine child values by expectation.

9. Selective search:
   Restrict the matrix to a candidate action set, then compare against
   full-width exact solves as an audit rather than assuming the restriction is
   exact.

10. Matrix-game MCTS:
   Use sampled rollouts to estimate a Q-matrix at each public state, then solve
   the current matrix game at selection and at the root.

## Perfect Information Clarification

Surpassing The Leader has public state, but each half-round is simultaneous:
the dropper and checker choose seconds before either action is revealed. That
means the local decision is a zero-sum normal-form matrix game, not a simple
alternating-move max/min tree node.

CFR+ is useful as an approximate no-regret matrix solver, especially for
bounded runtime resolves. It is not needed because of hidden information. The
canonical exact object for certified labels and audits is matrix minimax,
preferably checked against the Python `solve_minimax` behavior.

## Testing Style

Prefer tests that teach one invariant at a time:

- probability vectors reject negative mass and normalize/sum correctly,
- expected value matches hand-computed `p^T A q`,
- pure best responses pick the correct row or column,
- matching pennies solves near `[0.5, 0.5]` with value `0`,
- dominated actions receive zero probability in exact minimax,
- one-row or one-column matrices collapse to the obvious pure solution,
- leap-window action dimensions preserve Baku-dropper `61` versus checker cap
  `60`.

## Agent Boundary

When a user asks "what should I build next?", inspect the current Rust kernel
first. Answer with the next smallest piece, the math object, pseudocode, and focused tests. Do not
write the Rust implementation unless the user explicitly asks for code.
