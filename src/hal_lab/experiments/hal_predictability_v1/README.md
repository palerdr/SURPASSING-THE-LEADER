# Hal predictability v1

This study asks how well public information predicts a Hal policy's second, and
whether that turns into score. Revision 2 ran on 2026-10-20 in pure DTH; the
hosted game plays canonical STL. `run.py` holds the `PROTOCOL` (seed `20261019`).

The registry builds `dth`, the hosted `perfect-hal --perfect-hal-model
translated-v1`, and `adaptive-dth` with the uniform prior. Each one plays 1,000
games against `UniformProvider` and against `ExactProvider`, with one provider
per series. Hal takes the Hal seat in even games. A permutation drawn from the
seed holds out 200 test games, and 10 % of the rest validate.

- **Departure:** the test-move mean of the floor NLL minus the floor entropy. The floor is the `CompleteDTHAgent` row with 1e-3 uniform mass. It needs no fit and uses a game-clustered bootstrap.
- **Control:** `fit` fails unless each exact-policy departure interval contains 0.
- **State-only:** counts per (ST // 30, TTD // 60 per side, role), Laplace 1, on the fit games.
- **History:** `log floor + z W + b`, fit by L-BFGS on the fit games. `z` holds the standardized floor log-probabilities, the last 4 seconds of each side, both players' running counts, and the bucket. Validation picks L2 from the grid [1e-4, 1e-3, 1e-2]. **Leak** is the floor loss minus the history loss.
- **Exploit:** a counter-player forecasts Hal from the `translated-v1__exact` state-only table and plays the pure best response in the exact stage matrix.

Revision 1 tuned Adam with early stopping and chose its rate after it saw test
numbers at three rates. Revision 2 replaces that model and its control.

```sh
for p in generate fit exploit report; do uv run python -m hal_lab.experiments.hal_predictability_v1.run $p; done
```

## Results

Nats per move on held-out games. `results.md` in the ignored
`src/hal_lab/outputs/hal_predictability_v1/` adds the leak table.

| Policy | Opponent | Moves (test) | Off-support | Departure [95% CI] | Floor | State-only | History | L2 | Top-1 floor / state / history |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| dth | uniform | 9509 (1843) | 0 | -0.0024 [-0.0816, +0.0792] | 2.8930 | 3.3913 | 3.1891 | 0.01 | 0.433 / 0.397 / 0.415 |
| dth | exact | 11237 (2251) | 0 | -0.0064 [-0.0809, +0.0670] | 2.7517 | 3.4866 | 2.9603 | 0.01 | 0.451 / 0.383 / 0.439 |
| dth | pooled | 20746 (4094) | 0 | -0.0046 [-0.0605, +0.0527] | 2.8153 | 3.4437 | 3.0633 | 0.01, 0.01 | 0.443 / 0.389 / 0.428 |
| translated-v1 | uniform | 5734 (1139) | 0 | +1.0323 [+0.9567, +1.1050] | 4.0362 | 1.2200 | 0.6284 | 0.01 | 0.106 / 0.766 / 0.809 |
| translated-v1 | exact | 11290 (2162) | 38 | +0.7044 [+0.6271, +0.7822] | 3.5651 | 1.9202 | 1.3047 | 0.01 | 0.287 / 0.623 / 0.663 |
| translated-v1 | pooled | 17024 (3301) | 38 | +0.8175 [+0.7611, +0.8747] | 3.7276 | 1.6786 | 1.0713 | 0.01, 0.01 | 0.224 / 0.673 / 0.714 |
| adaptive-uniform | uniform | 9410 (1814) | 1 | +0.0845 [+0.0054, +0.1769] | 3.0048 | 3.3861 | 3.2127 | 0.01 | 0.406 / 0.364 / 0.387 |
| adaptive-uniform | exact | 11365 (2235) | 11 | +0.0791 [+0.0068, +0.1583] | 2.8254 | 3.5081 | 3.0226 | 0.01 | 0.432 / 0.373 / 0.421 |
| adaptive-uniform | pooled | 20775 (4049) | 12 | +0.0815 [+0.0234, +0.1441] | 2.9058 | 3.4534 | 3.1078 | 0.01, 0.01 | 0.420 / 0.369 / 0.406 |

The control held. Against a stationary opponent, translated-v1 becomes a
near-deterministic function of the public state. The function depends on the
opponent: as dropper it opened with 60 in 500 of 500 games against uniform, and
as checker with 1 in 470 of 500 against exact. `response_temperature` 0 in
`perfect_hal.py` causes this, because Hal plays the pure best response to its
forecast. Adaptive departs by about 0.08 nats per move, with intervals above 0.
Validation picked the top of the L2 grid for every pair, and the history model
overfits.

| Player vs translated-v1 (table translated-v1__exact) | Wins | Win rate [Wilson 95%] |
| --- | ---: | --- |
| counter | 26 of 500 | 0.052 [0.036, 0.075] |
| exact | 252 of 500 | 0.504 [0.460, 0.548] |

The counter loses 474 of 500 games, because translated-v1 models it and punishes its deterministic play.
