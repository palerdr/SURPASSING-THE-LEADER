//if the action value was better than my current strategy I feel regret for not having played it
//The goal of CFR+ is to clip -regret and minimize the regret over our strategy
fn regret(strategy_value: f64, action_value: f64) -> f64 {
    action_value - strategy_value
}

//play actions with +regret more often, clips negative regrets to 0, returns a uniform strategy if no regrets
pub(crate) fn regret_plus_strategy(regrets: &[f64]) -> Vec<f64> {
    let positives: Vec<f64> = regrets.iter().map(|r| r.max(0.0)).collect();
    let total: f64 = positives.iter().sum();
    if total > 1e-12 {
        positives.iter().map(|r| r / total).collect()
    } else {
        let n = regrets.len();
        vec![1.0 / n as f64; n]
    }
}

//computes the payoff against an opponent strategy column vector
//for EACH row p E[payoff | Hero samples from row p, Villain samples from column q]
pub(crate) fn row_action_values(
    payoff: &[f64],
    rows: usize,
    cols: usize,
    op_strategy_col: &[f64],
) -> Vec<f64> {
    assert_eq!(payoff.len(), rows * cols);
    assert_eq!(op_strategy_col.len(), cols);
    let mut values = vec![0.0; rows];
    for r in 0..rows {
        let mut total = 0.0;
        for c in 0..cols {
            total += payoff[r * cols + c] * op_strategy_col[c];
        }
        values[r] = total;
    }
    values
}

pub(crate) fn col_action_values(
    payoff: &[f64],
    rows: usize,
    cols: usize,
    op_strategy_row: &[f64],
) -> Vec<f64> {
    assert_eq!(payoff.len(), rows * cols);
    assert_eq!(op_strategy_row.len(), rows);
    let mut values = vec![0.0; cols];
    for c in 0..cols {
        let mut total = 0.0;
        for r in 0..rows {
            total += (-payoff[r * cols + c]) * op_strategy_row[r];
        }
        values[c] = total;
    }
    values
}

//computes the expected value of the current row strategy and the current action values
//E[payoff | Hero samples from row p, Villain samples from column q]
pub(crate) fn strategy_ev(action_values: &[f64], strategy: &[f64]) -> f64 {
    assert_eq!(strategy.len(), action_values.len());
    strategy
        .iter()
        .zip(action_values.iter())
        .map(|(p, v)| p * v)
        .sum()
}

pub(crate) fn update_regret_plus(
    regrets: &mut [f64],
    action_values: &[f64],
    current_value: f64,
) -> () {
    assert_eq!(regrets.len(), action_values.len());

    for i in 0..regrets.len() {
        let instant_regret = regret(current_value, action_values[i]);
        regrets[i] = (regrets[i] + instant_regret).max(0.0);
    }
}

pub(crate) fn cfr_iteration(
    payoff: &[f64],
    row_regret: &mut [f64],
    col_regret: &mut [f64],
    rows: usize,
    cols: usize,
) -> (Vec<f64>, Vec<f64>) {
    //turn regrets into a strategy for both players
    let p = regret_plus_strategy(row_regret);
    let q = regret_plus_strategy(col_regret);

    //compute row_values Vp = p @ A, compute expected value Vp @ p
    let vp = row_action_values(payoff, rows, cols, &q);
    let hero_ev = strategy_ev(&vp, &p);
    update_regret_plus(row_regret, &vp, hero_ev);
    let p_prime = regret_plus_strategy(row_regret);

    //compute col_values Vq = A @ q, compute expected value Vq @ q
    let vq = col_action_values(payoff, rows, cols, &p_prime);
    let villain_ev = strategy_ev(&vq, &q);
    update_regret_plus(col_regret, &vq, villain_ev);
    let q_prime = regret_plus_strategy(col_regret);

    //return the updated hero strategy, villain strategy
    (p_prime, q_prime)
}

pub(crate) fn average_strategy(strategy_sum: &[f64], regret: &[f64]) -> Vec<f64> {
    let total_sum: f64 = strategy_sum.iter().sum();
    //if total_sum is more than 0 we set the strategy to just the regret+ strategy
    if total_sum > 1e-12 {
        strategy_sum.iter().map(|s| s / total_sum).collect()
    } else {
        regret_plus_strategy(&regret)
    }
}

fn expected_payoff(payoff: &[f64], p: &[f64], q: &[f64], rows: usize, cols: usize) -> f64 {
    let mut value: f64 = 0.0;
    for row_idx in 0..rows {
        for col_idx in 0..cols {
            value += p[row_idx] * payoff[row_idx * cols + col_idx] * q[col_idx]
        }
    }
    value
}

pub(crate) fn solve_cfr_plus_dense(
    payoff: &[f64],
    rows: usize,
    cols: usize,
    iterations: usize,
    avg_delay: usize,
    linear_weighting: bool,
) -> (Vec<f64>, Vec<f64>, f64) {
    let mut row_regret = vec![0.0; rows];
    let mut col_regret = vec![0.0; cols];
    let mut row_strategy_sum: Vec<f64> = vec![0.0; rows];
    let mut col_strategy_sum: Vec<f64> = vec![0.0; cols];
    for t in 1..(iterations + 1) {
        //updates row and col strategy
        let (p_prime, q_prime) =
            cfr_iteration(payoff, &mut row_regret, &mut col_regret, rows, cols);
        if t > avg_delay {
            let weight = if linear_weighting { t as f64 } else { 1.0 };
            //accumulate the broadcasted strategy
            row_strategy_sum
                .iter_mut()
                .zip(p_prime.iter())
                .for_each(|(sum_val, &p_val)| {
                    *sum_val += weight * p_val;
                });
            col_strategy_sum
                .iter_mut()
                .zip(q_prime.iter())
                .for_each(|(sum_val, &q_val)| {
                    *sum_val += weight * q_val;
                });
        }
    }
    let p = average_strategy(&row_strategy_sum, &row_regret);
    let q = average_strategy(&col_strategy_sum, &col_regret);
    let value = expected_payoff(payoff, &p, &q, rows, cols);
    (p, q, value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_action_values_for_matching_pennies_uniform() {
        let payoff = vec![1.0, -1.0, -1.0, 1.0];
        let col_strategy = vec![0.5, 0.5];

        let values = row_action_values(&payoff, 2, 2, &col_strategy);

        assert_eq!(values, vec![0.0, 0.0]);
    }

    #[test]
    fn strategy_value_is_weighted_average() {
        let strategy = vec![0.25, 0.75];
        let action_values = vec![10.0, 2.0];

        let value = strategy_ev(&strategy, &action_values);

        assert!((value - 4.0).abs() < 1e-12);
    }
    #[test]
    fn update_regret_plus_clips_negative_totals() {
        let mut regrets = vec![0.2, 0.0];
        let action_values = vec![0.0, 1.5];

        update_regret_plus(&mut regrets, &action_values, 0.9);

        assert!((regrets[0] - 0.0).abs() < 1e-12);
        assert!((regrets[1] - 0.6).abs() < 1e-12);
    }
    #[test]
    fn col_action_values_are_column_player_payoffs() {
        let payoff = vec![3.0, -1.0, 0.0, 2.0];
        let row_strategy = vec![0.4, 0.6];

        let values = col_action_values(&payoff, 2, 2, &row_strategy);

        assert!((values[0] - -1.2).abs() < 1e-12);
        assert!((values[1] - -0.8).abs() < 1e-12);
    }

    #[test]
    fn regret_plus_strategy_normalizes_only_positive_regret() {
        let regrets = vec![5.0, -2.0, 3.0];

        let strategy = regret_plus_strategy(&regrets);

        assert!((strategy[0] - 5.0 / 8.0).abs() < 1e-12);
        assert!((strategy[1] - 0.0).abs() < 1e-12);
        assert!((strategy[2] - 3.0 / 8.0).abs() < 1e-12);
        assert!((strategy.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn regret_plus_strategy_uses_uniform_when_no_positive_regret() {
        let regrets = vec![-1.0, 0.0, -3.0, -7.0];

        let strategy = regret_plus_strategy(&regrets);

        assert_eq!(strategy.len(), 4);
        for p in strategy {
            assert!((p - 0.25).abs() < 1e-12);
        }
    }

    #[test]
    fn cfr_iteration_returns_row_strategy_after_row_regret_update() {
        let payoff = vec![1.0, 1.0, 0.0, 0.0];
        let mut row_regret = vec![0.0, 0.0];
        let mut col_regret = vec![0.0, 0.0];

        let (next_row_strategy, _) = cfr_iteration(&payoff, &mut row_regret, &mut col_regret, 2, 2);

        assert!((next_row_strategy[0] - 1.0).abs() < 1e-12);
        assert!((next_row_strategy[1] - 0.0).abs() < 1e-12);
        assert!(row_regret[0] > row_regret[1]);
    }

    #[test]
    fn cfr_iteration_updates_column_against_fresh_row_strategy() {
        let payoff = vec![2.0, 0.0, 0.0, 0.0];
        let mut row_regret = vec![0.0, 0.0];
        let mut col_regret = vec![0.0, 0.0];

        cfr_iteration(&payoff, &mut row_regret, &mut col_regret, 2, 2);

        assert!((col_regret[0] - 0.0).abs() < 1e-12);
        assert!((col_regret[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn solve_cfr_plus_dense_solves_matching_pennies() {
        let payoff = vec![1.0, -1.0, -1.0, 1.0];

        let (strategy, col_strategy, value) = solve_cfr_plus_dense(&payoff, 2, 2, 500, 50, true);

        assert_eq!(strategy.len(), 2);
        assert_eq!(col_strategy.len(), 2);
        assert!((strategy.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!((col_strategy.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!((strategy[0] - 0.5).abs() < 1e-3);
        assert!((strategy[1] - 0.5).abs() < 1e-3);
        assert!((col_strategy[0] - 0.5).abs() < 1e-3);
        assert!((col_strategy[1] - 0.5).abs() < 1e-3);
        assert!(value.abs() < 1e-3);
    }

    #[test]
    fn solve_cfr_plus_dense_prefers_dominating_row() {
        let payoff = vec![1.0, 1.0, 0.0, 0.0];

        let (strategy, col_strategy, value) = solve_cfr_plus_dense(&payoff, 2, 2, 100, 0, true);

        assert!(strategy[0] > 0.99);
        assert!(strategy[1] < 0.01);
        assert_eq!(col_strategy.len(), 2);
        assert!((col_strategy.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(value > 0.99);
    }

    #[test]
    fn solve_cfr_plus_dense_delay_past_iterations_returns_finite_strategy() {
        let payoff = vec![1.0, -1.0, -1.0, 1.0];

        let (strategy, col_strategy, value) = solve_cfr_plus_dense(&payoff, 2, 2, 2, 10, true);

        assert!(strategy.iter().all(|p| p.is_finite()));
        assert!(col_strategy.iter().all(|p| p.is_finite()));
        assert!(value.is_finite());
        assert!((strategy.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!((col_strategy.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }
}
