use crate::cfr::solve_cfr_plus_dense;

const CYLINDER_MAX: u32 = 300;
const FAILED_CHECK_PENALTY: u32 = 60;

pub(crate) fn half_round_payoff(drop_time: usize, check_time: usize, checker_cylinder: u32) -> f64 {
    if check_time < drop_time {
        let injected = (checker_cylinder + FAILED_CHECK_PENALTY).min(CYLINDER_MAX);
        -(injected as f64)
    } else {
        let st = (check_time - drop_time) as u32;
        if checker_cylinder + st >= CYLINDER_MAX {
            -(CYLINDER_MAX as f64)
        } else {
            -(st as f64)
        }
    }
}

pub(crate) fn assemble_payoff_matrix(
    n_drop: usize,
    n_check: usize,
    success_values: &[f64],
    fail_value: f64,
) -> Vec<f64> {
    assert!(success_values.len() >= n_check.saturating_sub(1));
    let mut payoff = vec![0.0; n_drop * n_check];
    for row in 0..n_drop {
        let drop = row + 1;
        for col in 0..n_check {
            let check = col + 1;
            let value = if check >= drop {
                let st = check - drop;
                success_values[st]
            } else {
                fail_value
            };
            payoff[row * n_check + col] = value;
        }
    }
    payoff
}

pub(crate) fn assemble_immediate_payoff_matrix(
    n_drop: usize,
    n_check: usize,
    checker_cylinder: u32,
) -> Vec<f64> {
    let mut payoff = vec![0.0; n_drop * n_check];

    for row in 0..n_drop {
        let drop_time = row + 1;

        for col in 0..n_check {
            let check_time = col + 1;

            payoff[row * n_check + col] =
                half_round_payoff(drop_time, check_time, checker_cylinder);
        }
    }

    payoff
}

//gives dropper probs, checker probs, and the E[payoff] given they adopt those strategies
pub(crate) fn solve_matrix_game(
    n_drop: usize,
    n_check: usize,
    success_values: &[f64],
    fail_value: f64,
    iterations: usize,
    avg_delay: usize,
    linear_weighting: bool,
) -> (Vec<f64>, Vec<f64>, f64) {
    let payoff = assemble_payoff_matrix(n_drop, n_check, success_values, fail_value);
    let (p, q, v) = solve_cfr_plus_dense(
        &payoff,
        n_drop,
        n_check,
        iterations,
        avg_delay,
        linear_weighting,
    );
    (p, q, v)
}

pub(crate) fn solve_half_round_matrix(
    drop_times: &[usize],
    check_times: &[usize],
    checker_cylinder: u32,
    iterations: usize,
    avg_delay: usize,
    linear_weighting: bool,
) -> (Vec<f64>, Vec<f64>, f64) {
    let n_drop = drop_times.len();
    let n_check = check_times.len();
    let mut payoff = vec![0.0; n_drop * n_check];

    for row in 0..n_drop {
        for col in 0..n_check {
            payoff[row * n_check + col] =
                half_round_payoff(drop_times[row], check_times[col], checker_cylinder);
        }
    }

    let (p, q, v) = solve_cfr_plus_dense(
        &payoff,
        n_drop,
        n_check,
        iterations,
        avg_delay,
        linear_weighting,
    );
    (p, q, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_round_payoff_has_zero_diagonal() {
        assert_eq!(half_round_payoff(10, 10, 0), -0.0);
    }

    #[test]
    fn half_round_payoff_failed_check_uses_penalty_and_cap() {
        assert_eq!(half_round_payoff(10, 9, 50), -110.0);
        assert_eq!(half_round_payoff(10, 9, 280), -300.0);
    }

    #[test]
    fn half_round_payoff_success_uses_elapsed_seconds() {
        assert_eq!(half_round_payoff(10, 13, 0), -3.0);
    }

    #[test]
    fn solve_half_round_matrix_uses_actual_action_values() {
        let drop_times = vec![10, 20];
        let check_times = vec![10, 13];

        let (p, q, v) = solve_half_round_matrix(&drop_times, &check_times, 0, 100, 0, true);

        assert_eq!(p.len(), 2);
        assert_eq!(q.len(), 2);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!((q.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(v.is_finite());
    }
}
