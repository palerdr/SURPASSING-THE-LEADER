use crate::cfr::solve_cfr_plus_dense;

const CYLINDER_MAX: u32 = 300;
const FAILED_CHECK_PENALTY: u32 = 60;

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
                let st = (check-drop).max(1) as f64;
                success_values[(st-1.0) as usize]
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

pub(crate) fn half_round_payoff(
    drop_time: usize,
    check_time: usize,
    checker_cylinder: u32,
) -> f64 {
    if check_time < drop_time {
        let injected = (checker_cylinder + FAILED_CHECK_PENALTY).min(CYLINDER_MAX);
        -(injected as f64)
    } else {
        let st = (check_time - drop_time).max(1) as u32;
        if checker_cylinder + st >= CYLINDER_MAX {
            -(CYLINDER_MAX as f64)
        } else {
            -(st as f64)
        }
    }
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
    let (p, q, v) = solve_cfr_plus_dense(&payoff, n_drop, n_check, iterations, avg_delay, linear_weighting);
    (p, q, v)
}