use crate::{
    game::{Game, JointAction, PlayerId, TerminalOutcome},
    transition::expand_joint_action,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UtilityBreakdown {
    pub value: f64,
    pub hal_win_probability: f64,
    pub baku_win_probability: f64,
    pub unresolved_probability: f64,
}

impl UtilityBreakdown {
    pub fn weighted_sum(parts: &[(f64, Self)]) -> Self {
        let mut combined = Self {
            value: 0.0,
            hal_win_probability: 0.0,
            baku_win_probability: 0.0,
            unresolved_probability: 0.0,
        };

        for &(branch_probability, breakdown) in parts {
            combined.value += branch_probability * breakdown.value;
            combined.hal_win_probability += branch_probability * breakdown.hal_win_probability;
            combined.baku_win_probability += branch_probability * breakdown.baku_win_probability;
            combined.unresolved_probability +=
                branch_probability * breakdown.unresolved_probability;
        }

        combined
    }
}

pub fn terminal_utility(outcome: TerminalOutcome, perspective: PlayerId) -> Option<f64> {
    match outcome {
        TerminalOutcome::Ongoing => None,
        TerminalOutcome::WonBy(winner) => Some(if winner == perspective { 1.0 } else { -1.0 }),
        TerminalOutcome::Draw => Some(0.0),
    }
}

fn terminal_breakdown(outcome: TerminalOutcome, perspective: PlayerId) -> UtilityBreakdown {
    let value = terminal_utility(outcome, perspective)
        .expect("terminal breakdown requires a terminal outcome");
    let (hal_win_probability, baku_win_probability) = match outcome {
        TerminalOutcome::WonBy(PlayerId::Hal) => (1.0, 0.0),
        TerminalOutcome::WonBy(PlayerId::Baku) => (0.0, 1.0),
        TerminalOutcome::Draw => (0.0, 0.0),
        TerminalOutcome::Ongoing => unreachable!("terminal outcome checked above"),
    };

    UtilityBreakdown {
        value,
        hal_win_probability,
        baku_win_probability,
        unresolved_probability: 0.0,
    }
}

fn unresolved_breakdown() -> UtilityBreakdown {
    UtilityBreakdown {
        value: 0.0,
        hal_win_probability: 0.0,
        baku_win_probability: 0.0,
        unresolved_probability: 1.0,
    }
}

pub fn evaluate_joint_action(
    state: &Game,
    action: JointAction,
    horizon: i32,
    perspective: PlayerId,
) -> Result<UtilityBreakdown, String> {
    let outcome = state.terminal_outcome();
    if outcome != TerminalOutcome::Ongoing {
        return Ok(terminal_breakdown(outcome, perspective));
    }

    if horizon <= 0 {
        return Ok(unresolved_breakdown());
    }

    if horizon != 1 {
        return Err(format!(
            "recursive horizon {horizon} is not implemented; supported horizons are 0 and 1"
        ));
    }

    let weighted_breakdowns: Vec<(f64, UtilityBreakdown)> = expand_joint_action(state, action)?
        .into_iter()
        .map(|branch| {
            let child_outcome = branch.child.terminal_outcome();
            let breakdown = if child_outcome == TerminalOutcome::Ongoing {
                unresolved_breakdown()
            } else {
                terminal_breakdown(child_outcome, perspective)
            };
            (branch.probability, breakdown)
        })
        .collect();

    Ok(UtilityBreakdown::weighted_sum(&weighted_breakdowns))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn action(drop_time: u32, check_time: u32) -> JointAction {
        JointAction {
            drop_time,
            check_time,
        }
    }

    #[test]
    fn terminal_utility_uses_requested_perspective() {
        assert_eq!(
            terminal_utility(TerminalOutcome::Ongoing, PlayerId::Hal),
            None
        );
        assert_eq!(
            terminal_utility(TerminalOutcome::WonBy(PlayerId::Hal), PlayerId::Hal),
            Some(1.0)
        );
        assert_eq!(
            terminal_utility(TerminalOutcome::WonBy(PlayerId::Hal), PlayerId::Baku),
            Some(-1.0)
        );
        assert_eq!(
            terminal_utility(TerminalOutcome::Draw, PlayerId::Hal),
            Some(0.0)
        );
    }

    #[test]
    fn weighted_sum_combines_each_breakdown_field() {
        let hal_win = UtilityBreakdown {
            value: 1.0,
            hal_win_probability: 1.0,
            baku_win_probability: 0.0,
            unresolved_probability: 0.0,
        };
        let baku_win = UtilityBreakdown {
            value: -1.0,
            hal_win_probability: 0.0,
            baku_win_probability: 1.0,
            unresolved_probability: 0.0,
        };

        let combined = UtilityBreakdown::weighted_sum(&[(0.25, hal_win), (0.75, baku_win)]);

        assert_eq!(combined.value, -0.5);
        assert_eq!(combined.hal_win_probability, 0.25);
        assert_eq!(combined.baku_win_probability, 0.75);
        assert_eq!(combined.unresolved_probability, 0.0);
    }

    #[test]
    fn terminal_state_takes_priority_over_horizon_cutoff() {
        let mut state = Game::new(0);
        state.resolve_half_round(60, 1, Some(false)).unwrap();

        let breakdown = evaluate_joint_action(&state, action(10, 10), 0, PlayerId::Hal).unwrap();

        assert_eq!(breakdown.value, 1.0);
        assert_eq!(breakdown.hal_win_probability, 1.0);
        assert_eq!(breakdown.unresolved_probability, 0.0);
    }

    #[test]
    fn nonpositive_horizon_is_unresolved() {
        let state = Game::new(0);

        for horizon in [0, -1] {
            let breakdown =
                evaluate_joint_action(&state, action(10, 10), horizon, PlayerId::Hal).unwrap();
            assert_eq!(breakdown.value, 0.0);
            assert_eq!(breakdown.unresolved_probability, 1.0);
        }
    }

    #[test]
    fn horizon_one_reports_nonterminal_child_as_unresolved() {
        let state = Game::new(0);

        let breakdown = evaluate_joint_action(&state, action(10, 10), 1, PlayerId::Hal).unwrap();

        assert_eq!(breakdown.value, 0.0);
        assert_eq!(breakdown.unresolved_probability, 1.0);
    }

    #[test]
    fn horizon_one_weights_terminal_and_unresolved_branches() {
        let state = Game::new(0);
        let branches = expand_joint_action(&state, action(60, 1)).unwrap();
        let expected_hal_win = branches
            .iter()
            .filter(|branch| branch.child.is_game_over())
            .map(|branch| branch.probability)
            .sum::<f64>();

        let breakdown = evaluate_joint_action(&state, action(60, 1), 1, PlayerId::Hal).unwrap();

        assert!((breakdown.value - expected_hal_win).abs() < 1e-12);
        assert!((breakdown.hal_win_probability - expected_hal_win).abs() < 1e-12);
        assert!((breakdown.unresolved_probability - (1.0 - expected_hal_win)).abs() < 1e-12);
    }

    #[test]
    fn recursive_horizon_is_rejected_until_implemented() {
        let state = Game::new(0);

        let error = evaluate_joint_action(&state, action(10, 10), 2, PlayerId::Hal).unwrap_err();

        assert!(error.contains("recursive horizon 2 is not implemented"));
    }

    #[test]
    fn invalid_action_error_is_propagated() {
        let state = Game::new(0);

        assert!(evaluate_joint_action(&state, action(0, 1), 1, PlayerId::Hal).is_err());
    }
}
