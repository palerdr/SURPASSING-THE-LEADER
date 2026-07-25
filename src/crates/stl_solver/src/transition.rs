use crate::game::{Game, HalfRoundRecord, JointAction};

#[derive(Debug, Clone, Copy)]
pub struct Probability {
    p: f64,
    q: f64,
}

impl Probability {
    pub fn new(x: f64) -> Self {
        assert!(0.0 <= x, "probability cannot be negative");
        assert!(x <= 1.0, "probability cannot exceed 1");
        assert!(x.is_finite(), "probability must be finite");
        Self { p: x, q: 1.0 - x }
    }

    pub fn success(&self) -> f64 {
        self.p
    }

    pub fn failure(&self) -> f64 {
        self.q
    }

    pub fn with_failure(failure: f64) -> Self {
        Self::new(1.0 - failure)
    }
}

#[derive(Debug, Clone)]
pub struct Distribution {
    masses: Vec<(f64, f64)>,
}

impl Distribution {
    pub fn new(masses: Vec<(f64, f64)>) -> Self {
        for &(_, prob) in &masses {
            assert!(
                0.0 <= prob && prob <= 1.0,
                "mass probability must be between 0 and 1"
            );
            assert!(prob.is_finite(), "mass probability must be finite");
        }
        let total: f64 = masses.iter().map(|(_, prob)| prob).sum();
        assert!((total - 1.0).abs() <= 1e-10, "probabilities must sum to 1");
        Self { masses }
    }

    pub fn expectation(&self) -> f64 {
        self.masses.iter().map(|(value, prob)| value * prob).sum()
    }

    pub fn cmf(&self) -> Vec<(f64, f64)> {
        let mut cumulative = 0.0;
        self.masses
            .iter()
            .map(|(value, prob)| {
                cumulative += prob;
                (*value, cumulative)
            })
            .collect()
    }

    pub fn pmf(&self, value: f64) -> f64 {
        self.masses
            .iter()
            .find(|(atom, _)| atom == &value)
            .map(|(_, prob)| *prob)
            .unwrap_or(0.0)
    }
}

#[derive(Clone)]
pub struct TransitionBranch {
    pub probability: f64,
    pub child: Game,
    pub record: HalfRoundRecord,
}

pub fn resolve_on_fork(
    parent: &Game,
    action: JointAction,
    forced_outcome: Option<bool>,
) -> (Game, Result<HalfRoundRecord, String>) {
    let mut child = parent.fork_exact();
    let record = child.resolve_half_round(action.drop_time, action.check_time, forced_outcome);
    (child, record)
}

pub fn expand_joint_action(
    parent: &Game,
    action: JointAction,
) -> Result<Vec<TransitionBranch>, String> {
    let (probe, probe_result) = resolve_on_fork(parent, action, None);
    let probe_record = probe_result?;

    if probe_record.survived.is_none() {
        return Ok(vec![TransitionBranch {
            probability: 1.0,
            child: probe,
            record: probe_record,
        }]);
    }

    let p_survive = probe_record
        .survival_probability
        .ok_or_else(|| "death branch is missing survival probability".to_string())?;
    let p_die = 1.0 - p_survive;

    // Create deterministic children for each nonzero chance branch.
    let mut branches = Vec::with_capacity(2);

    if p_survive > 0.0 {
        let (child, record) = resolve_on_fork(parent, action, Some(true));
        branches.push(TransitionBranch {
            probability: p_survive,
            child,
            record: record?,
        });
    }

    if p_die > 0.0 {
        let (child, record) = resolve_on_fork(parent, action, Some(false));
        branches.push(TransitionBranch {
            probability: p_die,
            child,
            record: record?,
        });
    }

    Ok(branches)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_joint_action_returns_one_branch_when_no_death_occurs() {
        let parent = Game::new(0);
        let action = JointAction {
            drop_time: 10,
            check_time: 10,
        };

        let branches = expand_joint_action(&parent, action).unwrap();

        assert_eq!(branches.len(), 1);
        assert_eq!(branches[0].probability, 1.0);
        assert_eq!(branches[0].record.survived, None);
        assert!(!parent.is_game_over());
    }

    #[test]
    fn expand_joint_action_returns_survival_and_death_branches() {
        let parent = Game::new(0);
        let action = JointAction {
            drop_time: 60,
            check_time: 1,
        };

        let branches = expand_joint_action(&parent, action).unwrap();

        assert_eq!(branches.len(), 2);
        assert!(
            (branches
                .iter()
                .map(|branch| branch.probability)
                .sum::<f64>()
                - 1.0)
                .abs()
                < 1e-12
        );
        assert!(
            branches
                .iter()
                .any(|branch| branch.record.survived == Some(true))
        );
        assert!(
            branches
                .iter()
                .any(|branch| branch.record.survived == Some(false))
        );
        assert!(!parent.is_game_over());
    }

    #[test]
    fn expand_joint_action_propagates_invalid_actions() {
        let parent = Game::new(0);
        let action = JointAction {
            drop_time: 0,
            check_time: 1,
        };

        assert!(expand_joint_action(&parent, action).is_err());
    }
}
