use rand::{rngs::StdRng, Rng, RngCore, SeedableRng};

pub const GAME_START_HOUR: u8 = 8;
pub const SECONDS_PER_MINUTE: u32 = 60;
pub const MINUTES_PER_HOUR: u32 = 60;
pub const OPENING_START_CLOCK: u32 = 12 * 60;
pub const LS_WINDOW_START: u32 = 59 * 60;
pub const LS_WINDOW_END: u32 = 60 * 60;
pub const TURN_DURATION_NORMAL: u32 = 60;
pub const TURN_DURATION_LEAP: u32 = 61;
pub const FAILED_CHECK_PENALTY: u32 = 60;
pub const CYLINDER_MAX: u32 = 300;
pub const DEATH_PROCEDURE_OVERHEAD: u32 = 120;
pub const WITHIN_ROUND_OVERHEAD: u32 = 60;
pub const BASE_CURVE_K: u32 = 3;
pub const CARDIAC_DECAY: f64 = 0.85;
pub const REFEREE_DECAY: f64 = 0.88;
pub const REFEREE_FLOOR: f64 = 0.4;
pub const PHYSICALITY_HAL: f64 = 1.0;
pub const PHYSICALITY_BAKU: f64 = 0.94;
pub const ACTION_NORMAL_MAX: u8 = 60;
pub const ACTION_LEAP_MAX: u8 = 61;
pub const ACTION_SIZE: usize = (ACTION_LEAP_MAX as usize) + 1;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum PlayerId {
    Hal = 0,
    Baku = 1,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Role {
    Dropper,
    Checker,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum HalfRoundResult {
    CheckSuccess,
    CheckFailSurvived,
    CheckFailDied,
    CylinderOverflowSurvived,
    CylinderOverflowDied,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct HalfRoundRecord {
    pub round_index: u32,
    pub half: u8,
    pub dropper: PlayerId,
    pub checker: PlayerId,
    pub drop_time: u32,
    pub check_time: u32,
    pub turn_duration: u32,
    pub result: HalfRoundResult,
    pub st_gained: u32,
    pub death_duration: u32,
    pub survived: Option<bool>,
    pub game_clock_at_start: u32,
    pub survival_probability: Option<f64>,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Player {
    pub id: PlayerId,
    pub cylinder: u32,
    pub ttd: u32,
    pub alive: bool,
    pub physicality: f64,
}
impl Player {
    pub fn new(id: PlayerId) -> Self {
        let physicality = if id == PlayerId::Hal {
            PHYSICALITY_HAL
        } else {
            PHYSICALITY_BAKU
        };
        Self {
            id,
            cylinder: 0,
            ttd: 0,
            alive: true,
            physicality,
        }
    }
    pub fn safe_strategies_remaining(&self) -> u32 {
        ((CYLINDER_MAX - 1 - self.cylinder) / TURN_DURATION_NORMAL).max(0)
    }
    pub fn add_to_cylinder(&mut self, amount: u32) -> bool {
        self.cylinder += amount;
        self.cylinder >= CYLINDER_MAX
    }
    pub fn on_death(&mut self, death_duration: u32) -> () {
        self.ttd += death_duration;
        self.alive = false;
    }
    pub fn on_revival(&mut self) -> () {
        self.cylinder = 0;
        self.alive = true;
    }
    pub fn on_permanent_death(&mut self) -> () {
        self.alive = false;
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Referee {
    pub cprs_performed: u32,
}
impl Referee {
    pub fn new() -> Self {
        Self { cprs_performed: 0 }
    }

    pub fn compute_survival_probability(&self, player: &Player, death_duration: u32) -> f64 {
        if death_duration >= CYLINDER_MAX {
            return 0.0;
        }
        let death_curve = |t: u32| -> f64 {
            let t = t as f64;
            let max_t = CYLINDER_MAX as f64;
            (1.0 - (t / max_t).powf(BASE_CURVE_K as f64)).max(0.0)
        };
        let cardiac_modifier = |ttd: u32| -> f64 {
            CARDIAC_DECAY.powf(ttd as f64 / 60.0)
        };
        let referee_modifier = |cprs: u32| -> f64 {
            REFEREE_FLOOR.max(REFEREE_DECAY.powf(cprs as f64))
        };
        death_curve(death_duration) * cardiac_modifier(player.ttd) * referee_modifier(self.cprs_performed) * player.physicality
    }

    pub fn attempt_revival(
        &mut self,
        player: &Player,
        death_duration: u32,
        rng: Option<&mut dyn RngCore>,
    ) -> bool {
        let prob = self.compute_survival_probability(player, death_duration);
        let roll: f64 = match rng {
            Some(r) => {
                let sample = r.next_u64();
                (sample as f64) / (u64::MAX as f64)
            }
            None => {
                let mut fallback = rand::thread_rng();
                let sample = fallback.next_u64();
                (sample as f64) / (u64::MAX as f64)
            }
        };
        let survived = prob > roll;
        self.cprs_performed += 1;
        survived
    }
}

#[derive(Clone, Debug)]
pub struct HalfRoundAction {
    pub drop_time: u8,
    pub check_time: u8,
}

pub fn is_leap_second_turn(game_clock: u32) -> bool {
    (LS_WINDOW_START..=LS_WINDOW_END).contains(&game_clock)
}
pub fn turn_duration_for_clock(game_clock: u32) -> u32 {
    if is_leap_second_turn(game_clock) {
        TURN_DURATION_LEAP
    } else {
        TURN_DURATION_NORMAL
    }
}
pub fn legal_max_second(actor: PlayerId, role: Role, turn_duration: u32) -> u8 {
    let turn_is_leap = turn_duration >= TURN_DURATION_LEAP;
    if !turn_is_leap {
        return ACTION_NORMAL_MAX.min(turn_duration as u8);
    }
    match role {
        Role::Dropper if !matches!(actor, PlayerId::Hal) => ACTION_LEAP_MAX,
        _ => ACTION_NORMAL_MAX,
    }
}
pub fn legal_seconds(actor: PlayerId, role: Role, turn_duration: u32) -> Vec<u8> {
    let max = legal_max_second(actor, role, turn_duration);
    (1..=max).collect()
}
pub fn legal_mask(actor: PlayerId, role: Role, turn_duration: u32) -> Vec<bool> {
    let mut mask = vec![false; ACTION_SIZE];
    let max = legal_max_second(actor, role, turn_duration) as usize;
    if max >= 1 {
        for slot in mask.iter_mut().take(max + 1).skip(1) {
            *slot = true;
        }
    }
    mask
}

pub fn sample_second<R: Rng>(rng: &mut R, actor: PlayerId, role: Role, game_clock: u32) -> u8 {
    let max = legal_max_second(actor, role, turn_duration_for_clock(game_clock)) as u8;
    if max <= 1 {
        return 1;
    }
    rng.gen_range(1..=max)
}

pub struct Game {
    hal: Player,
    baku: Player,
    referee: Referee,
    first_dropper: PlayerId,
    game_clock: u32,
    half: u8,
    round_index: u32,
    game_over: bool,
    winner: Option<Player>,
    loser: Option<Player>,
    rng: StdRng,
}
impl Game {
    pub fn new(seed: u64) -> Self {
        let hal = Player::new(PlayerId::Hal);
        let baku = Player::new(PlayerId::Baku);
        let referee = Referee::new();
        Self {
            hal,
            baku,
            referee,
            first_dropper: PlayerId::Hal,
            game_clock: OPENING_START_CLOCK,
            half: 1,
            round_index: 0,
            game_over: false,
            winner: None,
            loser: None,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn seed(&mut self, seed: u64) {
        self.rng = StdRng::seed_from_u64(seed);
    }

    pub fn get_turn_duration(&self) -> u32 {
        if (LS_WINDOW_START <= self.game_clock) && (self.game_clock <= LS_WINDOW_END) {
            TURN_DURATION_LEAP
        } else {
            TURN_DURATION_NORMAL
        }
    }

    pub fn advance_clock(&mut self, elapsed: u32) {
        self.game_clock += elapsed;
    }

    pub fn is_game_over(&self) -> bool {
        self.game_over
    }

    pub fn is_leap_second_turn(&self) -> bool {
        is_leap_second_turn(self.game_clock)
    }

    pub fn snap_clock_to_next_minute(&mut self) {
        let gc = self.game_clock;
        if gc < LS_WINDOW_END {
            let mut snapped = ((gc / 60) + 1) * 60;
            if snapped == LS_WINDOW_END {
                snapped = 3601;
            }
            self.game_clock = snapped;
        } else if gc <= LS_WINDOW_END {
            self.game_clock = LS_WINDOW_END + 1;
        } else {
            let elapsed = gc - (LS_WINDOW_END + 1);
            self.game_clock = (LS_WINDOW_END + 1) + ((elapsed / 60) + 1) * 60;
        }
    }

    pub fn format_game_clock(&self) -> String {
        let gc = self.game_clock;
        let (hours, minutes, seconds) = if gc <= 3599 {
            let total_seconds = gc;
            let hours = 8 + total_seconds / 3600;
            let remainder = total_seconds % 3600;
            (hours, remainder / 60, remainder % 60)
        } else if gc == 3600 {
            (8, 59, 60)
        } else {
            let total_seconds = gc - 1;
            let hours = 8 + total_seconds / 3600;
            let remainder = total_seconds % 3600;
            (hours, remainder / 60, remainder % 60)
        };
        format!("{hours}:{minutes:02}:{seconds:02} AM")
    }

    pub fn validate_drop_time(
        &self,
        drop_time: u32,
        turn_duration: u32,
        actor: Option<PlayerId>,
    ) -> Result<(), String> {
        if actor.is_none() {
            if !(1..=turn_duration).contains(&drop_time) {
                return Err(format!(
                    "drop_time must be in [1, {}], got {}",
                    turn_duration, drop_time
                ));
            }
            return Ok(());
        }
        let actor = actor.unwrap_or(PlayerId::Hal);
        let max_second = legal_max_second(actor, Role::Dropper, turn_duration);
        if !(1..=(max_second as u32)).contains(&drop_time) {
            return Err(format!(
                "illegal action second={} for actor={:?} role=dropper; legal range is [1, {}]",
                drop_time, actor, max_second
            ));
        }
        Ok(())
    }

    pub fn validate_check_time(
        &self,
        check_time: u32,
        turn_duration: u32,
        actor: Option<PlayerId>,
    ) -> Result<(), String> {
        if actor.is_none() {
            let max_second = TURN_DURATION_NORMAL.min(turn_duration);
            if !(1..=max_second).contains(&check_time) {
                return Err(format!(
                    "check_time must be in [1, {}], got {}",
                    max_second, check_time
                ));
            }
            return Ok(());
        }
        let actor = actor.unwrap_or(PlayerId::Hal);
        let max_second = legal_max_second(actor, Role::Checker, turn_duration);
        if !(1..=(max_second as u32)).contains(&check_time) {
            return Err(format!(
                "illegal action second={} for actor={:?} role=checker; legal range is [1, {}]",
                check_time, actor, max_second
            ));
        }
        Ok(())
    }

    pub fn get_roles_for_half(&mut self) -> (&mut Player, &mut Player) {
        if self.half == 1 {
            if self.first_dropper == PlayerId::Hal {
                (&mut self.hal, &mut self.baku)
            } else {
                (&mut self.baku, &mut self.hal)
            }
        } else if self.first_dropper == PlayerId::Hal {
            (&mut self.baku, &mut self.hal)
        } else {
            (&mut self.hal, &mut self.baku)
        }
    }

    pub fn play_half_round(
        &mut self,
        drop_time: u32,
        check_time: u32,
    ) -> Result<HalfRoundRecord, String> {
        self.resolve_half_round(drop_time, check_time, None)
    }

    pub fn play_round(
        &mut self,
        half1_drop: u32,
        half1_check: u32,
        half2_drop: u32,
        half2_check: u32,
    ) -> Result<Vec<HalfRoundRecord>, String> {
        let mut records = Vec::with_capacity(2);
        records.push(self.play_half_round(half1_drop, half1_check)?);
        if !self.game_over {
            records.push(self.play_half_round(half2_drop, half2_check)?);
        }
        Ok(records)
    }

    pub fn resolve_half_round(
        &mut self,
        drop_time: u32,
        check_time: u32,
        survived_outcome: Option<bool>,
    ) -> Result<HalfRoundRecord, String> {
        if self.game_over {
            return Err("Game is already over".to_string());
        }
        let clock_at_start = self.game_clock;
        let turn_duration = self.get_turn_duration();
        let (dropper_id, checker_id) = if self.half == 1 {
            if self.first_dropper == PlayerId::Hal {
                (PlayerId::Hal, PlayerId::Baku)
            } else {
                (PlayerId::Baku, PlayerId::Hal)
            }
        } else if self.first_dropper == PlayerId::Hal {
            (PlayerId::Baku, PlayerId::Hal)
        } else {
            (PlayerId::Hal, PlayerId::Baku)
        };

        self.validate_drop_time(drop_time, turn_duration, Some(dropper_id))?;
        self.validate_check_time(check_time, turn_duration, Some(checker_id))?;
        let (mut death_occurred, mut death_duration, mut survived, mut survival_probability, mut result, mut st_gained) = (
            false,
            0,
            None,
            None,
            HalfRoundResult::CheckSuccess,
            0,
        );
        let mut checker_snapshot = None;
        {
            let (_dropper, checker) = self.get_roles_for_half();
            let success = check_time >= drop_time;
            if success {
                st_gained = check_time - drop_time;
                let overflow = checker.add_to_cylinder(st_gained);
                if overflow {
                    death_occurred = true;
                    death_duration = checker.cylinder.min(CYLINDER_MAX);
                    checker_snapshot = Some(*checker);
                }
            } else {
                let _ = checker.add_to_cylinder(FAILED_CHECK_PENALTY);
                death_occurred = true;
                death_duration = checker.cylinder.min(CYLINDER_MAX);
                checker_snapshot = Some(*checker);
            }
        }

        if death_occurred {
            let snapshot = checker_snapshot.expect("checker snapshot must exist when death occurs");
            survival_probability = Some(self.referee.compute_survival_probability(&snapshot, death_duration));
            let did_survive = match survived_outcome {
                Some(value) => {
                    self.referee.cprs_performed += 1;
                    value
                }
                None => self.referee.attempt_revival(&snapshot, death_duration, Some(&mut self.rng)),
            };
            survived = Some(did_survive);
            let (dropper, checker) = self.get_roles_for_half();
            checker.on_death(death_duration);
            if did_survive {
                checker.on_revival();
                result = if check_time >= drop_time {
                    HalfRoundResult::CylinderOverflowSurvived
                } else {
                    HalfRoundResult::CheckFailSurvived
                };
            } else {
                checker.on_permanent_death();
                self.game_over = true;
                self.winner = Some(*dropper);
                self.loser = Some(*checker);
                result = if check_time >= drop_time {
                    HalfRoundResult::CylinderOverflowDied
                } else {
                    HalfRoundResult::CheckFailDied
                };
            }
        }
        self.advance_clock(turn_duration);
        if death_occurred {
            self.advance_clock(death_duration + DEATH_PROCEDURE_OVERHEAD);
        }
        let record = HalfRoundRecord {
            round_index: self.round_index,
            half: self.half,
            dropper: dropper_id,
            checker: checker_id,
            drop_time,
            check_time,
            turn_duration,
            result,
            st_gained,
            death_duration: if death_occurred { death_duration } else { 0 },
            survived,
            game_clock_at_start: clock_at_start,
            survival_probability,
        };
        if !self.game_over {
            if self.half == 1 {
                self.advance_clock(WITHIN_ROUND_OVERHEAD);
                self.half = 2;
            } else {
                self.snap_clock_to_next_minute();
                self.half = 1;
                self.round_index += 1;
            }
        }
        Ok(record)
    }
}
