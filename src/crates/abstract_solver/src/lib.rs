use numpy::{PyArray1, PyReadonlyArray1, PyReadwriteArray1};
use pyo3::prelude::*;

const PARITY_CONTRACT_VERSION: &str = "abstract-packed-parity-v2";
const UNREACHABLE: u32 = u32::MAX;

fn encode(
    checker_load: u32,
    checker_ttd: u32,
    dropper_load: u32,
    dropper_ttd: u32,
    cap: u32,
) -> u32 {
    let ttd_size = cap + 1;
    (((checker_load * ttd_size + checker_ttd) * cap + dropper_load) * ttd_size) + dropper_ttd
}

fn decode(index: u32, cap: u32) -> (u32, u32, u32, u32) {
    let ttd_size = cap + 1;
    let dropper_ttd = index % ttd_size;
    let quotient = index / ttd_size;
    let dropper_load = quotient % cap;
    let quotient = quotient / cap;
    let checker_ttd = quotient % ttd_size;
    let checker_load = quotient / ttd_size;
    (checker_load, checker_ttd, dropper_load, dropper_ttd)
}

fn live_successors(index: u32, cap: u32, action_size: u32, penalty: u32) -> Vec<u32> {
    let (checker_load, checker_ttd, dropper_load, dropper_ttd) = decode(index, cap);
    let mut result = Vec::with_capacity(action_size as usize + 1);
    for squandered in 1..=action_size {
        let candidate = checker_load + squandered;
        if candidate < cap {
            result.push(encode(
                dropper_load,
                dropper_ttd,
                candidate,
                checker_ttd,
                cap,
            ));
        }
    }
    let dose = checker_load + penalty;
    if dose < cap && checker_ttd + dose <= cap {
        result.push(encode(
            dropper_load,
            dropper_ttd,
            0,
            checker_ttd + dose,
            cap,
        ));
    }
    result
}

fn revival_probability(
    prior_ttd: u32,
    dose: u32,
    cap: u32,
    penalty: u32,
    model_kind: u8,
    baseline: f64,
    dose_exponent: f64,
    half_life: f64,
    ttd_exponent: f64,
    ttd_decay_per_death_dose: f64,
    referee_decay: f64,
    referee_floor: f64,
) -> f64 {
    if dose >= cap || prior_ttd + dose > cap {
        return 0.0;
    }
    let (dose_factor, referee_factor) = match model_kind {
        0 => (1.0 - (dose as f64 / cap as f64).powf(dose_exponent), 1.0),
        1 => {
            let pre_failure_st = dose - penalty;
            let survivable_st_span = cap - penalty;
            let effective_deaths = prior_ttd as f64 / penalty as f64;
            (
                1.0 - pre_failure_st as f64 / survivable_st_span as f64,
                referee_floor.max(referee_decay.powf(effective_deaths)),
            )
        }
        2 => {
            let pre_failure_st = dose - penalty;
            let survivable_st_span = cap - penalty;
            (1.0 - pre_failure_st as f64 / survivable_st_span as f64, 1.0)
        }
        _ => return f64::NAN,
    };
    let ttd_factor = if model_kind == 2 {
        ttd_decay_per_death_dose.powf(prior_ttd as f64 / penalty as f64)
    } else {
        2.0_f64.powf(-((prior_ttd as f64 / half_life).powf(ttd_exponent)))
    };
    (baseline * dose_factor * ttd_factor * referee_factor).clamp(0.0, 1.0)
}

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-12
}

fn pure_saddle(matrix: &[f64], size: usize) -> Option<(usize, usize, f64)> {
    let mut row_minima = vec![f64::INFINITY; size];
    let mut column_maxima = vec![f64::NEG_INFINITY; size];
    for row in 0..size {
        for column in 0..size {
            let value = matrix[row * size + column];
            row_minima[row] = row_minima[row].min(value);
            column_maxima[column] = column_maxima[column].max(value);
        }
    }
    let lower = row_minima.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let upper = column_maxima.iter().copied().fold(f64::INFINITY, f64::min);
    if !close(lower, upper) {
        return None;
    }
    for row in 0..size {
        if !close(row_minima[row], lower) {
            continue;
        }
        for column in 0..size {
            if close(column_maxima[column], upper) && close(matrix[row * size + column], lower) {
                return Some((row, column, matrix[row * size + column]));
            }
        }
    }
    None
}

#[pyfunction]
fn live_successors_rs(index: u32, cap: u32, action_size: u32, penalty: u32) -> PyResult<Vec<u32>> {
    if cap == 0 || action_size == 0 || penalty == 0 || penalty >= cap {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "invalid packed rules parameters",
        ));
    }
    Ok(live_successors(index, cap, action_size, penalty))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn expand_reachability_chunk_rs(
    mut queue: PyReadwriteArray1<'_, u32>,
    mut seen: PyReadwriteArray1<'_, u8>,
    mut head: usize,
    mut tail: usize,
    max_dequeues: usize,
    cap: u32,
    action_size: u32,
    penalty: u32,
) -> PyResult<(usize, usize)> {
    let queue = queue.as_slice_mut()?;
    let seen = seen.as_slice_mut()?;
    if head > tail || tail > queue.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "invalid reachability queue bounds",
        ));
    }
    let stop = head.saturating_add(max_dequeues);
    while head < tail && head < stop {
        let index = queue[head];
        for child in live_successors(index, cap, action_size, penalty) {
            let byte_index = (child >> 3) as usize;
            let mask = 1_u8 << (child & 7);
            if seen[byte_index] & mask != 0 {
                continue;
            }
            if tail >= queue.len() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "reachability queue exceeded physical domain",
                ));
            }
            seen[byte_index] |= mask;
            queue[tail] = child;
            tail += 1;
        }
        head += 1;
    }
    Ok((head, tail))
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
#[pyfunction]
fn backup_chunk_rs<'py>(
    py: Python<'py>,
    state_indices: PyReadonlyArray1<'_, u32>,
    ordinal_by_index: PyReadonlyArray1<'_, u32>,
    values: PyReadonlyArray1<'_, f64>,
    dropper_wins: PyReadonlyArray1<'_, f64>,
    checker_wins: PyReadonlyArray1<'_, f64>,
    cap: u32,
    action_size: u32,
    penalty: u32,
    revival_model_kind: u8,
    baseline: f64,
    dose_exponent: f64,
    half_life: f64,
    ttd_exponent: f64,
    ttd_decay_per_death_dose: f64,
    referee_decay: f64,
    referee_floor: f64,
) -> PyResult<(
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let indices = state_indices.as_slice()?;
    let ordinal = ordinal_by_index.as_slice()?;
    let values = values.as_slice()?;
    let dropper_wins = dropper_wins.as_slice()?;
    let checker_wins = checker_wins.as_slice()?;
    let size = action_size as usize;
    let cells = size * size;

    let mut pure_mask = vec![0_u8; indices.len()];
    let mut pure_value = vec![f64::NAN; indices.len()];
    let mut pure_drop_action = vec![u8::MAX; indices.len()];
    let mut pure_check_action = vec![u8::MAX; indices.len()];
    let mut pure_dropper_win = vec![f64::NAN; indices.len()];
    let mut pure_checker_win = vec![f64::NAN; indices.len()];
    let mut mixed_positions = Vec::<u32>::new();
    let mut mixed_payoff = Vec::<f64>::new();
    let mut mixed_dropper_win = Vec::<f64>::new();
    let mut mixed_checker_win = Vec::<f64>::new();

    for (position, index) in indices.iter().copied().enumerate() {
        let (checker_load, checker_ttd, dropper_load, dropper_ttd) = decode(index, cap);
        let dose = checker_load + penalty;
        let revive = revival_probability(
            checker_ttd,
            dose,
            cap,
            penalty,
            revival_model_kind,
            baseline,
            dose_exponent,
            half_life,
            ttd_exponent,
            ttd_decay_per_death_dose,
            referee_decay,
            referee_floor,
        );
        let (failure_value, failure_dropper_win, failure_checker_win) = if revive > 0.0 {
            let child = encode(dropper_load, dropper_ttd, 0, checker_ttd + dose, cap);
            let row = *ordinal.get(child as usize).ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("child index out of range")
            })?;
            if row == UNREACHABLE {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "live child missing from reachable closure",
                ));
            }
            let row = row as usize;
            (
                revive * -values[row] + (1.0 - revive),
                revive * checker_wins[row] + (1.0 - revive),
                revive * dropper_wins[row],
            )
        } else {
            (1.0, 1.0, 0.0)
        };

        let mut success = Vec::<(f64, f64, f64)>::with_capacity(size);
        for squandered in 1..=action_size {
            let candidate = checker_load + squandered;
            if candidate >= cap {
                success.push((1.0, 1.0, 0.0));
                continue;
            }
            let child = encode(dropper_load, dropper_ttd, candidate, checker_ttd, cap);
            let row = ordinal[child as usize];
            if row == UNREACHABLE {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "live child missing from reachable closure",
                ));
            }
            let row = row as usize;
            success.push((-values[row], checker_wins[row], dropper_wins[row]));
        }

        let mut payoff = vec![0.0_f64; cells];
        let mut cell_dropper_win = vec![0.0_f64; cells];
        let mut cell_checker_win = vec![0.0_f64; cells];
        for drop_action in 1..=action_size {
            for check_action in 1..=action_size {
                let cell = (drop_action as usize - 1) * size + (check_action as usize - 1);
                let outcome = if check_action < drop_action {
                    (failure_value, failure_dropper_win, failure_checker_win)
                } else {
                    success[(check_action - drop_action) as usize]
                };
                payoff[cell] = outcome.0;
                cell_dropper_win[cell] = outcome.1;
                cell_checker_win[cell] = outcome.2;
            }
        }

        if let Some((drop_action, check_action, value)) = pure_saddle(&payoff, size) {
            let cell = drop_action * size + check_action;
            pure_mask[position] = 1;
            pure_value[position] = value;
            pure_drop_action[position] = drop_action as u8;
            pure_check_action[position] = check_action as u8;
            pure_dropper_win[position] = cell_dropper_win[cell];
            pure_checker_win[position] = cell_checker_win[cell];
        } else {
            mixed_positions.push(position as u32);
            mixed_payoff.extend_from_slice(&payoff);
            mixed_dropper_win.extend_from_slice(&cell_dropper_win);
            mixed_checker_win.extend_from_slice(&cell_checker_win);
        }
    }

    Ok((
        PyArray1::from_vec(py, pure_mask),
        PyArray1::from_vec(py, pure_value),
        PyArray1::from_vec(py, pure_drop_action),
        PyArray1::from_vec(py, pure_check_action),
        PyArray1::from_vec(py, pure_dropper_win),
        PyArray1::from_vec(py, pure_checker_win),
        PyArray1::from_vec(py, mixed_positions),
        PyArray1::from_vec(py, mixed_payoff),
        PyArray1::from_vec(py, mixed_dropper_win),
        PyArray1::from_vec(py, mixed_checker_win),
    ))
}

#[pymodule]
fn abstract_solver_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("PARITY_CONTRACT_VERSION", PARITY_CONTRACT_VERSION)?;
    m.add_function(wrap_pyfunction!(live_successors_rs, m)?)?;
    m.add_function(wrap_pyfunction!(expand_reachability_chunk_rs, m)?)?;
    m.add_function(wrap_pyfunction!(backup_chunk_rs, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_codec_round_trips_boundaries() {
        for fields in [(0, 0, 0, 0), (59, 60, 59, 60), (17, 23, 41, 9)] {
            let index = encode(fields.0, fields.1, fields.2, fields.3, 60);
            assert_eq!(decode(index, 60), fields);
        }
    }

    #[test]
    fn successors_strictly_increase_potential() {
        let index = encode(17, 23, 41, 9, 60);
        let potential = 17 + 23 + 41 + 9;
        for child in live_successors(index, 60, 12, 12) {
            let fields = decode(child, 60);
            assert!(fields.0 + fields.1 + fields.2 + fields.3 > potential);
        }
    }

    #[test]
    fn detects_matching_pennies_as_mixed() {
        assert!(pure_saddle(&[1.0, -1.0, -1.0, 1.0], 2).is_none());
        assert_eq!(pure_saddle(&[1.0, 1.0, 0.0, 0.0], 2), Some((0, 0, 1.0)));
    }

    #[test]
    fn unified_revival_model_matches_seconds_equivalent_buckets() {
        let ten_second =
            revival_probability(12, 18, 30, 6, 1, 0.8, 3.0, 12.0, 1.3, 0.75, 0.88, 0.4);
        let five_second =
            revival_probability(24, 36, 60, 12, 1, 0.8, 3.0, 24.0, 1.3, 0.75, 0.88, 0.4);
        assert!((ten_second - 0.15488).abs() < 1e-12);
        assert!((ten_second - five_second).abs() < 1e-12);
        assert_eq!(
            revival_probability(0, 6, 30, 6, 1, 0.8, 3.0, 12.0, 1.3, 0.75, 0.88, 0.4),
            0.8
        );
    }

    #[test]
    fn frozen_revival_model_matches_seconds_equivalent_buckets() {
        let ten_second =
            revival_probability(12, 18, 30, 6, 2, 0.95, 3.0, 12.0, 1.3, 0.75, 0.88, 0.4);
        let five_second =
            revival_probability(24, 36, 60, 12, 2, 0.95, 3.0, 24.0, 1.3, 0.75, 0.88, 0.4);
        assert!((ten_second - (0.95 * 0.5 * 0.75_f64.powi(2))).abs() < 1e-12);
        assert!((ten_second - five_second).abs() < 1e-12);
        assert_eq!(
            revival_probability(0, 6, 30, 6, 2, 0.95, 3.0, 12.0, 1.3, 0.75, 0.88, 0.4),
            0.95
        );
    }
}
