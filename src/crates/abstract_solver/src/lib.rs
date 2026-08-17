use numpy::{PyArray1, PyReadonlyArray1, PyReadwriteArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashSet;

const PARITY_CONTRACT_VERSION: &str = "abstract-packed-parity-v3";
pub const SOURCE_BUNDLE_DIGEST: &str = env!("SOURCE_BUNDLE_DIGEST");
pub const SOURCE_BUNDLE_DIGEST_ALGORITHM: &str = env!("SOURCE_BUNDLE_DIGEST_ALGORITHM");
const UNREACHABLE: u32 = u32::MAX;
const VALUE_EPS: f64 = 1e-12;
const PROBABILITY_EPS: f64 = 1e-12;

fn packed_state_count(cap: u32) -> Option<usize> {
    let cap = u64::from(cap);
    let ttd_size = cap.checked_add(1)?;
    let count = cap
        .checked_mul(ttd_size)?
        .checked_mul(cap)?
        .checked_mul(ttd_size)?;
    if count > u64::from(u32::MAX) + 1 {
        return None;
    }
    usize::try_from(count).ok()
}

fn validate_rules(cap: u32, action_size: u32, penalty: u32) -> PyResult<usize> {
    if cap == 0 || action_size == 0 || penalty == 0 || penalty >= cap {
        return Err(PyValueError::new_err("invalid packed rules parameters"));
    }
    if action_size != penalty {
        return Err(PyValueError::new_err(
            "action_size must equal the failed-check penalty for this parity contract",
        ));
    }
    packed_state_count(cap)
        .ok_or_else(|| PyValueError::new_err("packed rules exceed the uint32 state-index domain"))
}

fn validate_index(index: u32, physical: usize, name: &str) -> PyResult<usize> {
    let index = index as usize;
    if index >= physical {
        return Err(PyValueError::new_err(format!(
            "{name} must be in the packed physical state domain"
        )));
    }
    Ok(index)
}

fn validate_child_row(
    child: u32,
    parent_potential: u32,
    cap: u32,
    ordinal: &[u32],
    values: &[f64],
    dropper_wins: &[f64],
    checker_wins: &[f64],
) -> PyResult<(f64, f64, f64)> {
    let child_index = child as usize;
    let row = *ordinal
        .get(child_index)
        .ok_or_else(|| PyRuntimeError::new_err("child index out of range"))?;
    if row == UNREACHABLE {
        return Err(PyRuntimeError::new_err(
            "live child missing from reachable closure",
        ));
    }
    let row = row as usize;
    let value = *values
        .get(row)
        .ok_or_else(|| PyRuntimeError::new_err("child ordinal exceeds value arrays"))?;
    let dropper_win = *dropper_wins
        .get(row)
        .ok_or_else(|| PyRuntimeError::new_err("child ordinal exceeds value arrays"))?;
    let checker_win = *checker_wins
        .get(row)
        .ok_or_else(|| PyRuntimeError::new_err("child ordinal exceeds value arrays"))?;
    if !value.is_finite()
        || !(-1.0 - VALUE_EPS..=1.0 + VALUE_EPS).contains(&value)
        || !dropper_win.is_finite()
        || !(0.0..=1.0).contains(&dropper_win)
        || !checker_win.is_finite()
        || !(0.0..=1.0).contains(&checker_win)
    {
        return Err(PyRuntimeError::new_err(
            "child value or win probability is unsolved or outside its contract domain",
        ));
    }
    if (dropper_win + checker_win - 1.0).abs() > PROBABILITY_EPS {
        return Err(PyRuntimeError::new_err(
            "child win probabilities do not sum to one",
        ));
    }
    let child_fields = decode(child, cap);
    let child_potential = child_fields.0 + child_fields.1 + child_fields.2 + child_fields.3;
    if child_potential <= parent_potential {
        return Err(PyRuntimeError::new_err(
            "live child does not strictly increase packed potential",
        ));
    }
    Ok((value, dropper_win, checker_win))
}

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

fn revival_probability(prior_ttd: u32, dose: u32, cap: u32, penalty: u32) -> f64 {
    if dose >= cap || prior_ttd + dose > cap {
        return 0.0;
    }
    let pre_failure_st = dose - penalty;
    let survivable_st_span = cap - penalty;
    let dose_factor = 1.0 - pre_failure_st as f64 / survivable_st_span as f64;
    let ttd_factor = 0.75_f64.powf(prior_ttd as f64 / penalty as f64);
    (0.95 * dose_factor * ttd_factor).clamp(0.0, 1.0)
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
    let physical = validate_rules(cap, action_size, penalty)?;
    validate_index(index, physical, "index")?;
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
    let physical = validate_rules(cap, action_size, penalty)?;
    let queue = queue.as_slice_mut()?;
    let seen = seen.as_slice_mut()?;
    if queue.len() != physical {
        return Err(PyValueError::new_err(
            "reachability queue length must equal the packed physical state count",
        ));
    }
    let expected_seen = physical
        .checked_add(7)
        .ok_or_else(|| PyValueError::new_err("packed state count overflow"))?
        / 8;
    if seen.len() != expected_seen {
        return Err(PyValueError::new_err(
            "reachability bitset length disagrees with the packed physical state count",
        ));
    }
    if head > tail || tail > queue.len() {
        return Err(PyValueError::new_err("invalid reachability queue bounds"));
    }
    let stop = head.saturating_add(max_dequeues);
    while head < tail && head < stop {
        let index = queue[head];
        let index_usize = validate_index(index, physical, "queued index")?;
        let queued_byte = index_usize >> 3;
        let queued_mask = 1_u8 << (index_usize & 7);
        if seen[queued_byte] & queued_mask == 0 {
            return Err(PyValueError::new_err(
                "queued index is not present in the reachability bitset",
            ));
        }
        for child in live_successors(index, cap, action_size, penalty) {
            let byte_index = (child >> 3) as usize;
            let mask = 1_u8 << (child & 7);
            if seen[byte_index] & mask != 0 {
                continue;
            }
            if tail >= queue.len() {
                return Err(PyRuntimeError::new_err(
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
    let physical = validate_rules(cap, action_size, penalty)?;
    let indices = state_indices.as_slice()?;
    let ordinal = ordinal_by_index.as_slice()?;
    let values = values.as_slice()?;
    let dropper_wins = dropper_wins.as_slice()?;
    let checker_wins = checker_wins.as_slice()?;
    let size = action_size as usize;
    let cells = size
        .checked_mul(size)
        .ok_or_else(|| PyValueError::new_err("action matrix size overflow"))?;
    if ordinal.len() != physical {
        return Err(PyValueError::new_err(
            "ordinal_by_index length must equal the packed physical state count",
        ));
    }
    if values.len() != dropper_wins.len() || values.len() != checker_wins.len() {
        return Err(PyValueError::new_err(
            "value and win-probability arrays must have identical lengths",
        ));
    }
    if values.len() > u32::MAX as usize {
        return Err(PyValueError::new_err(
            "reachable row arrays exceed the uint32 ordinal domain",
        ));
    }
    if indices.len() > u32::MAX as usize {
        return Err(PyValueError::new_err(
            "backup chunk exceeds the uint32 mixed-position domain",
        ));
    }
    let mut unique_indices = HashSet::with_capacity(indices.len());
    for &index in indices {
        validate_index(index, physical, "state index")?;
        if !unique_indices.insert(index) {
            return Err(PyValueError::new_err(
                "backup chunk contains a duplicate state index",
            ));
        }
    }

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
        let parent_potential = checker_load + checker_ttd + dropper_load + dropper_ttd;
        let dose = checker_load + penalty;
        let revive = revival_probability(checker_ttd, dose, cap, penalty);
        let (failure_value, failure_dropper_win, failure_checker_win) = if revive > 0.0 {
            let child = encode(dropper_load, dropper_ttd, 0, checker_ttd + dose, cap);
            let (child_value, child_dropper_win, child_checker_win) = validate_child_row(
                child,
                parent_potential,
                cap,
                ordinal,
                values,
                dropper_wins,
                checker_wins,
            )?;
            (
                revive * -child_value + (1.0 - revive),
                revive * child_checker_win + (1.0 - revive),
                revive * child_dropper_win,
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
            let (child_value, child_dropper_win, child_checker_win) = validate_child_row(
                child,
                parent_potential,
                cap,
                ordinal,
                values,
                dropper_wins,
                checker_wins,
            )?;
            success.push((-child_value, child_checker_win, child_dropper_win));
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
    m.add("SOURCE_BUNDLE_DIGEST", SOURCE_BUNDLE_DIGEST)?;
    m.add(
        "SOURCE_BUNDLE_DIGEST_ALGORITHM",
        SOURCE_BUNDLE_DIGEST_ALGORITHM,
    )?;
    m.add_function(wrap_pyfunction!(live_successors_rs, m)?)?;
    m.add_function(wrap_pyfunction!(expand_reachability_chunk_rs, m)?)?;
    m.add_function(wrap_pyfunction!(backup_chunk_rs, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_source_bundle_digest_has_contract_shape() {
        assert_eq!(
            SOURCE_BUNDLE_DIGEST_ALGORITHM,
            "sha256-framed-source-bundle-v1"
        );
        assert_eq!(SOURCE_BUNDLE_DIGEST.len(), 64);
        assert!(
            SOURCE_BUNDLE_DIGEST
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        );
    }

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
    fn frozen_revival_model_matches_seconds_equivalent_buckets() {
        let ten_second = revival_probability(12, 18, 30, 6);
        let five_second = revival_probability(24, 36, 60, 12);
        assert!((ten_second - (0.95 * 0.5 * 0.75_f64.powi(2))).abs() < 1e-12);
        assert!((ten_second - five_second).abs() < 1e-12);
        assert_eq!(revival_probability(0, 6, 30, 6), 0.95);
    }

    #[test]
    fn packed_domain_validation_rejects_invalid_or_overflowing_rules() {
        assert!(validate_rules(60, 12, 12).is_ok());
        assert!(validate_rules(60, 11, 12).is_err());
        assert!(validate_rules(60, 12, 60).is_err());
        assert!(validate_rules(u32::MAX, 1, 1).is_err());
    }

    #[test]
    fn packed_index_validation_rejects_the_exclusive_upper_bound() {
        let physical = packed_state_count(4).unwrap();
        assert!(validate_index((physical - 1) as u32, physical, "index").is_ok());
        assert!(validate_index(physical as u32, physical, "index").is_err());
    }
}
