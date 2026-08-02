//! Rayon-parallel complete-tablebase kernel for the DTH packed-quotient sweep.
//!
//! Behavioral authority is `src/dth/complete_tablebase.py`; this kernel mirrors
//! its pinned arithmetic operation for operation (no FMA, sequential
//! reductions, lowest-index tie-breaks) so that class values and solver
//! routing are reproduced bit for bit.  The contract is
//! `src/dth/docs/DTH_COMPLETE_PARITY.md`; this module performs no transcendental
//! arithmetic — every revival probability arrives precomputed from Python.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadwriteArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

pub const PARITY_CONTRACT_VERSION: &str = "dth-complete-parity-v1";
const POLICY_MASS_EPS: f64 = 1e-9;
const PIVOT_EPS: f64 = 1e-12;
const ACTIONS: usize = 60;

/// Raw shared view of the value/kind stores.  Safe because every class index
/// is written by exactly one work item and children live in strictly higher
/// (already completed, never concurrently written) potential layers.
struct SharedPtr<T>(*mut T);
unsafe impl<T> Send for SharedPtr<T> {}
unsafe impl<T> Sync for SharedPtr<T> {}

#[inline]
fn matrix_cell(success: &[f64; ACTIONS], failed: f64, drop: usize, check: usize) -> f64 {
    if check >= drop {
        success[check - drop]
    } else {
        failed
    }
}

/// O(60) pure-saddle reductions; returns `(gap, maximin, minimax)`.
fn toeplitz_saddle(success: &[f64; ACTIONS], failed: f64) -> (f64, f64, f64) {
    let mut prefix_min = [0.0_f64; ACTIONS];
    let mut running = f64::INFINITY;
    for index in 0..ACTIONS {
        if success[index] < running {
            running = success[index];
        }
        prefix_min[index] = running;
    }
    let mut maximin = f64::NEG_INFINITY;
    for drop in 0..ACTIONS {
        let mut row_min = prefix_min[ACTIONS - 1 - drop];
        if drop > 0 && failed < row_min {
            row_min = failed;
        }
        if row_min > maximin {
            maximin = row_min;
        }
    }
    let mut prefix_max = [0.0_f64; ACTIONS];
    let mut running = f64::NEG_INFINITY;
    for index in 0..ACTIONS {
        if success[index] > running {
            running = success[index];
        }
        prefix_max[index] = running;
    }
    let mut minimax = f64::INFINITY;
    for check in 0..ACTIONS {
        let mut col_max = prefix_max[check];
        if check < ACTIONS - 1 && failed > col_max {
            col_max = failed;
        }
        if col_max < minimax {
            minimax = col_max;
        }
    }
    (minimax - maximin, maximin, minimax)
}

/// Gaussian elimination with partial pivoting, pinned to the Python authority:
/// first-maximum pivots, explicit zeroing of eliminated cells, elementwise
/// `a[r][x] -= factor * a[p][x]` (no FMA), sequential back-substitution.
fn solve_linear_pinned(a: &mut [f64], b: &mut [f64], n: usize, solution: &mut [f64]) -> bool {
    for column in 0..n {
        let mut pivot = column;
        let mut best = a[column * n + column].abs();
        for row in column + 1..n {
            let magnitude = a[row * n + column].abs();
            if magnitude > best {
                best = magnitude;
                pivot = row;
            }
        }
        if best < PIVOT_EPS {
            return false;
        }
        if pivot != column {
            for cell in 0..n {
                a.swap(column * n + cell, pivot * n + cell);
            }
            b.swap(column, pivot);
        }
        for row in column + 1..n {
            let factor = a[row * n + column] / a[column * n + column];
            if factor != 0.0 {
                a[row * n + column] = 0.0;
                for cell in column + 1..n {
                    a[row * n + cell] -= factor * a[column * n + cell];
                }
                b[row] -= factor * b[column];
            }
        }
    }
    for row in (0..n).rev() {
        let mut accumulated = b[row];
        for column in row + 1..n {
            accumulated -= a[row * n + column] * solution[column];
        }
        solution[row] = accumulated / a[row * n + row];
    }
    true
}

struct SupportSolution {
    value: f64,
    drop_policy: [f64; ACTIONS],
    check_policy: [f64; ACTIONS],
}

/// The square-support equalizer solve plus full-matrix certificate; the
/// `k = 60` case is the full-support structured solve.
fn attempt_support(
    success: &[f64; ACTIONS],
    failed: f64,
    rows: &[usize],
    cols: &[usize],
    tolerance: f64,
) -> Option<SupportSolution> {
    let k = rows.len().min(cols.len());
    if k == 0 {
        return None;
    }
    let rows = &rows[..k];
    let cols = &cols[..k];
    let n = k + 1;
    let mut a = vec![0.0_f64; n * n];
    let mut b = vec![0.0_f64; n];
    let mut check_solution = vec![0.0_f64; n];
    for i in 0..k {
        for j in 0..k {
            a[i * n + j] = matrix_cell(success, failed, rows[i], cols[j]);
        }
        a[i * n + k] = -1.0;
    }
    for j in 0..k {
        a[k * n + j] = 1.0;
    }
    b[k] = 1.0;
    if !solve_linear_pinned(&mut a, &mut b, n, &mut check_solution) {
        return None;
    }
    let mut a = vec![0.0_f64; n * n];
    let mut b = vec![0.0_f64; n];
    let mut drop_solution = vec![0.0_f64; n];
    for i in 0..k {
        for j in 0..k {
            a[i * n + j] = matrix_cell(success, failed, rows[j], cols[i]);
        }
        a[i * n + k] = -1.0;
    }
    for j in 0..k {
        a[k * n + j] = 1.0;
    }
    b[k] = 1.0;
    if !solve_linear_pinned(&mut a, &mut b, n, &mut drop_solution) {
        return None;
    }
    for index in 0..k {
        if check_solution[index] < -PIVOT_EPS || drop_solution[index] < -PIVOT_EPS {
            return None;
        }
    }
    let mut check_total = 0.0_f64;
    for index in 0..k {
        if check_solution[index] < 0.0 {
            check_solution[index] = 0.0;
        }
        check_total += check_solution[index];
    }
    let mut drop_total = 0.0_f64;
    for index in 0..k {
        if drop_solution[index] < 0.0 {
            drop_solution[index] = 0.0;
        }
        drop_total += drop_solution[index];
    }
    if check_total <= 0.0 || drop_total <= 0.0 {
        return None;
    }
    let mut drop_policy = [0.0_f64; ACTIONS];
    let mut check_policy = [0.0_f64; ACTIONS];
    for index in 0..k {
        drop_policy[rows[index]] = drop_solution[index] / drop_total;
        check_policy[cols[index]] = check_solution[index] / check_total;
    }
    let mut upper = f64::NEG_INFINITY;
    for drop in 0..ACTIONS {
        let mut payoff = 0.0_f64;
        for j in 0..k {
            payoff += check_policy[cols[j]] * matrix_cell(success, failed, drop, cols[j]);
        }
        if payoff > upper {
            upper = payoff;
        }
    }
    let mut lower = f64::INFINITY;
    for check in 0..ACTIONS {
        let mut payoff = 0.0_f64;
        for i in 0..k {
            payoff += drop_policy[rows[i]] * matrix_cell(success, failed, rows[i], check);
        }
        if payoff < lower {
            lower = payoff;
        }
    }
    let gap = upper - lower;
    if (if gap > 0.0 { gap } else { 0.0 }) > tolerance {
        return None;
    }
    Some(SupportSolution {
        value: (lower + upper) / 2.0,
        drop_policy,
        check_policy,
    })
}

/// Pinned support extraction: threshold, top-mass trim with lowest-index
/// tie-breaks, ascending order.
fn support_of_policy(policy: &[f64; ACTIONS], max_support: usize) -> Vec<i32> {
    let mut indices: Vec<usize> = (0..ACTIONS)
        .filter(|&index| policy[index] > POLICY_MASS_EPS)
        .collect();
    if indices.len() > max_support {
        indices.sort_by(|&x, &y| {
            policy[y]
                .partial_cmp(&policy[x])
                .expect("finite policy mass")
                .then(x.cmp(&y))
        });
        indices.truncate(max_support);
        indices.sort_unstable();
    }
    indices.into_iter().map(|index| index as i32).collect()
}

#[derive(Default)]
struct ItemOutput {
    residue_classes: Vec<u64>,
    residue_success: Vec<f64>,
    residue_failed: Vec<f64>,
    hit_classes: Vec<u64>,
    hit_rows: Vec<i32>,
    hit_cols: Vec<i32>,
    pure: u64,
    warm_hits: u64,
    full_hits: u64,
    warm_attempts: u64,
    missing_child: bool,
}

struct LayerInputs {
    work_items: Vec<u64>,
    profile_pool: Vec<u32>,
    success_child: Vec<i32>,
    failure_child: Vec<i32>,
    revival: Vec<f64>,
    profile_count: usize,
    guess_classes: Vec<u64>,
    guess_rows: Vec<i32>,
    guess_cols: Vec<i32>,
    tolerance: f64,
    max_support: usize,
    warm_start: bool,
}

fn guess_lookup<'a>(inputs: &'a LayerInputs, class_id: u64) -> Option<(Vec<usize>, Vec<usize>)> {
    let index = inputs.guess_classes.binary_search(&class_id).ok()?;
    let start = index * inputs.max_support;
    let end = start + inputs.max_support;
    let rows: Vec<usize> = inputs.guess_rows[start..end]
        .iter()
        .filter(|&&v| v >= 0)
        .map(|&v| v as usize)
        .collect();
    let cols: Vec<usize> = inputs.guess_cols[start..end]
        .iter()
        .filter(|&&v| v >= 0)
        .map(|&v| v as usize)
        .collect();
    Some((rows, cols))
}

fn solve_item(
    inputs: &LayerInputs,
    item: usize,
    value: &SharedPtr<f64>,
    kind: &SharedPtr<u8>,
) -> ItemOutput {
    let count = inputs.profile_count;
    let checker_offset = inputs.work_items[item * 4] as usize;
    let checker_len = inputs.work_items[item * 4 + 1] as usize;
    let dropper_offset = inputs.work_items[item * 4 + 2] as usize;
    let dropper_len = inputs.work_items[item * 4 + 3] as usize;
    let full: Vec<usize> = (0..ACTIONS).collect();
    let mut output = ItemOutput::default();

    for dropper_slot in 0..dropper_len {
        let dropper = inputs.profile_pool[dropper_offset + dropper_slot] as usize;
        for checker_slot in 0..checker_len {
            let checker = inputs.profile_pool[checker_offset + checker_slot] as usize;
            let class_id = (checker * count + dropper) as u64;

            let mut success = [0.0_f64; ACTIONS];
            for lag in 0..ACTIONS {
                let child = inputs.success_child[checker * ACTIONS + lag];
                if child < 0 {
                    success[lag] = 1.0;
                } else {
                    let stored = unsafe { *value.0.add(dropper * count + child as usize) };
                    if !stored.is_finite() {
                        output.missing_child = true;
                        return output;
                    }
                    success[lag] = -stored;
                }
            }
            let failure = inputs.failure_child[checker];
            let failed = if failure < 0 {
                1.0
            } else {
                let stored = unsafe { *value.0.add(dropper * count + failure as usize) };
                if !stored.is_finite() {
                    output.missing_child = true;
                    return output;
                }
                let revive = inputs.revival[checker];
                revive * (-stored) + (1.0 - revive)
            };

            let (gap, maximin, minimax) = toeplitz_saddle(&success, failed);
            if gap <= inputs.tolerance {
                unsafe {
                    *value.0.add(class_id as usize) = (maximin + minimax) / 2.0;
                    *kind.0.add(class_id as usize) = 0;
                }
                output.pure += 1;
                continue;
            }

            let mut solution: Option<SupportSolution> = None;
            let mut warm = false;
            if inputs.warm_start {
                let checker_shift = inputs.success_child[checker * ACTIONS];
                let dropper_shift = inputs.success_child[dropper * ACTIONS];
                let neighbours = [
                    (checker_shift >= 0).then(|| (checker_shift as usize * count + dropper) as u64),
                    (dropper_shift >= 0).then(|| (checker * count + dropper_shift as usize) as u64),
                ];
                for neighbour in neighbours.into_iter().flatten() {
                    if let Some((rows, cols)) = guess_lookup(inputs, neighbour) {
                        output.warm_attempts += 1;
                        solution =
                            attempt_support(&success, failed, &rows, &cols, inputs.tolerance);
                        if solution.is_some() {
                            warm = true;
                            break;
                        }
                    }
                }
            }
            if solution.is_none() {
                solution = attempt_support(&success, failed, &full, &full, inputs.tolerance);
                if solution.is_some() {
                    output.full_hits += 1;
                }
            } else if warm {
                output.warm_hits += 1;
            }

            match solution {
                Some(solved) => {
                    unsafe {
                        *value.0.add(class_id as usize) = solved.value;
                        *kind.0.add(class_id as usize) = 1;
                    }
                    let drop_support = support_of_policy(&solved.drop_policy, inputs.max_support);
                    let check_support = support_of_policy(&solved.check_policy, inputs.max_support);
                    output.hit_classes.push(class_id);
                    for slot in 0..inputs.max_support {
                        output
                            .hit_rows
                            .push(drop_support.get(slot).copied().unwrap_or(-1));
                        output
                            .hit_cols
                            .push(check_support.get(slot).copied().unwrap_or(-1));
                    }
                }
                None => {
                    output.residue_classes.push(class_id);
                    output.residue_success.extend_from_slice(&success);
                    output.residue_failed.push(failed);
                }
            }
        }
    }
    output
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
#[pyfunction]
fn sweep_layer_rs<'py>(
    py: Python<'py>,
    work_items: PyReadonlyArray1<'_, u64>,
    profile_pool: PyReadonlyArray1<'_, u32>,
    success_child: PyReadonlyArray1<'_, i32>,
    failure_child: PyReadonlyArray1<'_, i32>,
    revival: PyReadonlyArray1<'_, f64>,
    profile_count: u64,
    guess_classes: PyReadonlyArray1<'_, u64>,
    guess_rows: PyReadonlyArray1<'_, i32>,
    guess_cols: PyReadonlyArray1<'_, i32>,
    mut value: PyReadwriteArray1<'_, f64>,
    mut solver_kind: PyReadwriteArray1<'_, u8>,
    saddle_tolerance: f64,
    max_support: u32,
    warm_start: bool,
) -> PyResult<(
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i32>>,
    u64,
    u64,
    u64,
    u64,
)> {
    let count = profile_count as usize;
    let inputs = LayerInputs {
        work_items: work_items.as_slice()?.to_vec(),
        profile_pool: profile_pool.as_slice()?.to_vec(),
        success_child: success_child.as_slice()?.to_vec(),
        failure_child: failure_child.as_slice()?.to_vec(),
        revival: revival.as_slice()?.to_vec(),
        profile_count: count,
        guess_classes: guess_classes.as_slice()?.to_vec(),
        guess_rows: guess_rows.as_slice()?.to_vec(),
        guess_cols: guess_cols.as_slice()?.to_vec(),
        tolerance: saddle_tolerance,
        max_support: max_support as usize,
        warm_start,
    };
    if inputs.work_items.len() % 4 != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "work_items must be flat (offset, len, offset, len) quadruples",
        ));
    }
    if inputs.success_child.len() != count * ACTIONS
        || inputs.failure_child.len() != count
        || inputs.revival.len() != count
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "profile tables disagree with profile_count",
        ));
    }
    if inputs.guess_rows.len() != inputs.guess_classes.len() * inputs.max_support
        || inputs.guess_cols.len() != inputs.guess_rows.len()
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "guess arrays disagree with max_support padding",
        ));
    }
    let value_slice = value.as_slice_mut()?;
    let kind_slice = solver_kind.as_slice_mut()?;
    if value_slice.len() != count * count || kind_slice.len() != count * count {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "value stores disagree with profile_count",
        ));
    }
    let value_ptr = SharedPtr(value_slice.as_mut_ptr());
    let kind_ptr = SharedPtr(kind_slice.as_mut_ptr());
    let items = inputs.work_items.len() / 4;

    let outputs: Vec<ItemOutput> = py.detach(|| {
        (0..items)
            .into_par_iter()
            .map(|item| solve_item(&inputs, item, &value_ptr, &kind_ptr))
            .collect()
    });

    if outputs.iter().any(|output| output.missing_child) {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "sweep layer read an unsolved child value",
        ));
    }
    let mut residue_classes = Vec::new();
    let mut residue_success = Vec::new();
    let mut residue_failed = Vec::new();
    let mut hit_classes = Vec::new();
    let mut hit_rows = Vec::new();
    let mut hit_cols = Vec::new();
    let (mut pure, mut warm_hits, mut full_hits, mut warm_attempts) = (0_u64, 0_u64, 0_u64, 0_u64);
    for output in outputs {
        residue_classes.extend(output.residue_classes);
        residue_success.extend(output.residue_success);
        residue_failed.extend(output.residue_failed);
        hit_classes.extend(output.hit_classes);
        hit_rows.extend(output.hit_rows);
        hit_cols.extend(output.hit_cols);
        pure += output.pure;
        warm_hits += output.warm_hits;
        full_hits += output.full_hits;
        warm_attempts += output.warm_attempts;
    }
    Ok((
        residue_classes.into_pyarray(py),
        residue_success.into_pyarray(py),
        residue_failed.into_pyarray(py),
        hit_classes.into_pyarray(py),
        hit_rows.into_pyarray(py),
        hit_cols.into_pyarray(py),
        pure,
        warm_hits,
        full_hits,
        warm_attempts,
    ))
}

#[pyfunction]
fn toeplitz_saddle_rs(
    success: PyReadonlyArray1<'_, f64>,
    failed: f64,
) -> PyResult<(f64, f64, f64)> {
    let slice = success.as_slice()?;
    if slice.len() != ACTIONS {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "success must hold exactly 60 class values",
        ));
    }
    let mut fixed = [0.0_f64; ACTIONS];
    fixed.copy_from_slice(slice);
    Ok(toeplitz_saddle(&fixed, failed))
}

#[allow(clippy::type_complexity)]
#[pyfunction]
fn attempt_support_rs<'py>(
    py: Python<'py>,
    success: PyReadonlyArray1<'_, f64>,
    failed: f64,
    rows: Vec<i64>,
    cols: Vec<i64>,
    tolerance: f64,
) -> PyResult<Option<(f64, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)>> {
    let slice = success.as_slice()?;
    if slice.len() != ACTIONS {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "success must hold exactly 60 class values",
        ));
    }
    let mut fixed = [0.0_f64; ACTIONS];
    fixed.copy_from_slice(slice);
    let rows: Vec<usize> = rows.into_iter().map(|v| v as usize).collect();
    let cols: Vec<usize> = cols.into_iter().map(|v| v as usize).collect();
    Ok(
        attempt_support(&fixed, failed, &rows, &cols, tolerance).map(|solution| {
            (
                solution.value,
                solution.drop_policy.to_vec().into_pyarray(py),
                solution.check_policy.to_vec().into_pyarray(py),
            )
        }),
    )
}

#[pymodule]
fn dth_complete_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("PARITY_CONTRACT_VERSION", PARITY_CONTRACT_VERSION)?;
    m.add_function(wrap_pyfunction!(sweep_layer_rs, m)?)?;
    m.add_function(wrap_pyfunction!(toeplitz_saddle_rs, m)?)?;
    m.add_function(wrap_pyfunction!(attempt_support_rs, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp_matrix() -> ([f64; ACTIONS], f64) {
        let mut success = [0.0_f64; ACTIONS];
        for index in 0..ACTIONS {
            success[index] = -1.0 + 2.0 * (index as f64) / 59.0;
        }
        (success, 0.25)
    }

    #[test]
    fn toeplitz_matches_brute_force() {
        let (success, failed) = ramp_matrix();
        let (gap, maximin, minimax) = toeplitz_saddle(&success, failed);
        let mut brute_maximin = f64::NEG_INFINITY;
        for drop in 0..ACTIONS {
            let mut row_min = f64::INFINITY;
            for check in 0..ACTIONS {
                let cell = matrix_cell(&success, failed, drop, check);
                if cell < row_min {
                    row_min = cell;
                }
            }
            if row_min > brute_maximin {
                brute_maximin = row_min;
            }
        }
        let mut brute_minimax = f64::INFINITY;
        for check in 0..ACTIONS {
            let mut col_max = f64::NEG_INFINITY;
            for drop in 0..ACTIONS {
                let cell = matrix_cell(&success, failed, drop, check);
                if cell > col_max {
                    col_max = cell;
                }
            }
            if col_max < brute_minimax {
                brute_minimax = col_max;
            }
        }
        assert_eq!(maximin, brute_maximin);
        assert_eq!(minimax, brute_minimax);
        assert_eq!(gap, brute_minimax - brute_maximin);
    }

    #[test]
    fn pinned_elimination_solves_a_known_system() {
        // 2x + y = 5, x + 3y = 10 -> x = 1, y = 3.
        let mut a = vec![2.0, 1.0, 1.0, 3.0];
        let mut b = vec![5.0, 10.0];
        let mut solution = vec![0.0, 0.0];
        assert!(solve_linear_pinned(&mut a, &mut b, 2, &mut solution));
        assert!((solution[0] - 1.0).abs() < 1e-12);
        assert!((solution[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn support_attempt_fails_closed_on_singular_supports() {
        let (success, failed) = ramp_matrix();
        // A constant submatrix is singular for k >= 2: rows/cols below the
        // diagonal all read `failed`.
        let rows = vec![58, 59];
        let cols = vec![0, 1];
        assert!(attempt_support(&success, failed, &rows, &cols, 1e-6).is_none());
    }

    #[test]
    fn full_support_attempt_certifies_or_declines() {
        let (success, failed) = ramp_matrix();
        let full: Vec<usize> = (0..ACTIONS).collect();
        if let Some(solution) = attempt_support(&success, failed, &full, &full, 1e-6) {
            assert!(solution.value.is_finite());
            let mass: f64 = solution.drop_policy.iter().sum();
            assert!((mass - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn support_extraction_trims_by_mass_with_index_ties() {
        let mut policy = [0.0_f64; ACTIONS];
        policy[3] = 0.4;
        policy[10] = 0.3;
        policy[20] = 0.3;
        policy[40] = 1e-12; // below the mass floor, excluded
        assert_eq!(support_of_policy(&policy, 2), vec![3, 10]);
        assert_eq!(support_of_policy(&policy, 4), vec![3, 10, 20]);
    }
}
