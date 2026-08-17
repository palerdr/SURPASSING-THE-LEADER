use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

mod cfr;
mod game;
mod matrix;
mod minimax;
mod payoff;
mod transition;
mod value;

const EXACT_MINIMAX_AVAILABLE: bool = false;

fn exact_minimax_available() -> bool {
    EXACT_MINIMAX_AVAILABLE
}

#[pyfunction]
fn solve_minimax_rs<'py>(
    _py: Python<'py>,
    _payoff: PyReadonlyArray2<'_, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, f64)> {
    Err(PyNotImplementedError::new_err(
        "stl_solver_rs has no certified exact-minimax implementation; use the Python authority",
    ))
}

#[pyfunction]
fn regret_plus_strategy_rs<'py>(
    py: Python<'py>,
    regrets: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let regrets = regrets.as_slice()?;
    if regrets.is_empty() {
        return Err(PyValueError::new_err("regrets must be non-empty"));
    }
    if regrets.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("regrets must be finite"));
    }
    let positive_total: f64 = regrets.iter().map(|value| value.max(0.0)).sum();
    if !positive_total.is_finite() {
        return Err(PyValueError::new_err(
            "positive regret mass must have a finite sum",
        ));
    }
    let strategy = cfr::regret_plus_strategy(regrets);
    Ok(PyArray1::from_vec(py, strategy))
}

#[pyfunction]
#[pyo3(signature = (payoff, iterations=2000, average_delay=100, linear_weighting=true))]
fn solve_cfr_plus_rs<'py>(
    py: Python<'py>,
    payoff: PyReadonlyArray2<'_, f64>,
    iterations: usize,
    average_delay: usize,
    linear_weighting: bool,
) -> PyResult<(Bound<'py, PyArray1<f64>>, f64)> {
    let matrix = payoff.as_array();
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    if rows == 0 || cols == 0 {
        return Err(PyValueError::new_err("payoff matrix must be non-empty"));
    }
    if iterations == 0 {
        return Err(PyValueError::new_err("CFR+ iterations must be positive"));
    }
    let payoff = payoff.as_slice()?;
    if payoff.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("payoff matrix must be finite"));
    }

    let (strategy, col_strategy, value) = cfr::solve_cfr_plus_dense(
        payoff,
        rows,
        cols,
        iterations,
        average_delay,
        linear_weighting,
    );
    let row_mass: f64 = strategy.iter().sum();
    let col_mass: f64 = col_strategy.iter().sum();
    if !value.is_finite()
        || strategy.iter().any(|mass| !mass.is_finite() || *mass < 0.0)
        || col_strategy
            .iter()
            .any(|mass| !mass.is_finite() || *mass < 0.0)
        || (row_mass - 1.0).abs() > 1e-9
        || (col_mass - 1.0).abs() > 1e-9
    {
        return Err(PyRuntimeError::new_err(
            "CFR+ failed to produce finite simplex strategies and value",
        ));
    }
    Ok((PyArray1::from_vec(py, strategy), value))
}

#[pymodule]
fn stl_solver_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("EXACT_MINIMAX_AVAILABLE", exact_minimax_available())?;
    m.add_function(wrap_pyfunction!(solve_minimax_rs, m)?)?;
    m.add_function(wrap_pyfunction!(regret_plus_strategy_rs, m)?)?;
    m.add_function(wrap_pyfunction!(solve_cfr_plus_rs, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_minimax_surface_is_explicitly_unavailable() {
        assert!(!exact_minimax_available());
    }
}
