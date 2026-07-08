use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

mod cfr;
mod payoff;
mod game;

#[pyfunction]
fn solve_minimax_rs<'py>(
    py: Python<'py>,
    payoff: PyReadonlyArray2<'_, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, f64)> {
    let matrix = payoff.as_array();
    let rows = matrix.nrows();
    if rows == 0 || matrix.ncols() == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "payoff matrix must be non-empty",
        ));
    }

    // Placeholder only: this proves the Python/Rust/Numpy extension boundary.
    // The production minimax implementation should replace this before use.
    let strategy = vec![1.0 / rows as f64; rows];
    let value = 0.0;

    Ok((PyArray1::from_vec(py, strategy), value))
}

#[pyfunction]
fn regret_plus_strategy_rs<'py>(
    py: Python<'py>,
    regrets: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let strategy = cfr::regret_plus_strategy(regrets.as_slice()?);
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
        return Err(pyo3::exceptions::PyValueError::new_err(
            "payoff matrix must be non-empty",
        ));
    }
    if iterations == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "CFR+ iterations must be positive",
        ));
    }

    let (strategy, _col_strategy, value) = cfr::solve_cfr_plus_dense(
        payoff.as_slice()?,
        rows,
        cols,
        iterations,
        average_delay,
        linear_weighting,
    );
    Ok((PyArray1::from_vec(py, strategy), value))
}

#[pymodule]
fn stl_solver_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_minimax_rs, m)?)?;
    m.add_function(wrap_pyfunction!(regret_plus_strategy_rs, m)?)?;
    m.add_function(wrap_pyfunction!(solve_cfr_plus_rs, m)?)?;
    Ok(())
}
