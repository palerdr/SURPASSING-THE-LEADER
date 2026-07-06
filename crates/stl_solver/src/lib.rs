use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

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

#[pymodule]
fn stl_solver_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_minimax_rs, m)?)?;
    Ok(())
}
