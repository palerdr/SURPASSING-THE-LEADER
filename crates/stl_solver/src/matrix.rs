struct Matrix<'a> {
    data: &'a [f64],
    rows: usize,
    cols: usize,
}

impl<'a> Matrix<'a> {
    fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.cols + col]
    }
}

struct MatrixSolveResult {
    strategy: Vec<f64>,
    value: f64,
}
