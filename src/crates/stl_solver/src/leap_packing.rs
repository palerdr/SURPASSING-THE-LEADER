//! We solve the packing LP and certify the original stage matrix.
const N: usize = 60;
const W: usize = 61;
const EPS: f64 = 1e-10;

pub struct Solver {
    t: [[f64; W]; W],
    basic: [usize; N],
    nonbasic: [usize; N],
    pub ready: bool,
}

impl Solver {
    pub fn new() -> Self {
        Self {
            t: [[0.; W]; W],
            basic: [0; N],
            nonbasic: [0; N],
            ready: false,
        }
    }
    fn coefficient(a: &[f64; N], row: usize, variable: usize) -> f64 {
        if variable >= N {
            f64::from(variable - N == row)
        } else if row >= variable {
            a[row - variable]
        } else {
            0.
        }
    }
    fn cold(&mut self, a: &[f64; N]) {
        self.t = [[0.; W]; W];
        for i in 0..N {
            self.basic[i] = N + i;
            self.nonbasic[i] = i;
            self.t[i][N] = 1.;
            self.t[N][i] = 1.;
            for j in 0..=i {
                self.t[i][j] = a[i - j];
            }
        }
    }
    fn refactor(&mut self, a: &[f64; N]) -> bool {
        // We retain variable identities, then rebuild the basis for new payoffs.
        let mut work = [[0.; 121]; N];
        for i in 0..N {
            for j in 0..N {
                work[i][j] = Self::coefficient(a, i, self.basic[j]);
                work[i][N + j] = Self::coefficient(a, i, self.nonbasic[j]);
            }
            work[i][120] = 1.;
        }
        for col in 0..N {
            let pivot = (col..N)
                .max_by(|&i, &j| work[i][col].abs().total_cmp(&work[j][col].abs()))
                .unwrap();
            if !work[pivot][col].is_finite() || work[pivot][col].abs() < 1e-12 {
                return false;
            }
            work.swap(col, pivot);
            let scale = work[col][col];
            for j in col..121 {
                work[col][j] /= scale;
            }
            let source = work[col];
            for i in 0..N {
                if i == col {
                    continue;
                }
                let factor = work[i][col];
                for j in col..121 {
                    work[i][j] = (-factor).mul_add(source[j], work[i][j]);
                }
            }
        }
        self.t = [[0.; W]; W];
        for i in 0..N {
            self.t[i].copy_from_slice(&work[i][N..]);
        }
        for j in 0..N {
            self.t[N][j] = f64::from(self.nonbasic[j] < N);
        }
        for i in 0..N {
            if self.basic[i] < N {
                for j in 0..W {
                    self.t[N][j] -= self.t[i][j];
                }
            }
        }
        self.t.iter().flatten().all(|x| x.is_finite())
    }
    fn pivot(&mut self, row: usize, col: usize) {
        let scale = self.t[row][col];
        for j in 0..W {
            self.t[row][j] /= scale;
        }
        self.t[row][col] = 1. / scale;
        let source = self.t[row];
        for i in 0..W {
            if i == row {
                continue;
            }
            let factor = self.t[i][col];
            for j in 0..W {
                if j != col {
                    self.t[i][j] = (-factor).mul_add(source[j], self.t[i][j]);
                }
            }
            self.t[i][col] = -factor / scale;
        }
        std::mem::swap(&mut self.basic[row], &mut self.nonbasic[col]);
    }
    fn optimize(&mut self) -> Option<usize> {
        for iteration in 0..512 {
            let primal = (0..N).all(|i| self.t[i][N] >= -EPS);
            let dual = (0..N).all(|j| self.t[N][j] <= EPS);
            if primal && dual {
                return Some(iteration);
            }
            let (row, col);
            if primal {
                col = (0..N)
                    .filter(|&j| self.t[N][j] > EPS)
                    .max_by(|&i, &j| self.t[N][i].total_cmp(&self.t[N][j]))?;
                row = (0..N).filter(|&i| self.t[i][col] > EPS).min_by(|&i, &j| {
                    (self.t[i][N].max(0.) / self.t[i][col])
                        .total_cmp(&(self.t[j][N].max(0.) / self.t[j][col]))
                        .then(self.basic[i].cmp(&self.basic[j]))
                })?;
            } else if dual {
                row = (0..N)
                    .filter(|&i| self.t[i][N] < -EPS)
                    .min_by(|&i, &j| self.t[i][N].total_cmp(&self.t[j][N]))?;
                col = (0..N).filter(|&j| self.t[row][j] < -EPS).min_by(|&i, &j| {
                    (self.t[N][i] / self.t[row][i])
                        .total_cmp(&(self.t[N][j] / self.t[row][j]))
                        .then(self.nonbasic[i].cmp(&self.nonbasic[j]))
                })?;
            } else {
                return None;
            }
            self.pivot(row, col);
        }
        None
    }
    fn certificate(&self, s: &[f64], f: f64) -> Option<(f64, f64)> {
        let mut p = [0.; N];
        let mut q = [0.; N];
        for i in 0..N {
            if self.basic[i] < N {
                p[self.basic[i]] = self.t[i][N].max(0.);
            }
        }
        for j in 0..N {
            if self.nonbasic[j] >= N {
                q[self.nonbasic[j] - N] = (-self.t[N][j]).max(0.);
            }
        }
        let ps: f64 = p.iter().sum();
        let qs: f64 = q.iter().sum();
        if !ps.is_finite() || !qs.is_finite() || ps <= 0. || qs <= 0. {
            return None;
        }
        for i in 0..N {
            p[i] /= ps;
            q[i] /= qs;
        }
        let mut rows = [0.; N];
        let mut cols = [0.; N];
        let mut prefix = 0.;
        let mut suffix = 0.;
        for i in 0..N {
            rows[i] = f * prefix;
            prefix += q[i];
            cols[N - 1 - i] = f * suffix;
            suffix += p[N - 1 - i];
        }
        for i in 0..N {
            for j in i..N {
                rows[i] = s[j - i].mul_add(q[j], rows[i]);
                cols[j] = s[j - i].mul_add(p[i], cols[j]);
            }
        }
        let upper = rows.into_iter().fold(f64::NEG_INFINITY, f64::max);
        let lower = cols.into_iter().fold(f64::INFINITY, f64::min);
        let gap = upper - lower;
        if !gap.is_finite() || !(-1e-12..=1e-6).contains(&gap) {
            None
        } else {
            Some(((upper + lower) * 0.5, gap))
        }
    }
    pub fn solve(&mut self, stage: &[f64], warm: bool) -> [f64; 5] {
        let f = stage[N];
        let d = f - stage[0];
        if d <= 1e-12 || stage[..W].iter().any(|x| !x.is_finite()) {
            self.ready = false;
            return [f64::NAN, f64::NAN, 0., 0., 0.];
        }
        let a = std::array::from_fn(|i| (f - stage[i]) / d);
        let reused = warm && self.ready && self.refactor(&a);
        if !reused {
            self.cold(&a);
        }
        let mut pivots = self.optimize();
        let mut certificate = pivots.and_then(|_| self.certificate(stage, f));
        let mut restart = false;
        if reused && certificate.is_none() {
            restart = true;
            self.cold(&a);
            pivots = self.optimize();
            certificate = pivots.and_then(|_| self.certificate(stage, f));
        }
        self.ready = certificate.is_some();
        if let Some((value, gap)) = certificate {
            [
                value,
                gap,
                pivots.unwrap() as f64,
                f64::from(reused),
                f64::from(restart),
            ]
        } else {
            [
                f64::NAN,
                f64::NAN,
                512.,
                f64::from(reused),
                f64::from(restart),
            ]
        }
    }
}
