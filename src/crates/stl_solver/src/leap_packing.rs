//! We solve the packing LP and certify the original stage matrix.
//!
//! You can start from the slack basis, from a warm basis, or from the
//! recurrence basis in which every packing variable is basic. The basis
//! matrix `A^T` of that last start is lower triangular Toeplitz, so we write
//! its inverse from one series recurrence without a factorization.
const N: usize = 60;
const W: usize = 61;
const EPS: f64 = 1e-10;

pub struct Solver {
    t: [[f64; W]; W],
    basic: [usize; N],
    nonbasic: [usize; N],
    pub ready: bool,
    /// We record the basic Dropper actions and the nonbasic Checker slacks of
    /// the last certified basis. The two masks have equal bit counts.
    pub support: (u64, u64),
}

impl Solver {
    pub fn new() -> Self {
        Self {
            t: [[0.; W]; W],
            basic: [0; N],
            nonbasic: [0; N],
            ready: false,
            support: (0, 0),
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
        self.objective();
        self.t.iter().flatten().all(|x| x.is_finite())
    }
    /// We rebuild the objective row for the true costs: one per packing
    /// variable and zero per slack.
    fn objective(&mut self) {
        self.t[N] = [0.; W];
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
    }
    /// We make every packing variable basic. The basis matrix is `A^T`, lower
    /// triangular Toeplitz with first column `a`. Its inverse is lower
    /// triangular Toeplitz with first column `c`, where `c[0] = 1` and
    /// `c[k] = -sum_{m=1..k} a[m] c[k-m]`. Basic value `i` is
    /// `c[0] + ... + c[i]`, and the reduced cost of slack `j` is
    /// `-(c[0] + ... + c[59-j])`, the negated sum of column `j` of the inverse.
    /// This start costs one recurrence instead of a 60-by-60 factorization.
    fn crash(&mut self, a: &[f64; N]) -> bool {
        let mut c = [0.; N];
        c[0] = 1.;
        for k in 1..N {
            let mut x = 0.;
            for m in 1..=k {
                x = (-a[m]).mul_add(c[k - m], x);
            }
            c[k] = x;
        }
        let mut prefix = [0.; N];
        let mut total = 0.;
        for k in 0..N {
            total += c[k];
            prefix[k] = total;
        }
        self.t = [[0.; W]; W];
        for i in 0..N {
            self.basic[i] = i;
            self.nonbasic[i] = N + i;
            for j in 0..=i {
                self.t[i][j] = c[i - j];
            }
            self.t[i][N] = prefix[i];
        }
        self.objective();
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
    fn basis_support(&self) -> (u64, u64) {
        let mut p = 0;
        let mut q = 0;
        for i in 0..N {
            if self.basic[i] < N {
                p |= 1 << self.basic[i];
            }
            if self.nonbasic[i] >= N {
                q |= 1 << (self.nonbasic[i] - N);
            }
        }
        (p, q)
    }
    /// We start from the recurrence basis. Its basic values and reduced costs
    /// can both have the wrong sign. We therefore zero each positive reduced
    /// cost, run the dual simplex to primal feasibility, restore the true
    /// costs, and finish with the primal simplex. If this path fails, we
    /// restart from the slack basis. The full-matrix certificate controls
    /// acceptance on both paths. Slot 3 of the result is 1 for this start;
    /// slot 4 is 1 after a slack restart.
    pub fn solve_crash(&mut self, stage: &[f64]) -> [f64; 5] {
        let Some(a) = Self::coefficients(stage) else {
            self.ready = false;
            return [f64::NAN, f64::NAN, 0., 0., 0.];
        };
        let f = stage[N];
        let mut pivots = None;
        let mut certificate = None;
        if self.crash(&a) {
            for j in 0..N {
                if self.t[N][j] > 0. {
                    self.t[N][j] = 0.;
                }
            }
            if let Some(first) = self.optimize() {
                self.objective();
                if let Some(second) = self.optimize() {
                    pivots = Some(first + second);
                    certificate = self.certificate(stage, f);
                }
            }
        }
        let restart = certificate.is_none();
        if restart {
            self.cold(&a);
            pivots = self.optimize();
            certificate = pivots.and_then(|_| self.certificate(stage, f));
        }
        self.finish(certificate, pivots, 1., f64::from(restart))
    }
    fn coefficients(stage: &[f64]) -> Option<[f64; N]> {
        let f = stage[N];
        let d = f - stage[0];
        if d <= 1e-12 || stage[..W].iter().any(|x| !x.is_finite()) {
            return None;
        }
        Some(std::array::from_fn(|i| (f - stage[i]) / d))
    }
    fn finish(
        &mut self,
        certificate: Option<(f64, f64)>,
        pivots: Option<usize>,
        start: f64,
        restart: f64,
    ) -> [f64; 5] {
        self.ready = certificate.is_some();
        match certificate {
            Some((value, gap)) => {
                self.support = self.basis_support();
                [value, gap, pivots.unwrap() as f64, start, restart]
            }
            None => [f64::NAN, f64::NAN, 512., start, restart],
        }
    }
    pub fn solve(&mut self, stage: &[f64], warm: bool) -> [f64; 5] {
        let f = stage[N];
        let Some(a) = Self::coefficients(stage) else {
            self.ready = false;
            return [f64::NAN, f64::NAN, 0., 0., 0.];
        };
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
        self.finish(certificate, pivots, f64::from(reused), f64::from(restart))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crash_restarts_from_slack_when_the_series_overflows() {
        // With d = 2e-12, each a[k] is near 2.5e11, so c overflows and the
        // recurrence basis is rejected before any pivot.
        let mut stage = [-0.5; W];
        stage[0] = 0.;
        stage[N] = 2e-12;
        let crash = Solver::new().solve_crash(&stage);
        let cold = Solver::new().solve(&stage, false);
        assert_eq!((crash[3], crash[4]), (1., 1.));
        assert!(crash[0].is_finite() && crash[1] <= 1e-6);
        assert_eq!(crash[0].to_bits(), cold[0].to_bits());
    }

    #[test]
    fn crash_certifies_a_kinked_stage_without_restart() {
        let mut stage = [0.; W];
        for k in 0..N {
            stage[k] = -0.35 + 0.006 * k as f64 - if k >= 30 { 0.04 } else { 0. };
        }
        stage[N] = 0.55;
        let mut solver = Solver::new();
        let crash = solver.solve_crash(&stage);
        let cold = Solver::new().solve(&stage, false);
        assert_eq!((crash[3], crash[4]), (1., 0.));
        assert!(crash[1] <= 1e-6 && cold[1] <= 1e-6);
        assert!((crash[0] - cold[0]).abs() <= 1e-6);
        let (p, q) = solver.support;
        assert!(p != 0 && p.count_ones() == q.count_ones());
        assert!(crash[2] < cold[2]);
    }
}
