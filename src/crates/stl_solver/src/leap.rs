//! We solve batches of simultaneous half-round games from the Dropper's view.
//!
//! The Dropper maximizes the payoff; the Checker minimizes it. With zero-based
//! action indices `i` and `j`, we represent the 60-by-60 payoff matrix as
//! `M[i, j] = s[j - i]` for `j >= i`, and `f` for `j < i`. Here `s[k]` is the
//! payoff after a successful check with elapsed time `k + 1`; `f` includes the
//! revival chance after a failed check. We exploit the constant diagonals of
//! this Toeplitz matrix to avoid constructing it for each state class.
//!
//! We try pure-action bounds before certifying a candidate mixed strategy
//! from a recurrence. In the leap window, Baku can add row 61, whose entries
//! are all `f`; we obtain the extended game's value as `max(v60, f)`.
//! We can reuse a preceding Checker-row support for a reduced solve. We
//! certify both reconstructed policies against all rows and columns before
//! acceptance. Python refreshes a missed support with HiGHS and retries the
//! deferred part of that row. Python also owns the clock schedule.
//! You can find the numerical enclosure proof in
//! `src/crates/docs/LEAP_CERTIFICATE.md` and the scalar authority in
//! `src/stl/solver/leap_oracle.py`.
use numpy::{
    PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1, PyReadwriteArray2, PyUntypedArrayMethods,
};
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;

const L: usize = 16;
const A: usize = 60;
const GAP: f64 = 1e-6;
unsafe extern "C" {
    fn fegetround() -> i32;
}

/// We collect each chunk's certified bases as (class, Dropper mask, Checker mask).
type Bases = Vec<(usize, u64, u64)>;

fn environment_ok() -> bool {
    // We require round-to-nearest and gradual underflow for the certificate.
    // We use black_box to keep the underflow probe as a runtime operation.
    (unsafe { fegetround() == 0 }) && std::hint::black_box(f64::MIN_POSITIVE) * 0.5 != 0.0
}

/// We certify the residue through a packing LP, with a fresh basis per stage.
/// We return NaN and kind 255 for unsupported or uncertified stages. Python
/// sends those stages to HiGHS. Kind 3 records a full-matrix certificate.
/// Set `crash` to start each stage from the recurrence basis; we restart from
/// the slack basis when that path fails. You can pass `support_out` with shape
/// `[class, 2]` to receive each certified basis as Dropper and Checker masks,
/// the seed that native kernel mode keeps; we write zeros for failed classes.
#[pyfunction]
#[pyo3(signature = (s, f, window, out, kind, crash=false, support_out=None))]
pub fn solve_packing_rs(
    s: PyReadonlyArray2<'_, f64>,
    f: PyReadonlyArray1<'_, f64>,
    window: bool,
    mut out: PyReadwriteArray1<'_, f64>,
    mut kind: PyReadwriteArray1<'_, u8>,
    crash: bool,
    mut support_out: Option<PyReadwriteArray2<'_, u64>>,
) -> PyResult<u64> {
    if s.shape() != [f.len(), A] || out.len() != f.len() || kind.len() != f.len() {
        return Err(PyValueError::new_err("packing shape mismatch"));
    }
    let supports = match support_out.as_mut() {
        Some(array) => {
            if array.shape() != [f.len(), 2] {
                return Err(PyValueError::new_err("packing support shape mismatch"));
            }
            let values = array.as_slice_mut()?;
            values.fill(0);
            Some(values)
        }
        None => None,
    };
    let s = s.as_slice()?;
    let f = f.as_slice()?;
    let out = out.as_slice_mut()?;
    let kind = kind.as_slice_mut()?;
    if s.iter().chain(f).any(|x| !x.is_finite()) {
        return Err(PyValueError::new_err("packing needs finite payoffs"));
    }
    if !environment_ok() {
        return Err(PyValueError::new_err(
            "unsupported floating-point environment",
        ));
    }
    let (failures, bases) = out
        .par_chunks_mut(2048)
        .zip(kind.par_chunks_mut(2048))
        .enumerate()
        .map(|(chunk, (values, kinds))| -> PyResult<(u64, Bases)> {
            if !environment_ok() {
                return Err(PyValueError::new_err(
                    "unsupported worker floating-point environment",
                ));
            }
            let mut solver = crate::leap_packing::Solver::new();
            let mut stage = [0.; A + 1];
            let mut failed = 0;
            let mut bases = Vec::new();
            for i in 0..values.len() {
                let j = chunk * 2048 + i;
                stage[..A].copy_from_slice(&s[j * A..(j + 1) * A]);
                stage[A] = f[j];
                let solved = if crash {
                    solver.solve_crash(&stage)
                } else {
                    solver.solve(&stage, false)
                };
                values[i] = f64::NAN;
                kinds[i] = 255;
                if solved[0].is_finite() {
                    let value = if window {
                        solved[0].max(f[j])
                    } else {
                        solved[0]
                    };
                    if value.abs() > 1. + 1e-9 {
                        return Err(PyValueError::new_err("packing value outside utility range"));
                    }
                    values[i] = value;
                    kinds[i] = 3;
                    if supports.is_some() {
                        bases.push((j, solver.support.0, solver.support.1));
                    }
                } else {
                    failed += 1;
                }
            }
            Ok((failed, bases))
        })
        .try_reduce(
            || (0, Vec::new()),
            |mut a, mut b| {
                a.0 += b.0;
                a.1.append(&mut b.1);
                Ok(a)
            },
        )?;
    if let Some(output) = supports {
        for (j, p, q) in bases {
            output[2 * j] = p;
            output[2 * j + 1] = q;
        }
    }
    Ok(failures)
}

/// We certify one value per input class and return the count that needs HiGHS.
///
/// You supply checker ids in `pcs` and dropper ids in `pds`, in matching order.
/// After the role swap, the old Dropper is the child Checker, so we read row
/// `pds[i]` from both child tables and negate its values. Python groups classes
/// that share these tables before calling this function.
///
/// You map each checker profile to 60 success columns through `succ_col` and
/// one failure column through `fail_col`. The failure table can use compressed
/// reset-profile columns; we use the supplied column map without decoding it.
/// You reserve the last column of each table for WIN, fill it with `-1.0`, and
/// map terminal children to that column. We reject nonfinite child values that
/// we read; you can leave NaN in unreachable cells that we do not read.
///
/// You set `window` when Baku can drop at 61. A turn duration of 61 alone does
/// not grant the extra row: Hal still has 60 actions in H1[59]. You must supply
/// `saddle_tolerance = 1e-6`; we do not permit a weaker certificate gate.
///
/// We write `out` and `kind` in input order. We use kind 0 for the pure test,
/// kind 1 for an equalizer certificate, and kind 2 when the leap row wins or
/// ties the square-stage value. We leave NaN and kind 255 for uncertified
/// classes. Python must send those classes to HiGHS before storing the key.
/// With optional support seeds, kind 4 denotes a reduced support certificate
/// and kind 5 denotes a certificate after a paired edge move. We use the
/// existing recurrence scratch to reconstruct both candidate policies.
///
/// You supply `supports` as `[profile, 2]` masks over zero-based actions 0..59
/// and `support_out` as `[class, 2]`. Sort residue classes by Checker and then
/// Dropper to reuse each row's preceding accepted support. Each Rayon chunk
/// starts from the supplied seed; workers own their policy state and outputs.
/// Set `stop_on_support_miss` to defer later classes in that row with kind 254.
/// Python then solves the first miss with HiGHS, updates its seed, and retries.
/// The returned count includes failed and deferred classes. Support masks name
/// the square game's actions; the existing constant-row proof lifts a window
/// certificate. Optional supports leave the original two-rung API unchanged.
///
/// Set `native` to solve the residue inside each worker. We then run the
/// packing LP of `leap_packing.rs` on each class that fails the two rungs and
/// record kind 3. Set `crash` to start that LP from the recurrence basis. Set
/// `kink_attempts` above zero to try a support guess before the LP. We move
/// the certified support of the preceding residue class in the same
/// 1024-class chunk with the success-payoff kink (see `kink_seed`), then try
/// at most `kink_attempts` supports: the
/// moved guess records kind 4 and a later edge move records kind 5. In this
/// mode we keep kinds 3, 4, and 5 in window stages, so Python can count the
/// residue; kind 255 then marks only the classes that need HiGHS. The native
/// mode excludes `supports`, `support_out`, and `stop_on_support_miss`.
///
/// # Errors
///
/// We reject incompatible shapes or layouts, invalid indices or probabilities,
/// and unsupported floating-point environments. We also reject nonfinite
/// gathered payoffs or accepted values outside the utility range. On an error,
/// you must discard both output buffers: workers may have written some entries.
#[pyfunction]
#[pyo3(signature = (pcs, pds, succ_table, fail_table, succ_col, fail_col, rev, window, saddle_tolerance, out, kind, supports=None, support_out=None, stop_on_support_miss=false, native=false, crash=false, kink_attempts=0))]
#[allow(clippy::too_many_arguments)]
pub fn sweep_key_rs(
    pcs: PyReadonlyArray1<'_, i32>,
    pds: PyReadonlyArray1<'_, i32>,
    succ_table: PyReadonlyArray2<'_, f64>,
    fail_table: PyReadonlyArray2<'_, f64>,
    succ_col: PyReadonlyArray2<'_, i32>,
    fail_col: PyReadonlyArray1<'_, i32>,
    rev: PyReadonlyArray1<'_, f64>,
    window: bool,
    saddle_tolerance: f64,
    mut out: PyReadwriteArray1<'_, f64>,
    mut kind: PyReadwriteArray1<'_, u8>,
    supports: Option<PyReadonlyArray2<'_, u64>>,
    mut support_out: Option<PyReadwriteArray2<'_, u64>>,
    stop_on_support_miss: bool,
    native: bool,
    crash: bool,
    kink_attempts: usize,
) -> PyResult<u64> {
    let error = |s: &str| PyValueError::new_err(s.to_owned());
    if saddle_tolerance != GAP {
        return Err(error("saddle tolerance must be 1e-6"));
    }
    if (crash || kink_attempts > 0) && !native {
        return Err(error(
            "crash and kink seeds require the native residue mode",
        ));
    }
    if native && (supports.is_some() || support_out.is_some() || stop_on_support_miss) {
        return Err(error("native residue mode excludes support seeds"));
    }
    if !environment_ok() {
        return Err(error("unsupported floating-point environment"));
    }
    if !succ_table.is_c_contiguous() || !fail_table.is_c_contiguous() || !succ_col.is_c_contiguous()
    {
        return Err(error("tables must be C contiguous"));
    }
    let ss = succ_table.shape();
    let fs = fail_table.shape();
    let cs = succ_col.shape();
    let (sr, sw, fr, fw, profiles) = (ss[0], ss[1], fs[0], fs[1], cs[0]);
    let pcs = pcs.as_slice()?;
    let pds = pds.as_slice()?;
    let sc = succ_col.as_slice()?;
    let fc = fail_col.as_slice()?;
    let rev = rev.as_slice()?;
    let st = succ_table.as_slice()?;
    let ft = fail_table.as_slice()?;
    let out = out.as_slice_mut()?;
    let kind = kind.as_slice_mut()?;
    if cs[1] != A
        || fc.len() != profiles
        || rev.len() != profiles
        || pds.len() != pcs.len()
        || out.len() != pcs.len()
        || kind.len() != pcs.len()
    {
        return Err(error("incompatible input shapes"));
    }
    // We validate the column maps at call entry. Together with the profile and
    // row checks, this establishes the bounds for the unchecked child gathers.
    if sc.iter().any(|&x| x < 0 || x as usize >= sw)
        || fc.iter().any(|&x| x < 0 || x as usize >= fw)
        || pcs.iter().any(|&x| x < 0 || x as usize >= profiles)
        || pds
            .iter()
            .any(|&x| x < 0 || x as usize >= sr || x as usize >= fr)
        || rev
            .iter()
            .any(|&x| !x.is_finite() || !(0.0..=1.0).contains(&x))
    {
        return Err(error(
            "profile, column, or revival probability out of range",
        ));
    }
    // We give each worker disjoint output slices, so we need no unsafe writes.
    // Within each chunk, we solve 16 classes at a time with private scratch.
    let support_input = match supports.as_ref() {
        Some(array) => {
            if array.shape() != [profiles, 2] {
                return Err(error("support seed shape mismatch"));
            }
            let values = array.as_slice()?;
            if values.iter().any(|x| x & !SUPPORT_BITS != 0) {
                return Err(error("support bit out of range"));
            }
            Some(values)
        }
        None => None,
    };
    let output_supports = match support_out.as_mut() {
        Some(array) => {
            if array.shape() != [pcs.len(), 2] {
                return Err(error("support output shape mismatch"));
            }
            let values = array.as_slice_mut()?;
            values.fill(0);
            Some(values)
        }
        None => None,
    };
    if support_input.is_some() != output_supports.is_some() {
        return Err(error("support seeds and output must be supplied together"));
    }
    if stop_on_support_miss
        && (support_input.is_none() || pcs.windows(2).any(|pair| pair[0] > pair[1]))
    {
        return Err(error(
            "support deferral requires seeds and Checker-row order",
        ));
    }
    let (failures, updates) = out
        .par_chunks_mut(1024)
        .zip(kind.par_chunks_mut(1024))
        .enumerate()
        .map(
            |(chunk, (values, kinds))| -> Result<(u64, Vec<(usize, u64, u64)>), &'static str> {
                // We check each worker because it has its own floating-point state.
                if !environment_ok() {
                    return Err("worker floating-point environment is unsupported");
                }
                let mut failed = 0;
                let mut updates = Vec::new();
                let mut previous_pc = usize::MAX;
                let mut blocked_pc = usize::MAX;
                let mut previous_support = (0, 0);
                // In native mode each 1024-class chunk owns one packing tableau
                // and the certified support of its preceding residue class, with
                // that class's kink index. Seeds therefore restart at each chunk.
                let mut packer = crate::leap_packing::Solver::new();
                let mut residue_seed: Option<(u64, u64, Option<usize>)> = None;
                // We index scratch as [action][lane] to keep the lane loop contiguous.
                // We reuse q for recurrence coefficients and Checker weights.
                let mut s = [[0.0_f64; L]; A];
                let mut r = [[0.0_f64; L]; A];
                let mut q = [[0.0_f64; L]; A];
                for base in (0..values.len()).step_by(L) {
                    let m = L.min(values.len() - base);
                    let mut f = [0.0; L];
                    let mut d = [0.0; L];
                    let mut mn = [0.0; L];
                    let mut mx = [0.0; L];
                    let mut bounded = [false; L];
                    for lane in 0..L {
                        // We fill spare lanes with the first class in this batch and
                        // write results for the m active lanes after certification.
                        let i = chunk * 1024 + base + if lane < m { lane } else { 0 };
                        let pc = pcs[i] as usize;
                        let pd = pds[i] as usize;
                        let mut lo = f64::INFINITY;
                        let mut hi = f64::NEG_INFINITY;
                        for k in 0..A {
                            // We checked profile and column ranges before the parallel loop.
                            let x = unsafe {
                                -*st.get_unchecked(pd * sw + *sc.get_unchecked(pc * A + k) as usize)
                            };
                            if !x.is_finite() {
                                return Err("nonfinite success child");
                            }
                            s[k][lane] = x;
                            lo = lo.min(x);
                            hi = hi.max(x);
                        }
                        let child =
                            unsafe { *ft.get_unchecked(pd * fw + *fc.get_unchecked(pc) as usize) };
                        if !child.is_finite() {
                            return Err("nonfinite failure child");
                        }
                        // We negate the revived child's value after the role swap.
                        // The current Dropper receives +1 if the Checker dies.
                        f[lane] = rev[pc] * -child + (1.0 - rev[pc]);
                        if !f[lane].is_finite() {
                            return Err("nonfinite failure payoff");
                        }
                        d[lane] = s[0][lane] - f[lane];
                        // We bound the game with the extreme pure actions: the
                        // Dropper can guarantee mn, and the Checker can enforce mx.
                        mn[lane] = lo.max(f[lane].min(s[0][lane]));
                        mx[lane] = hi.min(f[lane].max(s[0][lane]));
                        bounded[lane] = lo >= -2.0 && hi <= 2.0 && f[lane].abs() <= 2.0;
                    }
                    // We derive b[k] from adjacent success-payoff differences.
                    // We substitute 1 for a zero denominator while filling scratch;
                    // the pure test or the d-size guard controls acceptance.
                    // We build the candidate for the whole batch before selecting a
                    // rung to keep the lane loops uniform.
                    for k in 1..A {
                        for lane in 0..L {
                            q[k][lane] = (s[k - 1][lane] - s[k][lane])
                                / if d[lane] != 0.0 { d[lane] } else { 1.0 };
                        }
                    }
                    // We solve r[k] = sum_{j < k} b[k-j] * r[j], with r[0] = 1.
                    // We obtain this relation by setting adjacent row payoffs equal;
                    // we fix the weight scale with r[0] and remove it by normalization.
                    // With nonnegative weights, the Dropper uses r/sum(r) and the
                    // Checker uses its reversal to equalize the square-stage payoffs.
                    r[0] = [1.0; L];
                    for k in 1..A {
                        let mut acc = [0.0; L];
                        for j in 0..k {
                            for lane in 0..L {
                                // We use one fused rounding to match the C recurrence.
                                acc[lane] = q[k - j][lane].mul_add(r[j][lane], acc[lane]);
                            }
                        }
                        r[k] = acc;
                    }
                    // We reverse and clip the weights to form a Checker candidate.
                    // We record any negative weight because clipping requires the
                    // full payoff check; nonfinite weights must invalidate the sum.
                    let mut sum = [0.0; L];
                    let mut nonnegative = [true; L];
                    for k in 0..A {
                        for lane in 0..L {
                            let x = r[A - 1 - k][lane];
                            nonnegative[lane] &= x >= 0.0;
                            q[k][lane] = if x.is_finite() { x.max(0.0) } else { f64::NAN };
                            sum[lane] += q[k][lane];
                        }
                    }
                    for lane in 0..m {
                        let i = base + lane;
                        // We retain these sentinels if neither certificate passes.
                        values[i] = f64::NAN;
                        kinds[i] = 255;
                        let current_pc = pcs[chunk * 1024 + i] as usize;
                        if stop_on_support_miss && current_pc == blocked_pc {
                            kinds[i] = 254;
                            failed += 1;
                            continue;
                        }
                        let (mut lower, mut upper) = (mn[lane], mx[lane]);
                        if window {
                            // We lift both bounds through max(., f). This preserves
                            // the enclosure and can certify the constant leap row.
                            lower = lower.max(f[lane]);
                            upper = upper.max(f[lane]);
                        }
                        let mut v;
                        let mut rung;
                        if upper - lower <= GAP {
                            // We accept the midpoint when the pure-action gap passes.
                            v = 0.5 * (mn[lane] + mx[lane]);
                            rung = 0;
                        } else {
                            rung = 1;
                            if d[lane].abs() < 1e-12 || !sum[lane].is_finite() || sum[lane] <= 0.0 {
                                if !native {
                                    blocked_pc = current_pc;
                                    failed += 1;
                                    continue;
                                }
                                let Some((value, accepted_kind)) = solve_residue(
                                    &std::array::from_fn(
                                        |k| if k < A { s[k][lane] } else { f[lane] },
                                    ),
                                    &std::array::from_fn(|k| r[k][lane]),
                                    window,
                                    crash,
                                    kink_attempts,
                                    &mut residue_seed,
                                    &mut packer,
                                ) else {
                                    failed += 1;
                                    continue;
                                };
                                v = value;
                                rung = accepted_kind;
                            } else if nonnegative[lane] && bounded[lane] && d[lane].abs() >= GAP {
                                // Under the certificate's hypotheses, we enclose both
                                // saddle bounds within 1e-10 of this weight-sum value.
                                // We can omit the matrix product on this branch.
                                v = f[lane] + d[lane] / sum[lane];
                            } else {
                                // We normalize the clipped candidate and evaluate Mq.
                                // With the Dropper's mix as reverse(q), we have
                                // (p^T M)[j] = (Mq)[59-j], so one product gives both
                                // saddle bounds: min(Mq) and max(Mq).
                                for row in &mut q {
                                    row[lane] /= sum[lane];
                                }
                                let mut cum = 0.0;
                                lower = f64::INFINITY;
                                upper = f64::NEG_INFINITY;
                                let mut ok = true;
                                for k in 0..A {
                                    // We sum the success suffix of row k and add
                                    // f times the Checker mass before its diagonal.
                                    let mut x = 0.0;
                                    for j in 0..A - k {
                                        x = s[j][lane].mul_add(q[k + j][lane], x);
                                    }
                                    x = f[lane].mul_add(cum, x);
                                    ok &= x.is_finite();
                                    lower = lower.min(x);
                                    upper = upper.max(x);
                                    cum += q[k][lane];
                                }
                                v = 0.5 * (lower + upper);
                                if window {
                                    lower = lower.max(f[lane]);
                                    upper = upper.max(f[lane]);
                                }
                                if !ok || (cum - 1.0).abs() > 1e-12 || upper - lower > GAP {
                                    let mut reduced = None;
                                    if let Some(seeds) = support_input {
                                        let pc = pcs[chunk * 1024 + i] as usize;
                                        if pc != previous_pc {
                                            previous_pc = pc;
                                            previous_support = (seeds[2 * pc], seeds[2 * pc + 1]);
                                        }
                                        let stage = std::array::from_fn(|k| s[k][lane]);
                                        let recurrence = std::array::from_fn(|k| r[k][lane]);
                                        reduced = solve_support(
                                            &stage,
                                            f[lane],
                                            &recurrence,
                                            previous_support,
                                            window,
                                            usize::MAX,
                                        );
                                    }
                                    if let Some((value, accepted_kind, p_mask, q_mask)) = reduced {
                                        v = value;
                                        rung = accepted_kind;
                                        previous_support = (p_mask, q_mask);
                                        updates.push((chunk * 1024 + i, p_mask, q_mask));
                                    } else if native {
                                        let Some((value, accepted_kind)) = solve_residue(
                                            &std::array::from_fn(|k| {
                                                if k < A { s[k][lane] } else { f[lane] }
                                            }),
                                            &std::array::from_fn(|k| r[k][lane]),
                                            window,
                                            crash,
                                            kink_attempts,
                                            &mut residue_seed,
                                            &mut packer,
                                        ) else {
                                            failed += 1;
                                            continue;
                                        };
                                        v = value;
                                        rung = accepted_kind;
                                    } else {
                                        blocked_pc = current_pc;
                                        failed += 1;
                                        continue;
                                    }
                                }
                            }
                        }
                        if window {
                            // We choose between the square-stage value and the
                            // constant leap row; ties belong to kind 2 by contract.
                            // Native mode keeps its residue kinds for counting.
                            if f[lane] >= v && !(native && rung >= 3) {
                                rung = 2;
                            }
                            v = v.max(f[lane]);
                        }
                        if !v.is_finite() || v.abs() > 1.0 + 1e-9 {
                            return Err("stage value out of range");
                        }
                        values[i] = v;
                        kinds[i] = rung;
                    }
                }
                Ok((failed, updates))
            },
        )
        .try_reduce(
            || (0, Vec::new()),
            |mut a, mut b| {
                a.0 += b.0;
                a.1.append(&mut b.1);
                Ok(a)
            },
        )
        .map_err(error)?;
    if let Some(output) = output_supports {
        for (i, p, q) in updates {
            output[2 * i] = p;
            output[2 * i + 1] = q;
        }
    }
    Ok(failures)
}

const SUPPORT_BITS: u64 = (1_u64 << A) - 1;
const EDGE_LIMIT: usize = 64;

fn reverse_support(mask: u64) -> u64 {
    mask.reverse_bits() >> (64 - A)
}

/// We solve the hole equations in g and reconstruct a feasible column mix.
/// We have (Mq)[i] - (Mq)[i+1] = (s[0]-f)*g[i]. Adjacent supported
/// rows therefore require g[i]=0; across a hole we require a zero sum.
/// The inverse recurrence gives q[j] = sum_{t>=j} r[t-j]*g[t].
fn reduced_weights(r: &[f64; A], p_mask: u64, q_mask: u64) -> Option<[f64; A]> {
    if p_mask == 0 || q_mask == 0 || p_mask.count_ones() != q_mask.count_ones() {
        return None;
    }
    let mut positions = [0_usize; A];
    let mut n = 0;
    for t in 0..A - 1 {
        if (p_mask >> t) & 3 != 3 {
            positions[n] = t;
            n += 1;
        }
    }
    positions[n] = A - 1;
    n += 1;
    let mut a = [[0.0; A + 1]; A];
    let mut equation = 0;
    for j in 0..A {
        if q_mask & (1 << j) == 0 {
            for k in 0..n {
                let t = positions[k];
                if t >= j {
                    a[equation][k] = r[t - j];
                }
            }
            equation += 1;
        }
    }
    let mut previous = None;
    for c in 0..A {
        if p_mask & (1 << c) != 0 {
            if let Some(start) = previous {
                if c > start + 1 {
                    if equation >= n {
                        return None;
                    }
                    for k in 0..n {
                        a[equation][k] = if positions[k] >= start && positions[k] < c {
                            1.0
                        } else {
                            0.0
                        };
                    }
                    equation += 1;
                }
            }
            previous = Some(c);
        }
    }
    if equation + 1 != n {
        return None;
    }
    let mut cumulative = [0.0; A];
    let mut total = 0.0;
    for j in 0..A {
        total += r[j];
        cumulative[j] = total;
    }
    for k in 0..n {
        a[equation][k] = cumulative[positions[k]];
    }
    a[equation][n] = 1.0;
    // We scale rows and pivot on magnitude. The full game certificate below
    // remains the acceptance test if a system has poor conditioning.
    for row in &mut a[..n] {
        let scale = row[..n].iter().fold(0.0_f64, |x, y| x.max(y.abs()));
        if !scale.is_finite() || scale == 0.0 {
            return None;
        }
        for x in &mut row[..=n] {
            *x /= scale;
        }
    }
    for col in 0..n {
        let mut pivot = col;
        for row in col + 1..n {
            if a[row][col].abs() > a[pivot][col].abs() {
                pivot = row;
            }
        }
        if !a[pivot][col].is_finite() || a[pivot][col].abs() < 1e-14 {
            return None;
        }
        a.swap(col, pivot);
        let source = a[col];
        for row in col + 1..n {
            let factor = a[row][col] / source[col];
            for k in col + 1..=n {
                a[row][k] = (-factor).mul_add(source[k], a[row][k]);
            }
        }
    }
    let mut g = [0.0; A];
    for j in (0..n).rev() {
        let mut x = a[j][n];
        for k in j + 1..n {
            x = (-a[j][k]).mul_add(g[k], x);
        }
        g[j] = x / a[j][j];
    }
    let mut q = [0.0; A];
    total = 0.0;
    for j in 0..A {
        if q_mask & (1 << j) == 0 {
            continue;
        }
        let mut x = 0.0;
        for k in 0..n {
            if positions[k] >= j {
                x = r[positions[k] - j].mul_add(g[k], x);
            }
        }
        if !x.is_finite() || x < 0.0 {
            return None;
        }
        q[j] = x;
        total += x;
    }
    if !total.is_finite() || total <= 0.0 {
        return None;
    }
    for x in &mut q {
        *x /= total;
    }
    Some(q)
}

fn support_attempt(
    s: &[f64; A],
    f: f64,
    r: &[f64; A],
    p_mask: u64,
    q_mask: u64,
    window: bool,
) -> Option<f64> {
    let q = reduced_weights(r, p_mask, q_mask)?;
    let reverse_p = reduced_weights(r, reverse_support(q_mask), reverse_support(p_mask))?;
    let p: [f64; A] = std::array::from_fn(|j| reverse_p[A - 1 - j]);
    let mut rows = [0.0; A];
    let mut cols = [0.0; A];
    let mut prefix_q = 0.0;
    let mut suffix_p = 0.0;
    for j in 0..A {
        rows[j] = f * prefix_q;
        cols[A - 1 - j] = f * suffix_p;
        prefix_q += q[j];
        suffix_p += p[A - 1 - j];
    }
    if (prefix_q - 1.0).abs() > 1e-12 || (suffix_p - 1.0).abs() > 1e-12 {
        return None;
    }
    // We check every row and column of the original matrix. We use its
    // constant failure triangle to avoid constructing 3,600 matrix cells.
    for i in 0..A {
        for j in i..A {
            rows[i] = s[j - i].mul_add(q[j], rows[i]);
            cols[j] = s[j - i].mul_add(p[i], cols[j]);
        }
    }
    if rows.iter().chain(cols.iter()).any(|x| !x.is_finite()) {
        return None;
    }
    let mut lower = cols.iter().copied().fold(f64::INFINITY, f64::min);
    let mut upper = rows.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let value = (lower + upper) * 0.5;
    if window {
        lower = lower.max(f);
        upper = upper.max(f);
    }
    if upper - lower < -1e-12 || upper - lower > GAP {
        return None;
    }
    Some(value)
}

fn shifted_isolated_holes(mask: u64, mode: usize) -> u64 {
    let mut result = mask;
    for i in 0..A {
        if mask & (1 << i) != 0 {
            continue;
        }
        if (i > 0 && mask & (1 << (i - 1)) == 0) || (i + 1 < A && mask & (1 << (i + 1)) == 0) {
            continue;
        }
        let direction = match mode {
            0 => {
                if i < A / 2 {
                    -1
                } else {
                    1
                }
            }
            1 => {
                if i < A / 2 {
                    1
                } else {
                    -1
                }
            }
            2 => -1,
            _ => 1,
        };
        let target = i as i32 + direction;
        if (0..A as i32).contains(&target) && mask & (1 << target) != 0 {
            result = (result | (1 << i)) & !(1 << target);
        }
    }
    result
}

fn support_neighbors(mask: u64) -> Vec<u64> {
    let mut neighbors = Vec::with_capacity(180);
    neighbors.push(mask);
    let mut i = 0;
    while i < A {
        if mask & (1 << i) != 0 {
            i += 1;
            continue;
        }
        let start = i;
        while i + 1 < A && mask & (1 << (i + 1)) == 0 {
            i += 1;
        }
        let end = i;
        if start > 0 {
            neighbors.push(mask ^ (1 << (start - 1)) ^ (1 << end));
        }
        if end + 1 < A {
            neighbors.push(mask ^ (1 << start) ^ (1 << (end + 1)));
        }
        i += 1;
    }
    for i in 0..A {
        let bit = (mask >> i) & 1;
        let boundary =
            (i > 0 && (mask >> (i - 1)) & 1 != bit) || (i + 1 < A && (mask >> (i + 1)) & 1 != bit);
        if boundary {
            neighbors.push(mask ^ (1 << i));
        }
        if i + 1 < A && (mask >> (i + 1)) & 1 != bit {
            neighbors.push(mask ^ (3 << i));
        }
    }
    neighbors
}

/// We try the preceding row support, then at most 64 paired boundary moves.
/// We stop after `limit` support attempts, counting the seed as the first.
fn solve_support(
    s: &[f64; A],
    f: f64,
    r: &[f64; A],
    seed: (u64, u64),
    window: bool,
    limit: usize,
) -> Option<(f64, u8, u64, u64)> {
    let (p, q) = seed;
    if limit == 0 || p == 0 || q == 0 || r.iter().any(|x| !x.is_finite()) {
        return None;
    }
    if let Some(value) = support_attempt(s, f, r, p, q, window) {
        return Some((value, 4, p, q));
    }
    let mut tried = [(0_u64, 0_u64); EDGE_LIMIT];
    let mut count = 0;
    for mode in 0..4 {
        let pair = (
            shifted_isolated_holes(p, mode),
            shifted_isolated_holes(q, mode),
        );
        if pair != seed && !tried[..count].contains(&pair) {
            if count + 1 >= limit {
                return None;
            }
            tried[count] = pair;
            count += 1;
            if let Some(value) = support_attempt(s, f, r, pair.0, pair.1, window) {
                return Some((value, 5, pair.0, pair.1));
            }
        }
    }
    for i in 0..A - 1 {
        if ((p >> i) ^ (p >> (i + 1))) & 1 != 0
            && ((q >> (A - 2 - i)) ^ (q >> (A - 1 - i))) & 1 != 0
        {
            let pair = (p ^ (3 << i), q ^ (3 << (A - 2 - i)));
            if tried[..count].contains(&pair) {
                continue;
            }
            if count + 1 >= limit {
                return None;
            }
            tried[count] = pair;
            count += 1;
            if let Some(value) = support_attempt(s, f, r, pair.0, pair.1, window) {
                return Some((value, 5, pair.0, pair.1));
            }
        }
    }
    let ps = support_neighbors(p);
    let qs = support_neighbors(q);
    for pp in ps {
        for &qq in &qs {
            if (pp, qq) == seed
                || pp.count_ones() != qq.count_ones()
                || tried[..count].contains(&(pp, qq))
            {
                continue;
            }
            if count == EDGE_LIMIT || count + 1 >= limit {
                return None;
            }
            tried[count] = (pp, qq);
            count += 1;
            if let Some(value) = support_attempt(s, f, r, pp, qq, window) {
                return Some((value, 5, pp, qq));
            }
        }
    }
    None
}

/// We return the index `k` of the most negative success-payoff step
/// `s[k] - s[k-1]`, the first on a tie, or `None` when no step is negative.
/// A nondecreasing `s` with `f > s[0]` gives nonnegative recurrence weights,
/// so a residue class needs at least one negative step.
fn kink_index(s: &[f64; A]) -> Option<usize> {
    let mut best = 0.0;
    let mut index = None;
    for k in 1..A {
        let step = s[k] - s[k - 1];
        if step < best {
            best = step;
            index = Some(k);
        }
    }
    index
}

/// We move each hole of `mask` with the nearer of two kink anchors. The first
/// anchor is `near` (the kink `old` for the Dropper, `old - 1` for the
/// Checker); the second is `59 - old`. A hole at the first anchor moves by
/// `delta`, and a hole at the second moves by `-delta`. A hole that starts at
/// action 0 keeps its place. We clip moved holes to actions 0..59.
fn move_holes(mask: u64, near: i64, old: i64, delta: i64) -> u64 {
    let far = A as i64 - 1 - old;
    let mut out = SUPPORT_BITS;
    let mut i = 0;
    while i < A {
        if (mask >> i) & 1 == 1 {
            i += 1;
            continue;
        }
        let start = i;
        while i + 1 < A && (mask >> (i + 1)) & 1 == 0 {
            i += 1;
        }
        let end = i as i64;
        let shift = if start == 0 {
            0
        } else if (end - near).abs() <= (end - far).abs() {
            delta
        } else {
            -delta
        };
        let low = (start as i64 + shift).max(0);
        let high = (end + shift).min(A as i64 - 1);
        for x in low..=high {
            out &= !(1 << x);
        }
        i += 1;
    }
    out
}

/// We express a certified support relative to its kink and move it to a new
/// kink. We return the masks unchanged when either kink is absent or the two
/// kinks agree.
fn kink_seed(p: u64, q: u64, old: Option<usize>, new: Option<usize>) -> (u64, u64) {
    match (old, new) {
        (Some(old), Some(new)) if old != new => {
            let (old, delta) = (old as i64, new as i64 - old as i64);
            (
                move_holes(p, old, old, delta),
                move_holes(q, old - 1, old, delta),
            )
        }
        _ => (p, q),
    }
}

/// We solve one residue stage in the worker. We first try the moved seed
/// support and at most `kink_attempts` supports. We then run the packing LP.
/// We return the square-stage value and its kind, or `None` for HiGHS. Each
/// accepted support or LP basis becomes the next seed.
fn solve_residue(
    stage: &[f64; A + 1],
    r: &[f64; A],
    window: bool,
    crash: bool,
    kink_attempts: usize,
    seed: &mut Option<(u64, u64, Option<usize>)>,
    packer: &mut crate::leap_packing::Solver,
) -> Option<(f64, u8)> {
    let s: &[f64; A] = stage[..A].try_into().unwrap();
    let f = stage[A];
    let kink = kink_index(s);
    if let Some((p, q, old)) = *seed {
        let guess = kink_seed(p, q, old, kink);
        if let Some((value, kind, p, q)) = solve_support(s, f, r, guess, window, kink_attempts) {
            *seed = Some((p, q, kink));
            return Some((value, kind));
        }
    }
    let solved = if crash {
        packer.solve_crash(stage)
    } else {
        packer.solve(stage, false)
    };
    if !solved[0].is_finite() {
        return None;
    }
    *seed = Some((packer.support.0, packer.support.1, kink));
    Some((solved[0], 3))
}

/// We expose the kink index for parity tests against `leap_support.py`.
#[pyfunction]
pub fn kink_index_rs(s: PyReadonlyArray1<'_, f64>) -> PyResult<Option<usize>> {
    let s: [f64; A] = s
        .as_slice()?
        .try_into()
        .map_err(|_| PyValueError::new_err("kink index needs 60 payoffs"))?;
    if s.iter().any(|x| !x.is_finite()) {
        return Err(PyValueError::new_err("kink index needs finite payoffs"));
    }
    Ok(kink_index(&s))
}

/// We expose the kink seed move for parity tests against `leap_support.py`.
#[pyfunction]
#[pyo3(signature = (p, q, old, new))]
pub fn kink_seed_rs(
    p: u64,
    q: u64,
    old: Option<usize>,
    new: Option<usize>,
) -> PyResult<(u64, u64)> {
    if (p | q) & !SUPPORT_BITS != 0
        || old.is_some_and(|k| k == 0 || k >= A)
        || new.is_some_and(|k| k == 0 || k >= A)
    {
        return Err(PyValueError::new_err(
            "kink seed needs actions 0..59 and kinks 1..59",
        ));
    }
    Ok(kink_seed(p, q, old, new))
}
