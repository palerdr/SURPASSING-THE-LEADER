/* Adapted from palerdr/dth at 1ef73c93. We retain a numerical full-matrix
   certificate for all accepted recurrence policies. No game rules live here. */
#include "dth.hpp"
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#if defined(__FAST_MATH__) || FLT_RADIX != 2 || DBL_MANT_DIG != 53 || DBL_MAX_EXP != 1024
#error "Require IEEE binary64 without fast-math"
#endif
#define N 60
#define L 16
#define GAP 1e-6

/* Return unresolved count, or -1 on invalid children/environment. */
namespace dth {
int64_t solve_recurrence_chunk(int64_t n, const ProfileId* pcs, const ProfileId* pds, double* V,
                               uint8_t* kind, int64_t size, const int32_t* succ,
                               const int32_t* fail, const double* rev) {
    volatile double tiny = DBL_MIN;
    if (fegetround() != FE_TONEAREST || tiny * 0.5 == 0.0)
        return -1;
    double S[N][L], R[N][L], Q[N][L];
    double F[L], D[L], MN[L], MX[L], SUM[L], CUM[L];
    int64_t rejected = 0;
    for (int64_t base = 0; base < n; base += L) {
        int m = (int)(n - base < L ? n - base : L);
        for (int l = 0; l < L; ++l) {
            int64_t i = base + (l < m ? l : 0);
            if (pcs[i] >= size || pds[i] >= size)
                return -1;
            const double* row = V + (int64_t)pds[i] * size;
            const int32_t* ch = succ + (int64_t)pcs[i] * N;
            double lo = INFINITY, hi = -INFINITY;
            for (int k = 0; k < N; ++k) {
                if (ch[k] < -1 || ch[k] >= size)
                    return -1;
                double s = ch[k] == -1 ? 1.0 : -row[ch[k]];
                if (!isfinite(s) || fabs(s) > 1.0 + 1e-9)
                    return -1;
                S[k][l] = s;
                lo = fmin(lo, s);
                hi = fmax(hi, s);
            }
            int32_t child = fail[pcs[i]];
            double p = rev[pcs[i]];
            if (child < -1 || child >= size || !(p >= 0.0 && p <= 1.0))
                return -1;
            F[l] = child == -1 ? 1.0 : p * (-row[child]) + (1.0 - p);
            if (!isfinite(F[l]) || fabs(F[l]) > 1.0 + 1e-9)
                return -1;
            MN[l] = fmax(lo, fmin(F[l], S[0][l]));
            MX[l] = fmin(hi, fmax(F[l], S[0][l]));
            D[l] = S[0][l] - F[l];
        }
        for (int k = 1; k < N; ++k)
            for (int l = 0; l < L; ++l)
                Q[k][l] = (S[k - 1][l] - S[k][l]) / (D[l] != 0.0 ? D[l] : 1.0);
        for (int l = 0; l < L; ++l)
            R[0][l] = 1.0;
        for (int k = 1; k < N; ++k) {
            double acc[L] = {0};
            for (int j = 0; j < k; ++j)
                for (int l = 0; l < L; ++l)
                    acc[l] += Q[k - j][l] * R[j][l];
            for (int l = 0; l < L; ++l)
                R[k][l] = acc[l];
        }
        for (int l = 0; l < L; ++l)
            SUM[l] = 0.0;
        for (int k = 0; k < N; ++k)
            for (int l = 0; l < L; ++l) {
                double r = R[N - 1 - k][l];
                Q[k][l] = isfinite(r) ? fmax(r, 0.0) : NAN;
                SUM[l] += Q[k][l];
            }
        double lower[L], upper[L];
        for (int l = 0; l < L; ++l) {
            lower[l] = INFINITY;
            upper[l] = -INFINITY;
            CUM[l] = 0.0;
        }
        for (int k = 0; k < N; ++k)
            for (int l = 0; l < L; ++l)
                Q[k][l] /= SUM[l];
        /* For p = reverse(q), (p^T M)[j] = (M q)[59-j]. We evaluate all
           60 pure deviations on both sides through this identity. */
        for (int k = 0; k < N; ++k) {
            double x[L] = {0};
            for (int j = 0; j < N - k; ++j)
                for (int l = 0; l < L; ++l)
                    x[l] += S[j][l] * Q[k + j][l];
            for (int l = 0; l < L; ++l) {
                x[l] += F[l] * CUM[l];
                if (!isfinite(x[l]))
                    SUM[l] = NAN;
                lower[l] = fmin(lower[l], x[l]);
                upper[l] = fmax(upper[l], x[l]);
                CUM[l] += Q[k][l];
            }
        }
        for (int l = 0; l < m; ++l) {
            double v;
            uint8_t route;
            if (MX[l] - MN[l] <= GAP) {
                v = (MN[l] + MX[l]) * 0.5;
                route = 0;
            } else if (fabs(D[l]) >= 1e-12 && isfinite(SUM[l]) && SUM[l] > 0.0 &&
                       fabs(CUM[l] - 1.0) <= 1e-12 && upper[l] - lower[l] <= GAP) {
                v = (lower[l] + upper[l]) * 0.5;
                route = 1;
            } else {
                ++rejected;
                continue;
            }
            if (!isfinite(v) || fabs(v) > 1.0 + 1e-9)
                return -1;
            int64_t index = (int64_t)pcs[base + l] * size + pds[base + l];
            V[index] = v;
            kind[index] = route;
        }
    }
    return rejected;
}

} // namespace dth
