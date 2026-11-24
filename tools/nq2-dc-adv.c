/* nq2-dc-adv.c ... NanoQuant Dictionary Compute, adversarial suite.
 *
 * Implemented by Elfie (E.L.F.I.E.). Direction and executive
 * planning by JD. Companion to tools/nq2-dc.c (the baseline
 * probe). JD's charge, 2026-11-24: run every kind of test, try
 * to break it, run adversarial tests, test the hashes.
 *
 * Each fixture states what would count as breaking the method,
 * then measures. Verdicts print as HOLD or BREAK with the
 * numbers, so a failure is as visible as a pass.
 *
 * Plain hosted C99. Build: cc -O2 -std=c99 -o nq2-dc-adv
 * tools/nq2-dc-adv.c -lm
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

static uint64_t rng_state = 42;
static uint64_t splitmix(void)
{
    uint64_t z = (rng_state += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
static double randn(void)
{
    double u1 = ((splitmix() >> 11) + 1) / 9007199254740992.0;
    double u2 = (splitmix() >> 11) / 9007199254740992.0;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}
static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
static uint64_t fnv1a(const void *p, size_t n)
{
    const uint8_t *b = (const uint8_t *)p;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}
static double rel_l2(const double *a, const double *b, uint64_t n)
{
    double d2 = 0, r2 = 0;
    for (uint64_t i = 0; i < n; i++) {
        /* Gizmo's review, 2026-11-24: nonfinite values poison
         * the comparison explicitly; they never slip through. */
        if (!isfinite(a[i]) || !isfinite(b[i])) return NAN;
        double d = a[i] - b[i];
        d2 += d * d; r2 += b[i] * b[i];
    }
    return r2 > 0 ? sqrt(d2 / r2) : (d2 > 0 ? 1e30 : 0);
}

enum { B = 256, K = 256 };
typedef struct {
    int M, J;                 /* rows, segments; N = J * B */
    float *C, *sc, *mn, *x;
    uint8_t *idx;
} Fx;

static void fx_alloc(Fx *f, int M, int J)
{
    f->M = M; f->J = J;
    f->C = (float *)malloc((size_t)K * B * 4);
    f->sc = (float *)malloc((size_t)M * J * 4);
    f->mn = (float *)malloc((size_t)M * J * 4);
    f->idx = (uint8_t *)malloc((size_t)M * J);
    f->x = (float *)malloc((size_t)J * B * 4);
}
static void fx_free(Fx *f)
{
    free(f->C); free(f->sc); free(f->mn); free(f->idx); free(f->x);
}
static void fx_random(Fx *f, uint64_t seed, double proto_sd,
                      double sc_lo, double sc_hi, double mn_sd)
{
    rng_state = seed;
    for (int i = 0; i < K * B; i++) f->C[i] = (float)(randn() * proto_sd);
    for (int i = 0; i < f->M * f->J; i++) {
        f->idx[i] = (uint8_t)(splitmix() & 0xff);
        f->sc[i] = (float)(sc_lo + (sc_hi - sc_lo)
                     * ((splitmix() >> 8) % 1000) / 1000.0);
        f->mn[i] = (float)(randn() * mn_sd);
    }
    for (int i = 0; i < f->J * B; i++) f->x[i] = (float)randn();
}
static void oracle(const Fx *f, double *y)
{
    for (int r = 0; r < f->M; r++) {
        double acc = 0;
        for (int j = 0; j < f->J; j++) {
            const float *proto = f->C + (size_t)f->idx[r * f->J + j] * B;
            double s = f->sc[r * f->J + j], m = f->mn[r * f->J + j];
            for (int i = 0; i < B; i++)
                acc += (s * proto[i] + m) * f->x[j * B + i];
        }
        y[r] = acc;
    }
}
static void recon(const Fx *f, float *y)
{
    for (int r = 0; r < f->M; r++) {
        float acc = 0;
        for (int j = 0; j < f->J; j++) {
            const float *proto = f->C + (size_t)f->idx[r * f->J + j] * B;
            float s = f->sc[r * f->J + j], m = f->mn[r * f->J + j];
            for (int i = 0; i < B; i++)
                acc += (m + s * proto[i]) * f->x[j * B + i];
        }
        y[r] = acc;
    }
}
static void dc_build(const Fx *f, float *T, float *A)
{
    for (int j = 0; j < f->J; j++) {
        float a = 0;
        for (int i = 0; i < B; i++) a += f->x[j * B + i];
        A[j] = a;
        for (int k = 0; k < K; k++) {
            float acc = 0;
            const float *proto = f->C + (size_t)k * B;
            for (int i = 0; i < B; i++) acc += proto[i] * f->x[j * B + i];
            T[j * K + k] = acc;
        }
    }
}
static void dc_apply(const Fx *f, const float *T, const float *A, float *y)
{
    for (int r = 0; r < f->M; r++) {
        float acc = 0;
        for (int j = 0; j < f->J; j++)
            acc += f->sc[r * f->J + j] * T[j * K + f->idx[r * f->J + j]]
                 + f->mn[r * f->J + j] * A[j];
        y[r] = acc;
    }
}
static void dc(const Fx *f, float *y)
{
    float *T = (float *)malloc((size_t)f->J * K * 4);
    float *A = (float *)malloc((size_t)f->J * 4);
    dc_build(f, T, A);
    dc_apply(f, T, A, y);
    free(T); free(A);
}
static double dc_vs_oracle(Fx *f)   /* helper: returns rel L2 */
{
    double *yo = (double *)malloc((size_t)f->M * 8);
    float *yd = (float *)malloc((size_t)f->M * 4);
    double *dd = (double *)malloc((size_t)f->M * 8);
    oracle(f, yo); dc(f, yd);
    for (int r = 0; r < f->M; r++) dd[r] = yd[r];
    double v = rel_l2(dd, yo, f->M);
    free(yo); free(yd); free(dd);
    return v;
}
static void verdict(const char *name, int hold, const char *detail)
{
    printf("  [%s] %-46s %s\n", hold ? "HOLD" : "BREAK", name, detail);
}

int main(void)
{
    char buf[256];
    printf("NQDC adversarial suite (B=%d K=%d)\n", B, K);

    /* 0. comparator self-tests (Gizmo's review, 2026-11-24):
     * the measuring instrument itself goes on trial first. */
    {
        double zero[8] = { 0 }, ones[8], withnan[8] = { 0 };
        for (int i = 0; i < 8; i++) ones[i] = 1.0;
        withnan[3] = NAN;
        double v_bad = rel_l2(ones, zero, 8);
        double v_ok = rel_l2(zero, zero, 8);
        double v_nan = rel_l2(withnan, zero, 8);
        snprintf(buf, sizeof buf, "wrong-vs-zero %.3e (must be "
                 "huge), zero-vs-zero %.1f, NaN candidate %s",
                 v_bad, v_ok, isnan(v_nan) ? "scores NaN" : "PASSED?!");
        verdict("comparator: no false success at zero reference",
                v_bad >= 1e29 && v_ok == 0 && isnan(v_nan)
                && !(v_nan < 1e-5), buf);
    }
    /* 0b. zero representation: oracle and DC both exactly zero */
    {
        Fx f; fx_alloc(&f, 64, 4); fx_random(&f, 10, 0.1, 0.01, 0.03, 0.005);
        for (int i = 0; i < K * B; i++) f.C[i] = 0;
        for (int i = 0; i < f.M * f.J; i++) f.mn[i] = 0;
        double v = dc_vs_oracle(&f);
        snprintf(buf, sizeof buf, "rel L2 %.1f at a zero reference "
                 "(agreement, not false success)", v);
        verdict("zero representation agrees at zero", v == 0, buf);
        fx_free(&f);
    }

    /* 1. delta activation: output must be the touched column */
    {
        Fx f; fx_alloc(&f, 128, 4); fx_random(&f, 11, 0.1, 0.01, 0.03, 0.005);
        for (int i = 0; i < f.J * B; i++) f.x[i] = 0;
        f.x[2 * B + 37] = 1.0f;
        double v = dc_vs_oracle(&f);
        snprintf(buf, sizeof buf, "rel L2 vs oracle %.3e", v);
        verdict("delta activation", v < 1e-6, buf);
        fx_free(&f);
    }
    /* 2. zero activation: output bitwise zero */
    {
        Fx f; fx_alloc(&f, 128, 4); fx_random(&f, 12, 0.1, 0.01, 0.03, 0.005);
        for (int i = 0; i < f.J * B; i++) f.x[i] = 0;
        float *y = (float *)malloc((size_t)f.M * 4);
        dc(&f, y);
        int nz = 0;
        for (int r = 0; r < f.M; r++) if (y[r] != 0.0f) nz++;
        snprintf(buf, sizeof buf, "nonzero outputs: %d of %d", nz, f.M);
        verdict("zero activation gives exact zero", nz == 0, buf);
        free(y); fx_free(&f);
    }
    /* 3. constant activation */
    {
        Fx f; fx_alloc(&f, 128, 4); fx_random(&f, 13, 0.1, 0.01, 0.03, 0.005);
        for (int i = 0; i < f.J * B; i++) f.x[i] = 3.25f;
        double v = dc_vs_oracle(&f);
        snprintf(buf, sizeof buf, "rel L2 vs oracle %.3e", v);
        verdict("constant activation", v < 1e-5, buf);
        fx_free(&f);
    }
    /* 4. extreme dynamic range in x */
    {
        Fx f; fx_alloc(&f, 128, 4); fx_random(&f, 14, 0.1, 0.01, 0.03, 0.005);
        rng_state = 99;
        for (int i = 0; i < f.J * B; i++) {
            int e = (int)(splitmix() % 37) - 18;
            f.x[i] = (float)(randn() * pow(10.0, e));
        }
        double v = dc_vs_oracle(&f);
        snprintf(buf, sizeof buf, "rel L2 vs oracle %.3e (10^+/-18 mix)", v);
        verdict("extreme dynamic range activation", v < 1e-2, buf);
        fx_free(&f);
    }
    /* 5. cancellation stress: huge cancelling terms in T */
    {
        Fx f; fx_alloc(&f, 128, 4); fx_random(&f, 15, 100.0, 0.01, 0.03, 0.005);
        for (int i = 0; i < f.J * B; i++)
            f.x[i] = (i % 2 ? -1.0f : 1.0f) * 1.0e4f;
        double v = dc_vs_oracle(&f);
        snprintf(buf, sizeof buf, "rel L2 vs oracle %.3e", v);
        verdict("cancellation stress (proto sd 100, x +/-1e4)", v < 1e-2, buf);
        fx_free(&f);
    }
    /* 6. NaN in activation: both paths poison identically */
    {
        Fx f; fx_alloc(&f, 64, 4); fx_random(&f, 16, 0.1, 0.01, 0.03, 0.005);
        f.x[5] = NAN;
        float *yd = (float *)malloc((size_t)f.M * 4);
        float *yr = (float *)malloc((size_t)f.M * 4);
        dc(&f, yd); recon(&f, yr);
        int nd = 0, nr = 0;
        for (int r = 0; r < f.M; r++) {
            if (!isfinite(yd[r])) nd++;
            if (!isfinite(yr[r])) nr++;
        }
        snprintf(buf, sizeof buf, "nonfinite outputs: dc %d, recon %d, "
                 "of %d", nd, nr, f.M);
        verdict("NaN poisons both paths identically", nd == nr && nd > 0, buf);
        free(yd); free(yr); fx_free(&f);
    }
    /* 7. zero scale: constant blocks */
    {
        Fx f; fx_alloc(&f, 128, 4); fx_random(&f, 17, 0.1, 0.01, 0.03, 0.005);
        for (int i = 0; i < f.M * f.J; i++) f.sc[i] = 0;
        double v = dc_vs_oracle(&f);
        snprintf(buf, sizeof buf, "rel L2 vs oracle %.3e", v);
        verdict("zero-scale blocks (pure offset)", v < 1e-6, buf);
        fx_free(&f);
    }
    /* 8. extreme frames: fp16-max scale and offset */
    {
        Fx f; fx_alloc(&f, 128, 4); fx_random(&f, 18, 0.1, 0.01, 0.03, 0.005);
        for (int i = 0; i < f.M * f.J; i++) {
            f.sc[i] = 65504.0f; f.mn[i] = -65504.0f;
        }
        double v = dc_vs_oracle(&f);
        snprintf(buf, sizeof buf, "rel L2 vs oracle %.3e", v);
        verdict("fp16-max frames", v < 1e-4, buf);
        fx_free(&f);
    }
    /* 9/10. reuse extremes under lazy build: timing contrast */
    {
        Fx f; fx_alloc(&f, 1024, 12);
        fx_random(&f, 19, 0.1, 0.01, 0.03, 0.005);
        float *T = (float *)malloc((size_t)f.J * K * 4);
        float *A = (float *)malloc((size_t)f.J * 4);
        float *y = (float *)malloc((size_t)f.M * 4);
        uint8_t *built = (uint8_t *)malloc((size_t)f.J * K);
        /* all rows use prototype 7 in every segment */
        for (int i = 0; i < f.M * f.J; i++) f.idx[i] = 7;
        double best = 1e18;
        for (int rep = 0; rep < 15; rep++) {
            memset(built, 0, (size_t)f.J * K);
            dc_build(&f, T, A);   /* A only; T rebuilt lazily below */
            memset(T, 0, (size_t)f.J * K * 4);
            double t0 = now_ms();
            for (int r = 0; r < f.M; r++) {
                float acc = 0;
                for (int j = 0; j < f.J; j++) {
                    int k = f.idx[r * f.J + j];
                    if (!built[j * K + k]) {
                        float a2 = 0;
                        const float *proto = f.C + (size_t)k * B;
                        for (int i = 0; i < B; i++)
                            a2 += proto[i] * f.x[j * B + i];
                        T[j * K + k] = a2; built[j * K + k] = 1;
                    }
                    acc += f.sc[r * f.J + j] * T[j * K + k]
                         + f.mn[r * f.J + j] * A[j];
                }
                y[r] = acc;
            }
            double dt = now_ms() - t0;
            if (dt < best) best = dt;
        }
        double v = dc_vs_oracle(&f);
        snprintf(buf, sizeof buf, "lazy max-reuse %.3f ms, rel L2 %.2e",
                 best, v);
        verdict("max reuse (one prototype per segment)", v < 1e-5, buf);
        /* min reuse: indices cycle through all K in each segment */
        for (int r = 0; r < f.M; r++)
            for (int j = 0; j < f.J; j++)
                f.idx[r * f.J + j] = (uint8_t)((r + j) & 0xff);
        best = 1e18;
        for (int rep = 0; rep < 15; rep++) {
            memset(built, 0, (size_t)f.J * K);
            memset(T, 0, (size_t)f.J * K * 4);
            double t0 = now_ms();
            for (int r = 0; r < f.M; r++) {
                float acc = 0;
                for (int j = 0; j < f.J; j++) {
                    int k = f.idx[r * f.J + j];
                    if (!built[j * K + k]) {
                        float a2 = 0;
                        const float *proto = f.C + (size_t)k * B;
                        for (int i = 0; i < B; i++)
                            a2 += proto[i] * f.x[j * B + i];
                        T[j * K + k] = a2; built[j * K + k] = 1;
                    }
                    acc += f.sc[r * f.J + j] * T[j * K + k]
                         + f.mn[r * f.J + j] * A[j];
                }
                y[r] = acc;
            }
            double dt = now_ms() - t0;
            if (dt < best) best = dt;
        }
        v = dc_vs_oracle(&f);
        snprintf(buf, sizeof buf, "lazy min-reuse %.3f ms, rel L2 %.2e",
                 best, v);
        verdict("min reuse (full book per segment)", v < 1e-5, buf);
        free(T); free(A); free(y); free(built); fx_free(&f);
    }
    /* 11. single row: correctness when reuse cannot pay */
    {
        Fx f; fx_alloc(&f, 1, 12); fx_random(&f, 20, 0.1, 0.01, 0.03, 0.005);
        double v = dc_vs_oracle(&f);
        snprintf(buf, sizeof buf, "rel L2 vs oracle %.3e at M=1", v);
        verdict("single row (M=1)", v < 1e-5, buf);
        fx_free(&f);
    }
    /* 12. batch of 8 activations sharing tables */
    {
        Fx f; fx_alloc(&f, 512, 12);
        fx_random(&f, 21, 0.1, 0.01, 0.03, 0.005);
        const int A8 = 8;
        float *xs = (float *)malloc((size_t)A8 * f.J * B * 4);
        rng_state = 77;
        for (int a = 0; a < A8; a++)
            for (int i = 0; i < f.J * B; i++)
                xs[a * f.J * B + i] = (float)randn();
        double worst = 0;
        float *T = (float *)malloc((size_t)f.J * K * A8 * 4);
        float *Av = (float *)malloc((size_t)f.J * A8 * 4);
        float *y = (float *)malloc((size_t)f.M * 4);
        double *yo = (double *)malloc((size_t)f.M * 8);
        double *dd = (double *)malloc((size_t)f.M * 8);
        for (int a = 0; a < A8; a++) {
            memcpy(f.x, xs + a * f.J * B, (size_t)f.J * B * 4);
            oracle(&f, yo);
            for (int j = 0; j < f.J; j++) {
                float av = 0;
                for (int i = 0; i < B; i++) av += f.x[j * B + i];
                Av[j * A8 + a] = av;
                for (int k = 0; k < K; k++) {
                    float acc = 0;
                    const float *proto = f.C + (size_t)k * B;
                    for (int i = 0; i < B; i++)
                        acc += proto[i] * f.x[j * B + i];
                    T[(j * K + k) * A8 + a] = acc;
                }
            }
            for (int r = 0; r < f.M; r++) {
                float acc = 0;
                for (int j = 0; j < f.J; j++)
                    acc += f.sc[r * f.J + j]
                         * T[(j * K + f.idx[r * f.J + j]) * A8 + a]
                         + f.mn[r * f.J + j] * Av[j * A8 + a];
                y[r] = acc; dd[r] = y[r];
            }
            double v = rel_l2(dd, yo, f.M);
            if (v > worst) worst = v;
        }
        snprintf(buf, sizeof buf, "worst rel L2 over 8 columns %.3e", worst);
        verdict("batched activations (8 columns)", worst < 1e-5, buf);
        free(xs); free(T); free(Av); free(y); free(yo); free(dd);
        fx_free(&f);
    }
    /* 13. PQ d8: correct shared-frame form vs the offset trap */
    {
        const int d = 8, P = B / d;
        Fx f; fx_alloc(&f, 256, 4);
        fx_random(&f, 22, 0.1, 0.01, 0.03, 0.02);
        /* PQ books: P subspaces x K entries x d */
        float *Cp = (float *)malloc((size_t)P * K * d * 4);
        uint8_t *qp = (uint8_t *)malloc((size_t)f.M * f.J * P);
        rng_state = 55;
        for (int i = 0; i < P * K * d; i++) Cp[i] = (float)(randn() * 0.1);
        for (int i = 0; i < f.M * f.J * P; i++)
            qp[i] = (uint8_t)(splitmix() & 0xff);
        /* oracle over PQ reconstruction */
        double *yo = (double *)malloc((size_t)f.M * 8);
        for (int r = 0; r < f.M; r++) {
            double acc = 0;
            for (int j = 0; j < f.J; j++) {
                double s = f.sc[r * f.J + j], m = f.mn[r * f.J + j];
                for (int p = 0; p < P; p++) {
                    const float *proto = Cp + ((size_t)p * K
                        + qp[(r * f.J + j) * P + p]) * d;
                    for (int i = 0; i < d; i++)
                        acc += (s * proto[i] + m)
                             * f.x[j * B + p * d + i];
                }
            }
            yo[r] = acc;
        }
        /* correct DC form: offset counted once per block */
        float *Tp = (float *)malloc((size_t)f.J * P * K * 4);
        float *Av = (float *)malloc((size_t)f.J * 4);
        double *yc = (double *)malloc((size_t)f.M * 8);
        double *yn = (double *)malloc((size_t)f.M * 8);
        for (int j = 0; j < f.J; j++) {
            float av = 0;
            for (int i = 0; i < B; i++) av += f.x[j * B + i];
            Av[j] = av;
            for (int p = 0; p < P; p++)
                for (int k = 0; k < K; k++) {
                    float acc = 0;
                    const float *proto = Cp + ((size_t)p * K + k) * d;
                    for (int i = 0; i < d; i++)
                        acc += proto[i] * f.x[j * B + p * d + i];
                    Tp[(j * P + p) * K + k] = acc;
                }
        }
        for (int r = 0; r < f.M; r++) {
            float acc = 0, accn = 0;
            for (int j = 0; j < f.J; j++) {
                float inner = 0;
                for (int p = 0; p < P; p++)
                    inner += Tp[(j * P + p) * K
                                + qp[(r * f.J + j) * P + p]];
                acc += f.sc[r * f.J + j] * inner
                     + f.mn[r * f.J + j] * Av[j];
                accn += f.sc[r * f.J + j] * inner
                      + P * f.mn[r * f.J + j] * Av[j];
            }
            yc[r] = acc; yn[r] = accn;
        }
        double vc = rel_l2(yc, yo, f.M), vn = rel_l2(yn, yo, f.M);
        snprintf(buf, sizeof buf, "correct form rel L2 %.3e", vc);
        verdict("PQ d8 shared frame (offset once)", vc < 1e-5, buf);
        snprintf(buf, sizeof buf, "naive form rel L2 %.3e (must be large)",
                 vn);
        verdict("PQ offset trap is real (naive form fails)", vn > 0.05, buf);
        free(Cp); free(qp); free(yo); free(Tp); free(Av); free(yc); free(yn);
        fx_free(&f);
    }
    /* 14. two-stage residual: stage tables sum */
    {
        Fx f; fx_alloc(&f, 256, 4);
        fx_random(&f, 23, 0.1, 0.01, 0.03, 0.005);
        float *C2 = (float *)malloc((size_t)K * B * 4);
        uint8_t *q2 = (uint8_t *)malloc((size_t)f.M * f.J);
        rng_state = 66;
        for (int i = 0; i < K * B; i++) C2[i] = (float)(randn() * 0.02);
        for (int i = 0; i < f.M * f.J; i++) q2[i] = (uint8_t)(splitmix() & 0xff);
        double *yo = (double *)malloc((size_t)f.M * 8);
        for (int r = 0; r < f.M; r++) {
            double acc = 0;
            for (int j = 0; j < f.J; j++) {
                double s = f.sc[r * f.J + j], m = f.mn[r * f.J + j];
                const float *p1 = f.C + (size_t)f.idx[r * f.J + j] * B;
                const float *p2 = C2 + (size_t)q2[r * f.J + j] * B;
                for (int i = 0; i < B; i++)
                    acc += (s * (p1[i] + p2[i]) + m) * f.x[j * B + i];
            }
            yo[r] = acc;
        }
        float *T1 = (float *)malloc((size_t)f.J * K * 4);
        float *T2 = (float *)malloc((size_t)f.J * K * 4);
        float *Av = (float *)malloc((size_t)f.J * 4);
        for (int j = 0; j < f.J; j++) {
            float av = 0;
            for (int i = 0; i < B; i++) av += f.x[j * B + i];
            Av[j] = av;
            for (int k = 0; k < K; k++) {
                float a1 = 0, a2 = 0;
                for (int i = 0; i < B; i++) {
                    a1 += f.C[(size_t)k * B + i] * f.x[j * B + i];
                    a2 += C2[(size_t)k * B + i] * f.x[j * B + i];
                }
                T1[j * K + k] = a1; T2[j * K + k] = a2;
            }
        }
        double *yd = (double *)malloc((size_t)f.M * 8);
        for (int r = 0; r < f.M; r++) {
            float acc = 0;
            for (int j = 0; j < f.J; j++)
                acc += f.sc[r * f.J + j]
                     * (T1[j * K + f.idx[r * f.J + j]]
                        + T2[j * K + q2[r * f.J + j]])
                     + f.mn[r * f.J + j] * Av[j];
            yd[r] = acc;
        }
        double v = rel_l2(yd, yo, f.M);
        snprintf(buf, sizeof buf, "rel L2 vs oracle %.3e", v);
        verdict("two-stage residual dictionaries", v < 1e-5, buf);
        free(C2); free(q2); free(yo); free(T1); free(T2); free(Av); free(yd);
        fx_free(&f);
    }
    /* 15. rotation identity, end to end */
    {
        /* R: fp64 Gram-Schmidt on a random matrix */
        double *R = (double *)malloc((size_t)B * B * 8);
        rng_state = 31;
        for (int i = 0; i < B * B; i++) R[i] = randn();
        for (int c = 0; c < B; c++) {
            for (int p = 0; p < c; p++) {
                double d = 0;
                for (int i = 0; i < B; i++) d += R[i * B + c] * R[i * B + p];
                for (int i = 0; i < B; i++) R[i * B + c] -= d * R[i * B + p];
            }
            double n = 0;
            for (int i = 0; i < B; i++) n += R[i * B + c] * R[i * B + c];
            n = sqrt(n);
            for (int i = 0; i < B; i++) R[i * B + c] /= n;
        }
        Fx f; fx_alloc(&f, 128, 2);
        fx_random(&f, 24, 0.1, 0.01, 0.03, 0.005);
        /* rotated book: Cz = R * C (columns are prototypes) */
        float *Cz = (float *)malloc((size_t)K * B * 4);
        for (int k = 0; k < K; k++)
            for (int i = 0; i < B; i++) {
                double acc = 0;
                for (int t = 0; t < B; t++)
                    acc += R[i * B + t] * f.C[(size_t)k * B + t];
                Cz[(size_t)k * B + i] = (float)acc;
            }
        /* oracle on decoded weights w = s * R^T Cz[idx] + m */
        double *yo = (double *)malloc((size_t)f.M * 8);
        for (int r = 0; r < f.M; r++) {
            double acc = 0;
            for (int j = 0; j < f.J; j++) {
                double s = f.sc[r * f.J + j], m = f.mn[r * f.J + j];
                const float *cz = Cz + (size_t)f.idx[r * f.J + j] * B;
                for (int i = 0; i < B; i++) {
                    double w = m;
                    double dec = 0;
                    for (int t = 0; t < B; t++)
                        dec += R[t * B + i] * cz[t];   /* (R^T cz)[i] */
                    w += s * dec;
                    acc += w * f.x[j * B + i];
                }
            }
            yo[r] = acc;
        }
        /* DC path: tables over Cz against Rx; offset uses raw x */
        /* T4 (CODE-REVIEW-2026-11-24): calloc, not malloc: yd is
         * accumulated into below, and the old malloc read relied
         * on fresh heap pages happening to be zero. */
        double *yd = (double *)calloc((size_t)f.M, 8);
        for (int j = 0; j < f.J; j++) {
            double rx[B];
            for (int i = 0; i < B; i++) {
                double acc = 0;
                for (int t = 0; t < B; t++)
                    acc += R[i * B + t] * f.x[j * B + t];  /* (Rx)[i] */
                rx[i] = acc;
            }
            float Tj[K];
            for (int k = 0; k < K; k++) {
                double acc = 0;
                const float *cz = Cz + (size_t)k * B;
                for (int i = 0; i < B; i++) acc += cz[i] * rx[i];
                Tj[k] = (float)acc;
            }
            double av = 0;
            for (int i = 0; i < B; i++) av += f.x[j * B + i];
            for (int r = 0; r < f.M; r++)
                yd[r] += f.sc[r * f.J + j] * Tj[f.idx[r * f.J + j]]
                       + f.mn[r * f.J + j] * av;
        }
        double v = rel_l2(yd, yo, f.M);
        snprintf(buf, sizeof buf, "rel L2 vs decoded oracle %.3e", v);
        verdict("learned rotation (tables against Rx)", v < 1e-4, buf);
        free(R); free(Cz); free(yo); free(yd);
        fx_free(&f);
    }
    /* 16. corrupted index: DC tracks the representation */
    {
        Fx f; fx_alloc(&f, 128, 4);
        fx_random(&f, 25, 0.1, 0.01, 0.03, 0.005);
        float *yc = (float *)malloc((size_t)f.M * 4);
        dc(&f, yc);
        f.idx[0] ^= 0xff;
        double v_track = dc_vs_oracle(&f);   /* oracle sees same corruption */
        float *yk = (float *)malloc((size_t)f.M * 4);
        dc(&f, yk);
        double *a = (double *)malloc((size_t)f.M * 8);
        double *b = (double *)malloc((size_t)f.M * 8);
        for (int r = 0; r < f.M; r++) { a[r] = yk[r]; b[r] = yc[r]; }
        double v_sens = rel_l2(a, b, f.M);
        snprintf(buf, sizeof buf, "vs corrupted oracle %.2e; vs clean "
                 "output %.3f", v_track, v_sens);
        verdict("corrupted index tracked, not hidden",
                v_track < 1e-5 && v_sens > 0.001, buf);
        free(yc); free(yk); free(a); free(b); fx_free(&f);
    }
    /* 17. stale table: T from x_a, A from x_b must diverge */
    {
        Fx f; fx_alloc(&f, 128, 4);
        fx_random(&f, 26, 0.1, 0.01, 0.03, 0.005);
        float *T = (float *)malloc((size_t)f.J * K * 4);
        float *A = (float *)malloc((size_t)f.J * 4);
        dc_build(&f, T, A);
        float *xsav = (float *)malloc((size_t)f.J * B * 4);
        memcpy(xsav, f.x, (size_t)f.J * B * 4);
        rng_state = 123;
        for (int i = 0; i < f.J * B; i++) f.x[i] = (float)randn();
        /* rebuild only A for the new x (the stale mixup) */
        float *A2 = (float *)malloc((size_t)f.J * 4);
        for (int j = 0; j < f.J; j++) {
            float av = 0;
            for (int i = 0; i < B; i++) av += f.x[j * B + i];
            A2[j] = av;
        }
        float *y = (float *)malloc((size_t)f.M * 4);
        dc_apply(&f, T, A2, y);
        double *yo = (double *)malloc((size_t)f.M * 8);
        double *dd = (double *)malloc((size_t)f.M * 8);
        oracle(&f, yo);
        for (int r = 0; r < f.M; r++) dd[r] = y[r];
        double v = rel_l2(dd, yo, f.M);
        snprintf(buf, sizeof buf, "stale-table output rel L2 %.3f "
                 "(must be large)", v);
        verdict("stale tables diverge detectably", v > 0.1, buf);
        memcpy(f.x, xsav, (size_t)f.J * B * 4);
        free(T); free(A); free(A2); free(xsav); free(y); free(yo); free(dd);
        fx_free(&f);
    }
    /* 18. determinism and hashes */
    {
        Fx f; fx_alloc(&f, 256, 12);
        fx_random(&f, 27, 0.1, 0.01, 0.03, 0.005);
        float *y = (float *)malloc((size_t)f.M * 4);
        uint64_t h0 = 0; int same = 1;
        for (int rep = 0; rep < 5; rep++) {
            dc(&f, y);
            uint64_t h = fnv1a(y, (size_t)f.M * 4);
            if (rep == 0) h0 = h; else if (h != h0) same = 0;
        }
        float *yr = (float *)malloc((size_t)f.M * 4);
        recon(&f, yr);
        uint64_t hr = fnv1a(yr, (size_t)f.M * 4);
        snprintf(buf, sizeof buf, "dc FNV %016llx stable x5; recon FNV "
                 "%016llx; equal: %s", (unsigned long long)h0,
                 (unsigned long long)hr, h0 == hr ? "yes" : "no "
                 "(expected: different summation order)");
        verdict("determinism: dc output hash stable", same, buf);
        printf("    book FNV %016llx, index-stream FNV %016llx\n",
               (unsigned long long)fnv1a(f.C, (size_t)K * B * 4),
               (unsigned long long)fnv1a(f.idx, (size_t)f.M * f.J));
        free(y); free(yr); fx_free(&f);
    }
    /* 19. long accumulation chain: J = 512 segments */
    {
        Fx f; fx_alloc(&f, 64, 512);
        fx_random(&f, 28, 0.1, 0.01, 0.03, 0.005);
        double v = dc_vs_oracle(&f);
        snprintf(buf, sizeof buf, "rel L2 vs oracle %.3e at J=512", v);
        verdict("long accumulation (512 segments)", v < 1e-4, buf);
        fx_free(&f);
    }
    /* 20. subnormal activations */
    {
        Fx f; fx_alloc(&f, 128, 4);
        fx_random(&f, 29, 0.1, 0.01, 0.03, 0.005);
        for (int i = 0; i < f.J * B; i++) f.x[i] *= 1.0e-40f;
        double v = dc_vs_oracle(&f);
        snprintf(buf, sizeof buf, "rel L2 vs oracle %.3e (x ~ 1e-40)", v);
        verdict("subnormal activations", v < 1e-2, buf);
        fx_free(&f);
    }
    printf("suite done.\n");
    return 0;
}
