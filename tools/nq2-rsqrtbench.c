/*
 * nq2-rsqrtbench.c - NanoQuatTwo normalization research, Tests A+B.
 *
 * Reciprocal square root, measured the way Gizmo's directive
 * demands: a preserved scalar reference, error recorded against
 * it, cycles per block on a real normalization-shaped workload,
 * and end-to-end throughput... not isolated instruction numbers
 * alone. Methods: exact scalar, AVX2 rsqrt (raw, +1 Newton, +2),
 * AVX-512 rsqrt14 (raw, +1, +2) where compiled in, and lookup
 * tables of 256 and 1024 entries (raw, +1, +2) built on
 * exponent/mantissa decomposition.
 *
 * The acceptance rule from the directive stands: a lookup table
 * is promoted only if it beats hardware rsqrt in matched
 * end-to-end measurements. This bench is the evidence, either way.
 *
 * Build: gcc -O3 -march=native. Architecture: JD. Outside review:
 * Gizmo. Implementation: Elfie.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}
static uint64_t rdtsc(void)
{
#if defined(__x86_64__) || defined(__i386__)
    return __rdtsc();
#else
    return 0;
#endif
}
static float ref_rsqrt(float x) { return (float)(1.0 / sqrt((double)x)); }

static uint64_t lcg_state = 0x9E3779B97F4A7C15ull;
static float frand_range(void)
{
    uint64_t x = lcg_state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    lcg_state = x;
    uint32_t u = (uint32_t)((x * 0x2545F4914F6CDD1Dull) >> 40);
    /* log-spread over 1e-6 .. 1e2: our mean-square territory */
    double t = (double)(u & 0xffffff) / 16777216.0;
    double v = pow(10.0, -6.0 + 8.0 * t);
    return (float)v;
}

/* ---------------- lookup tables ----------------------------------- */

static float g_tab256[256], g_tab1024[1024];

static void build_tab(float *tab, int bits)
{
    int n = 1 << bits;
    for (int i = 0; i < n; i++) {
        double mc = 1.0 + ((double)i + 0.5) / (double)n;
        tab[i] = (float)(1.0 / sqrt(mc));
    }
}

static float lut_rsqrt(float x, const float *tab, int bits)
{
    uint32_t u;
    memcpy(&u, &x, 4);
    int e = (int)((u >> 23) & 0xff) - 127;
    uint32_t idx = (u >> (23 - bits)) & ((1u << bits) - 1u);
    float y = tab[idx];
    int he;
    float adj;
    if (e & 1) { he = -((e - 1) / 2); adj = 0.70710678118654752440f; }
    else       { he = -(e / 2);       adj = 1.0f; }
    /* exact power of two from bits (ldexpf here would be a libm
     * call per element and would swamp the measurement) */
    uint32_t sb = (uint32_t)(127 + he) << 23;
    float scale;
    memcpy(&scale, &sb, 4);
    return y * adj * scale;
}

static float newton1(float y, float x)
{
    return y * (1.5f - 0.5f * x * y * y);
}
static float newton2(float y, float x)
{
    return newton1(newton1(y, x), x);
}

/* ---------------- accuracy ---------------------------------------- */

typedef float (*RsqrtFn)(float);
static float f_exact(float x) { return ref_rsqrt(x); }
static float f_lut256_0(float x) { return lut_rsqrt(x, g_tab256, 8); }
static float f_lut256_1(float x) { return newton1(f_lut256_0(x), x); }
static float f_lut256_2(float x) { return newton2(f_lut256_0(x), x); }
static float f_lut1024_0(float x) { return lut_rsqrt(x, g_tab1024, 10); }
static float f_lut1024_1(float x) { return newton1(f_lut1024_0(x), x); }
static float f_lut1024_2(float x) { return newton2(f_lut1024_0(x), x); }

static void accuracy(const char *name, RsqrtFn fn, const float *xs, int n)
{
    double maxrel = 0.0, sumrel2 = 0.0;
    for (int i = 0; i < n; i++) {
        double r = ref_rsqrt(xs[i]);
        double v = fn(xs[i]);
        double rel = fabs(v - r) / r;
        if (rel > maxrel) maxrel = rel;
        sumrel2 += rel * rel;
    }
    printf("  %-22s max rel err %.3e   rms rel err %.3e\n",
           name, maxrel, sqrt(sumrel2 / n));
}

#if defined(__AVX2__)
static void acc_avx2(const float *xs, int n)
{
    double mx[3] = {0, 0, 0}, sm[3] = {0, 0, 0};
    for (int i = 0; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(xs + i);
        __m256 y0 = _mm256_rsqrt_ps(x);
        __m256 half = _mm256_set1_ps(0.5f), three = _mm256_set1_ps(3.0f);
        __m256 y1 = _mm256_mul_ps(y0, _mm256_sub_ps(three,
            _mm256_mul_ps(x, _mm256_mul_ps(y0, y0))));
        y1 = _mm256_mul_ps(y1, half);
        __m256 y2 = _mm256_mul_ps(y1, _mm256_sub_ps(three,
            _mm256_mul_ps(x, _mm256_mul_ps(y1, y1))));
        y2 = _mm256_mul_ps(y2, half);
        float out[3][8];
        _mm256_storeu_ps(out[0], y0);
        _mm256_storeu_ps(out[1], y1);
        _mm256_storeu_ps(out[2], y2);
        for (int k = 0; k < 8; k++) {
            double r = ref_rsqrt(xs[i + k]);
            for (int m = 0; m < 3; m++) {
                double rel = fabs((double)out[m][k] - r) / r;
                if (rel > mx[m]) mx[m] = rel;
                sm[m] += rel * rel;
            }
        }
    }
    const char *names[3] = {"avx2 rsqrt raw", "avx2 rsqrt +1N", "avx2 rsqrt +2N"};
    for (int m = 0; m < 3; m++)
        printf("  %-22s max rel err %.3e   rms rel err %.3e\n",
               names[m], mx[m], sqrt(sm[m] / n));
}
#endif

#if defined(__AVX512F__)
static void acc_avx512(const float *xs, int n)
{
    double mx[3] = {0, 0, 0}, sm[3] = {0, 0, 0};
    for (int i = 0; i + 16 <= n; i += 16) {
        __m512 x = _mm512_loadu_ps(xs + i);
        __m512 y0 = _mm512_rsqrt14_ps(x);
        __m512 half = _mm512_set1_ps(0.5f), three = _mm512_set1_ps(3.0f);
        __m512 y1 = _mm512_mul_ps(y0, _mm512_sub_ps(three,
            _mm512_mul_ps(x, _mm512_mul_ps(y0, y0))));
        y1 = _mm512_mul_ps(y1, half);
        __m512 y2 = _mm512_mul_ps(y1, _mm512_sub_ps(three,
            _mm512_mul_ps(x, _mm512_mul_ps(y1, y1))));
        y2 = _mm512_mul_ps(y2, half);
        float out[3][16];
        _mm512_storeu_ps(out[0], y0);
        _mm512_storeu_ps(out[1], y1);
        _mm512_storeu_ps(out[2], y2);
        for (int k = 0; k < 16; k++) {
            double r = ref_rsqrt(xs[i + k]);
            for (int m = 0; m < 3; m++) {
                double rel = fabs((double)out[m][k] - r) / r;
                if (rel > mx[m]) mx[m] = rel;
                sm[m] += rel * rel;
            }
        }
    }
    const char *names[3] = {"avx512 rsqrt14 raw", "avx512 rsqrt14 +1N",
                            "avx512 rsqrt14 +2N"};
    for (int m = 0; m < 3; m++)
        printf("  %-22s max rel err %.3e   rms rel err %.3e\n",
               names[m], mx[m], sqrt(sm[m] / n));
}
#endif

/* ---------------- block normalize (cycles per block) --------------- */
/* The real workload shape: a 256-wide row, mean of squares, one
 * rsqrt, then scale every element. Only the rsqrt method varies. */

#define BLK 256
static float g_blk[BLK], g_out[BLK];

static double bench_block(const char *name, int method, int reps)
{
    uint64_t best = UINT64_MAX;
    volatile float sink = 0.0f;
    for (int r = 0; r < reps; r++) {
        uint64_t t0 = rdtsc();
        for (int it = 0; it < 2000; it++) {
            float ss = 0.0f;
            for (int i = 0; i < BLK; i++) ss += g_blk[i] * g_blk[i];
            float ms = ss / (float)BLK + 1e-5f;
            float y;
            switch (method) {
                case 0: y = ref_rsqrt(ms); break;
                case 1: y = lut_rsqrt(ms, g_tab256, 8); break;
                case 2: y = newton1(lut_rsqrt(ms, g_tab256, 8), ms); break;
                case 3: y = lut_rsqrt(ms, g_tab1024, 10); break;
                case 4: y = newton1(lut_rsqrt(ms, g_tab1024, 10), ms); break;
                default: y = ref_rsqrt(ms); break;
            }
#if defined(__AVX2__)
            if (method == 5 || method == 6) {
                __m256 xv = _mm256_set1_ps(ms);
                __m256 yv = _mm256_rsqrt_ps(xv);
                if (method == 6) {
                    __m256 half = _mm256_set1_ps(0.5f),
                           three = _mm256_set1_ps(3.0f);
                    yv = _mm256_mul_ps(yv, _mm256_sub_ps(three,
                        _mm256_mul_ps(xv, _mm256_mul_ps(yv, yv))));
                    yv = _mm256_mul_ps(yv, half);
                }
                float tmp[8];
                _mm256_storeu_ps(tmp, yv);
                y = tmp[0];
            }
#endif
            for (int i = 0; i < BLK; i++) g_out[i] = g_blk[i] * y;
            sink += g_out[BLK - 1];
        }
        uint64_t dt = rdtsc() - t0;
        if (dt < best) best = dt;
    }
    double cyc = (double)best / 2000.0;
    printf("  %-22s %8.1f cycles/block (%6.2f cycles/element)\n",
           name, cyc, cyc / BLK);
    (void)sink;
    return cyc;
}

/* ---------------- isolated throughput ------------------------------ */

static void bench_throughput(const float *xs, int n)
{
    printf("isolated rsqrt throughput (values recycled in L1):\n");
    const int reps = 200;
    struct { const char *name; RsqrtFn fn; } sc[] = {
        {"exact scalar", f_exact},
        {"lut256 raw", f_lut256_0}, {"lut256 +1N", f_lut256_1},
        {"lut1024 raw", f_lut1024_0}, {"lut1024 +1N", f_lut1024_1},
    };
    for (size_t m = 0; m < sizeof(sc) / sizeof(sc[0]); m++) {
        double t0 = now_s();
        volatile double sink = 0.0;
        for (int r = 0; r < reps; r++)
            for (int i = 0; i < n; i++) sink += sc[m].fn(xs[i]);
        double dt = now_s() - t0;
        printf("  %-22s %7.3f ns/value\n", sc[m].name,
               dt * 1e9 / ((double)reps * n));
        (void)sink;
    }
#if defined(__AVX2__)
    {
        double t0 = now_s();
        /* four independent accumulators, summed and stored every
         * rep: the loop cannot be eliminated and is not a pure
         * latency chain */
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(),
               a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        float out[8];
        for (int r = 0; r < reps; r++) {
            for (int i = 0; i + 32 <= n; i += 32) {
                a0 = _mm256_add_ps(a0, _mm256_rsqrt_ps(_mm256_loadu_ps(xs + i)));
                a1 = _mm256_add_ps(a1, _mm256_rsqrt_ps(_mm256_loadu_ps(xs + i + 8)));
                a2 = _mm256_add_ps(a2, _mm256_rsqrt_ps(_mm256_loadu_ps(xs + i + 16)));
                a3 = _mm256_add_ps(a3, _mm256_rsqrt_ps(_mm256_loadu_ps(xs + i + 24)));
            }
            _mm256_storeu_ps(out, _mm256_add_ps(_mm256_add_ps(a0, a1),
                                                _mm256_add_ps(a2, a3)));
        }
        double dt = now_s() - t0;
        printf("  %-22s %7.3f ns/value (sink %.1f)\n", "avx2 rsqrt raw",
               dt * 1e9 / ((double)reps * n), (double)out[0]);
    }
#endif
}

int main(void)
{
    build_tab(g_tab256, 8);
    build_tab(g_tab1024, 10);
    const int n = 1 << 16;
    float *xs = (float *)malloc((size_t)n * sizeof(float));
    if (!xs) return 1;
    for (int i = 0; i < n; i++) xs[i] = frand_range();
    for (int i = 0; i < BLK; i++) g_blk[i] = frand_range() - 0.5f;

    printf("rsqrt accuracy over 1e-6..1e2 (log-spread, %d samples):\n", n);
    accuracy("exact scalar", f_exact, xs, n);
#if defined(__AVX2__)
    acc_avx2(xs, n);
#endif
#if defined(__AVX512F__)
    acc_avx512(xs, n);
#endif
    accuracy("lut256 raw", f_lut256_0, xs, n);
    accuracy("lut256 +1N", f_lut256_1, xs, n);
    accuracy("lut256 +2N", f_lut256_2, xs, n);
    accuracy("lut1024 raw", f_lut1024_0, xs, n);
    accuracy("lut1024 +1N", f_lut1024_1, xs, n);
    accuracy("lut1024 +2N", f_lut1024_2, xs, n);

    printf("\nblock normalize, 256-wide rows (best of 5):\n");
    bench_block("exact scalar", 0, 5);
#if defined(__AVX2__)
    bench_block("avx2 rsqrt raw", 5, 5);
    bench_block("avx2 rsqrt +1N", 6, 5);
#endif
    bench_block("lut256 raw", 1, 5);
    bench_block("lut256 +1N", 2, 5);
    bench_block("lut1024 raw", 3, 5);
    bench_block("lut1024 +1N", 4, 5);

    printf("\n");
    bench_throughput(xs, n);
    free(xs);
    return 0;
}
