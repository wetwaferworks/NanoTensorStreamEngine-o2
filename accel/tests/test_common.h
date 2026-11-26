/* test_common.h - fixtures and metrics for the accel suites.
 * Co-implemented by Cortana. Deterministic: every fixture comes
 * from a fixed splitmix64 seed, so every run sees the same bits. */
#ifndef NTSE_TEST_COMMON_H
#define NTSE_TEST_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../include/ntse_accel.h"

static __attribute__((unused)) int g_checks = 0, g_fail = 0;
static __attribute__((unused)) void expect(const char *name, int ok)
{
    g_checks++;
    if (!ok) g_fail++;
    printf("%s %s\n", ok ? "ok  " : "FAIL", name);
    fflush(stdout);
}

static __attribute__((unused)) uint64_t g_rng = 42;
static __attribute__((unused)) uint64_t rnd(void)
{
    uint64_t z = (g_rng += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
static __attribute__((unused)) double urand(void) { return (rnd() >> 11) / 9007199254740992.0; }
static __attribute__((unused)) double nrand(void)
{
    double u1 = ((rnd() >> 11) + 1) / 9007199254740992.0, u2 = urand();
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

/* A synthetic tensor that looks like a real pack: a book of small
 * prototypes, a skewed index (only `active` of the entries used per
 * segment, so the active-index op has something to skip), positive
 * fp16 scales and signed fp16 minima drawn as direct bit patterns. */
typedef struct {
    ntse_dtc_tensor t;
    float *book;
    uint8_t *idx;
    uint16_t *mn, *sc;
} fixture;

static __attribute__((unused)) void fixture_free(fixture *f)
{
    free(f->book); free(f->idx); free(f->mn); free(f->sc);
    memset(f, 0, sizeof *f);
}

static __attribute__((unused)) int fixture_make(fixture *f, uint64_t rows, uint64_t cols, uint32_t block,
                        uint32_t entries, uint32_t active, uint64_t seed)
{
    memset(f, 0, sizeof *f);
    g_rng = seed;
    uint64_t nseg = cols / block, nblk = rows * nseg;
    f->book = (float *)malloc((size_t)entries * block * 4);
    f->idx = (uint8_t *)malloc(nblk);
    f->mn = (uint16_t *)malloc(nblk * 2);
    f->sc = (uint16_t *)malloc(nblk * 2);
    if (!f->book || !f->idx || !f->mn || !f->sc) { fixture_free(f); return -1; }
    for (uint64_t i = 0; i < (uint64_t)entries * block; i++)
        f->book[i] = (float)(nrand() * 0.5);
    if (active == 0 || active > entries) active = entries;
    for (uint64_t j = 0; j < nseg; j++) {
        uint32_t base = (uint32_t)(rnd() % entries);
        for (uint64_t r = 0; r < rows; r++) {
            uint64_t b = r * nseg + j;
            f->idx[b] = (uint8_t)((base + (rnd() % active) * 7u) % entries);
        }
    }
    for (uint64_t b = 0; b < nblk; b++) {
        /* scale: positive, exponent 2^-9..2^-5; min: signed, 2^-12..2^-7 */
        f->sc[b] = (uint16_t)(((6 + rnd() % 5) << 10) | (rnd() & 0x3ff));
        f->mn[b] = (uint16_t)(((rnd() & 1) << 15) | ((3 + rnd() % 6) << 10) | (rnd() & 0x3ff));
    }
    f->t.rows = rows; f->t.cols = cols; f->t.block = block; f->t.entries = entries;
    f->t.book = f->book; f->t.idx = f->idx; f->t.mn_h = f->mn; f->t.sc_h = f->sc;
    return 0;
}

static __attribute__((unused)) float *activations(uint64_t n, uint64_t seed)
{
    float *x = (float *)malloc(n * 4);
    if (!x) return NULL;
    g_rng = seed;
    for (uint64_t i = 0; i < n; i++) x[i] = (float)nrand();
    return x;
}

typedef struct {
    double max_abs, max_rel, rel_l2, cosine;
    uint64_t differ, nonfinite;
} metrics;

static __attribute__((unused)) metrics compare_f(const float *a, const float *ref, uint64_t n)
{
    metrics m = { 0, 0, 0, 0, 0, 0 };
    double d2 = 0, r2 = 0, ab = 0, a2 = 0;
    for (uint64_t i = 0; i < n; i++) {
        if (!isfinite(a[i]) || !isfinite(ref[i])) { m.nonfinite++; continue; }
        if (memcmp(&a[i], &ref[i], 4) != 0) m.differ++;
        double d = (double)a[i] - ref[i];
        double ad = fabs(d);
        if (ad > m.max_abs) m.max_abs = ad;
        double den = fabs((double)ref[i]);
        if (den > 1e-6 && ad / den > m.max_rel) m.max_rel = ad / den;
        d2 += d * d; r2 += (double)ref[i] * ref[i];
        ab += (double)a[i] * ref[i]; a2 += (double)a[i] * a[i];
    }
    m.rel_l2 = r2 > 0 ? sqrt(d2 / r2) : sqrt(d2);
    m.cosine = (a2 > 0 && r2 > 0) ? ab / sqrt(a2 * r2) : (d2 == 0 ? 1.0 : 0.0);
    return m;
}

static __attribute__((unused)) double rel_l2_vs_oracle(const float *a, const double *o, uint64_t n)
{
    double d2 = 0, r2 = 0;
    for (uint64_t i = 0; i < n; i++) {
        double d = (double)a[i] - o[i];
        d2 += d * d; r2 += o[i] * o[i];
    }
    return r2 > 0 ? sqrt(d2 / r2) : sqrt(d2);
}

static __attribute__((unused)) void print_metrics(const char *tag, const metrics *m)
{
    printf("     %-44s max_abs %.3e max_rel %.3e relL2 %.3e cos %.9f "
           "differ %llu nonfinite %llu%s\n", tag, m->max_abs, m->max_rel,
           m->rel_l2, m->cosine, (unsigned long long)m->differ,
           (unsigned long long)m->nonfinite, m->differ == 0 ? "  BITEXACT" : "");
}

/* Tolerances (documented in accel/README.md): the DTC ops are
 * designed bit-exact; the gate is still a tolerance so a device
 * that flushes denormals reports, rather than hides, its drift. */
#define TOL_DTC_RELL2  1e-6
#define TOL_EXP_RELL2  1e-5
#define TOL_COS        0.999999

#endif
