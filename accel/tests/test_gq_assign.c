/* test_gq_assign.c - the GPU nearest-code search against the CPU
 * search of nq2-quantize, byte for byte.
 *
 * Co-implemented by Cortana at JD's direction.
 *
 * The quantizer source is compiled into this test unchanged (its
 * main renamed), so the reference is the quantizer's own
 * nearest_centroid() and learn_book(), not a copy.
 *
 *   A. assignment: many widths, code counts, row counts and value
 *      patterns chosen to break a careless GPU search: exact ties,
 *      duplicated codes, one-ulp near ties, grid values, subnormals,
 *      underflowing squares, fp32 overflow, NaN and infinity.
 *   B. k-means: learn_book() on the CPU (1 and 4 threads) against
 *      the GPU lane, flat books and 64-book PQ batches; the books
 *      and the random-number state afterwards must match exactly.
 *
 * Exit code 0 only if every comparison matches. Needs a CUDA
 * device; without one it reports SKIP and exits 2.
 */
#define main nq2_quantize_main
#include "../../tools/nq2-quantize.c"
#undef main

static uint64_t t_rng = 0x9E3779B97F4A7C15ull;
static uint64_t trand(void)
{
    uint64_t x = t_rng;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    t_rng = x;
    return x * 0x2545F4914F6CDD1Dull;
}
static double tuni(void) { return (double)(trand() >> 11) * (1.0 / 9007199254740992.0); }
static float tgauss(void)
{
    double s = 0.0;
    for (int i = 0; i < 12; i++) s += tuni();
    return (float)(s - 6.0);
}
static float next_up(float f)
{
    uint32_t u; memcpy(&u, &f, 4);
    if (f >= 0) u++; else u--;
    memcpy(&f, &u, 4);
    return f;
}

static int g_fail = 0, g_cases = 0;
static uint64_t g_cmp = 0;

#define NPAT 13
static const char *pat_name[NPAT] = {
    "gaussian", "duplicate codes", "rows equal codes", "one-ulp near ties",
    "grid values", "subnormal", "underflowing squares", "fp32 overflow",
    "NaN and inf", "all zero", "mixed magnitude", "rms-normalized", "all codes equal"
};

static void fill(int pat, float *x, uint64_t n, uint32_t stride,
                 float *books, uint32_t nsub, uint32_t entries, uint32_t d)
{
    size_t nb = (size_t)nsub * entries * d;
    size_t nx = (size_t)n * stride;
    switch (pat) {
    case 0:
        for (size_t i = 0; i < nb; i++) books[i] = tgauss();
        for (size_t i = 0; i < nx; i++) x[i] = tgauss();
        break;
    case 1:   /* every code duplicated: exact ties everywhere */
        for (uint32_t s = 0; s < nsub; s++)
            for (uint32_t k = 0; k < entries; k++)
                for (uint32_t i = 0; i < d; i++) {
                    size_t at = ((size_t)s * entries + k) * d + i;
                    books[at] = (k & 1u) ? books[at - d] : tgauss();
                }
        for (size_t i = 0; i < nx; i++) x[i] = tgauss();
        break;
    case 2:   /* rows are codes: zero distance, plus duplicates */
        for (size_t i = 0; i < nb; i++) books[i] = (float)((int)(trand() % 7) - 3) * 0.5f;
        for (uint64_t b = 0; b < n; b++)
            for (uint32_t s = 0; s < nsub; s++) {
                uint32_t k = (uint32_t)(trand() % entries);
                memcpy(x + b * stride + (size_t)s * d,
                       books + ((size_t)s * entries + k) * d, d * sizeof(float));
            }
        break;
    case 3:   /* codes one ulp apart around a shared base */
        for (uint32_t s = 0; s < nsub; s++) {
            for (uint32_t i = 0; i < d; i++) books[(size_t)s * entries * d + i] = tgauss();
            for (uint32_t k = 1; k < entries; k++)
                for (uint32_t i = 0; i < d; i++) {
                    size_t at = ((size_t)s * entries + k) * d + i;
                    float base = books[(size_t)s * entries * d + i];
                    uint32_t steps = (uint32_t)(trand() % 3);
                    for (uint32_t q = 0; q < steps; q++) base = next_up(base);
                    books[at] = base;
                }
        }
        for (uint64_t b = 0; b < n; b++)
            for (uint32_t s = 0; s < nsub; s++)
                for (uint32_t i = 0; i < d; i++) {
                    float base = books[(size_t)s * entries * d + i];
                    uint32_t steps = (uint32_t)(trand() % 4);
                    for (uint32_t q = 0; q < steps; q++) base = next_up(base);
                    x[b * stride + (size_t)s * d + i] = base;
                }
        break;
    case 4:   /* small grid: many equal distances */
        for (size_t i = 0; i < nb; i++) books[i] = (float)((int)(trand() % 5) - 2) * 0.25f;
        for (size_t i = 0; i < nx; i++) x[i] = (float)((int)(trand() % 9) - 4) * 0.125f;
        break;
    case 5:   /* subnormal values */
        for (size_t i = 0; i < nb; i++) books[i] = tgauss() * 1e-40f;
        for (size_t i = 0; i < nx; i++) x[i] = tgauss() * 1e-40f;
        break;
    case 6:   /* squares underflow fp32 but not double */
        for (size_t i = 0; i < nb; i++) books[i] = tgauss() * 1e-22f;
        for (size_t i = 0; i < nx; i++) x[i] = tgauss() * 1e-22f;
        break;
    case 7:   /* differences square past fp32 range */
        for (size_t i = 0; i < nb; i++) books[i] = tgauss() * 1e20f;
        for (size_t i = 0; i < nx; i++) x[i] = tgauss() * (trand() & 1 ? 1e20f : 3e38f);
        break;
    case 8: { /* NaN and infinity in rows and codes */
        float inf = 1e30f; inf *= 1e30f;
        float nan = inf - inf;
        for (size_t i = 0; i < nb; i++) {
            uint64_t r = trand() % 64;
            books[i] = r == 0 ? nan : r == 1 ? inf : r == 2 ? -inf : tgauss();
        }
        for (size_t i = 0; i < nx; i++) {
            uint64_t r = trand() % 128;
            x[i] = r == 0 ? nan : r == 1 ? inf : tgauss();
        }
        break; }
    case 9:
        memset(books, 0, nb * sizeof(float));
        memset(x, 0, nx * sizeof(float));
        break;
    case 10:
        for (size_t i = 0; i < nb; i++) books[i] = tgauss();
        for (uint64_t b = 0; b < n; b++) {
            float sc = (b % 3 == 0) ? 1e-30f : (b % 3 == 1) ? 1.0f : 1e15f;
            for (uint32_t i = 0; i < stride; i++) x[b * stride + i] = tgauss() * sc;
        }
        break;
    case 11: { /* like the quantizer's own RMS-normalized blocks */
        for (size_t i = 0; i < nb; i++) books[i] = tgauss();
        for (uint64_t b = 0; b < n; b++) {
            double ss = 0.0;
            for (uint32_t i = 0; i < stride; i++) {
                float v = tgauss() * 0.02f;
                x[b * stride + i] = v;
                ss += (double)v * v;
            }
            float sc = (float)dsqrt(ss / stride);
            for (uint32_t i = 0; i < stride; i++)
                x[b * stride + i] = sc > 0 ? x[b * stride + i] / sc : 0.0f;
        }
        break; }
    default: /* every code identical: the answer must be 0 */
        for (uint32_t s = 0; s < nsub; s++)
            for (uint32_t i = 0; i < d; i++) {
                float v = tgauss();
                for (uint32_t k = 0; k < entries; k++)
                    books[((size_t)s * entries + k) * d + i] = v;
            }
        for (size_t i = 0; i < nx; i++) x[i] = tgauss();
        break;
    }
}

static void case_assign(int pat, uint32_t d, uint32_t nsub, uint32_t entries,
                        uint64_t n, uint32_t extra)
{
    uint32_t stride = nsub * d + extra;
    float *x = (float *)malloc((size_t)n * stride * sizeof(float) + 4);
    float *books = (float *)malloc((size_t)nsub * entries * d * sizeof(float));
    uint8_t *a = (uint8_t *)malloc((size_t)n * nsub + 1);
    uint8_t *g = (uint8_t *)malloc((size_t)n * nsub + 1);
    if (!x || !books || !a || !g) { fprintf(stderr, "out of memory\n"); exit(1); }
    for (size_t i = 0; i < (size_t)n * stride; i++) x[i] = 0.0f;
    fill(pat, x, n, stride, books, nsub, entries, d);
    cpu_assign(x, n, stride, nsub, d, books, entries, a);
    memset(g, 0xEE, (size_t)n * nsub);
    int rc = gq_set_rows(x, n, stride);
    if (rc == 0) rc = gq_assign_rows(nsub, d, books, entries, g);
    g_cases++;
    size_t bad = 0, first = 0;
    for (size_t i = 0; i < (size_t)n * nsub; i++)
        if (a[i] != g[i]) { if (!bad) first = i; bad++; }
    g_cmp += (uint64_t)n * nsub;
    if (rc != 0 || bad) {
        g_fail++;
        printf("FAIL assign %-22s d %4u nsub %3u entries %3u n %6llu: rc %d, %zu differ",
               pat_name[pat], d, nsub, entries, (unsigned long long)n, rc, bad);
        if (bad) printf(" (first at %zu: gpu %u cpu %u)", first, g[first], a[first]);
        printf("\n");
    }
    free(x); free(books); free(a); free(g);
}

static int same_bytes(const void *p, const void *q, size_t n) { return memcmp(p, q, n) == 0; }

static void case_learn_flat(int pat, uint32_t d, uint32_t entries, uint64_t n, int iters, int cpu_threads)
{
    float *x = (float *)malloc((size_t)n * d * sizeof(float) + 4);
    float *tmpb = (float *)malloc((size_t)entries * d * sizeof(float));
    float *c1 = (float *)malloc((size_t)entries * d * sizeof(float));
    float *c2 = (float *)malloc((size_t)entries * d * sizeof(float));
    if (!x || !tmpb || !c1 || !c2) { fprintf(stderr, "out of memory\n"); exit(1); }
    fill(pat, x, n, d, tmpb, 1, entries, d);
    uint64_t seed = trand() | 1u;
    g_kmeans_verbose = 0;
    g_gpu = 0; g_threads = cpu_threads; g_learn_threads = 0;
    rng_state = seed;
    learn_book(x, n, d, entries, iters, c1);
    uint64_t r1 = rng_state;
    g_gpu = 1;
    rng_state = seed;
    learn_book(x, n, d, entries, iters, c2);
    uint64_t r2 = rng_state;
    g_gpu = 0;
    g_cases++;
    g_cmp += (uint64_t)n * (uint64_t)(iters > 0 ? iters : 0);
    if (!same_bytes(c1, c2, (size_t)entries * d * sizeof(float)) || r1 != r2) {
        g_fail++;
        printf("FAIL learn flat %-22s d %4u entries %3u n %6llu iters %d threads %d%s\n",
               pat_name[pat], d, entries, (unsigned long long)n, iters, cpu_threads,
               r1 != r2 ? " (rng state differs)" : "");
    }
    free(x); free(tmpb); free(c1); free(c2);
}

/* PQ: pack_one_tensor's CPU loop (one learn_book per subvector on a
 * packed copy) against one batched gpu_learn_books call. */
static void case_learn_pq(int pat, uint32_t d, uint32_t entries, uint64_t n, int iters)
{
    const uint32_t block = 256, nsub = block / d;
    float *x = (float *)malloc((size_t)n * block * sizeof(float) + 4);
    float *tmpb = (float *)malloc((size_t)nsub * entries * d * sizeof(float));
    float *b1 = (float *)calloc((size_t)nsub * entries * d, sizeof(float));
    float *b2 = (float *)calloc((size_t)nsub * entries * d, sizeof(float));
    float *sub = (float *)malloc((size_t)n * d * sizeof(float) + 4);
    if (!x || !tmpb || !b1 || !b2 || !sub) { fprintf(stderr, "out of memory\n"); exit(1); }
    fill(pat, x, n, block, tmpb, nsub, entries, d);
    uint64_t seed = trand() | 1u;
    g_kmeans_verbose = 0;
    g_gpu = 0; g_threads = 1; g_learn_threads = 1;
    rng_state = seed;
    for (uint32_t s = 0; s < nsub; s++) {
        for (uint64_t b = 0; b < n; b++)
            memcpy(sub + b * d, x + b * block + s * d, d * sizeof(float));
        learn_book(sub, n, d, entries, iters, b1 + (size_t)s * entries * d);
    }
    uint64_t r1 = rng_state;
    rng_state = seed;
    gpu_learn_books(x, n, block, nsub, d, entries, iters, b2);
    uint64_t r2 = rng_state;
    g_learn_threads = 0;
    g_cases++;
    g_cmp += (uint64_t)n * nsub * (uint64_t)(iters > 0 ? iters : 0);
    if (!same_bytes(b1, b2, (size_t)nsub * entries * d * sizeof(float)) || r1 != r2) {
        g_fail++;
        printf("FAIL learn pq %-22s d %u entries %3u n %6llu iters %d%s\n",
               pat_name[pat], d, entries, (unsigned long long)n, iters,
               r1 != r2 ? " (rng state differs)" : "");
    }
    free(x); free(tmpb); free(b1); free(b2); free(sub);
}

int main(int argc, char **argv)
{
    int heavy = argc > 1 && strcmp(argv[1], "--heavy") == 0;
    char why[512] = "";
    if (gq_init(0, why, sizeof why) != 0) {
        printf("SKIP: no CUDA lane (%s)\n", why);
        return 2;
    }
    printf("device: %s\n", gq_device_name());
    time_t t0 = time(NULL);

    /* A. assignment */
    static const uint32_t ds[] = { 1, 2, 3, 4, 5, 8, 16, 17, 64, 256, 1024, 4096 };
    static const uint32_t es[] = { 1, 2, 3, 16, 255, 256 };
    static const uint64_t ns[] = { 1, 5, 127, 128, 129, 3000 };
    int na = 0;
    for (int pat = 0; pat < NPAT; pat++)
        for (size_t di = 0; di < sizeof ds / sizeof ds[0]; di++)
            for (size_t ei = 0; ei < sizeof es / sizeof es[0]; ei++) {
                uint32_t d = ds[di], e = es[ei];
                uint32_t nsub = d <= 8 ? 256 / (d > 0 ? d : 1) : (d <= 64 ? 4 : 1);
                if (nsub > 64) nsub = 64;
                uint64_t n = ns[(pat + di + ei) % (sizeof ns / sizeof ns[0])];
                if ((uint64_t)d * e > 65536 && n > 129) n = 129;
                if (heavy && d <= 16) n *= 4;
                case_assign(pat, d, nsub, e, n, (uint32_t)((pat + ei) % 3));
                na++;
            }
    /* the SuperVL shape: block 256, PQ d 4, 256 codes, a full chunk */
    for (int pat = 0; pat < NPAT; pat++) { case_assign(pat, 4, 64, 256, heavy ? 8192 : 2048, 0); na++; }
    for (int pat = 0; pat < NPAT; pat++) { case_assign(pat, 8, 32, 256, heavy ? 8192 : 2048, 0); na++; }
    for (int pat = 0; pat < NPAT; pat++) { case_assign(pat, 256, 1, 256, heavy ? 8192 : 1024, 0); na++; }
    printf("A. assignment: %d cases\n", na);

    /* B. k-means */
    int nl = 0;
    static const int pats_l[] = { 0, 1, 2, 4, 5, 6, 9, 10, 11, 12 };
    for (size_t pi = 0; pi < sizeof pats_l / sizeof pats_l[0]; pi++) {
        int pat = pats_l[pi];
        case_learn_flat(pat, 256, 256, heavy ? 8192 : 1500, 8, 1); nl++;
        case_learn_flat(pat, 256, 256, heavy ? 8192 : 1500, 8, 4); nl++;
        case_learn_flat(pat, 64, 16, 700, 3, 4); nl++;
        case_learn_flat(pat, 16, 256, 7, 8, 1); nl++;
        case_learn_flat(pat, 4, 256, 1, 2, 1); nl++;
        case_learn_flat(pat, 256, 256, 600, 0, 1); nl++;
        case_learn_pq(pat, 4, 256, heavy ? 8192 : 1200, 8); nl++;
        case_learn_pq(pat, 8, 256, heavy ? 8192 : 900, 8); nl++;
        case_learn_pq(pat, 4, 16, 300, 5); nl++;
        case_learn_pq(pat, 4, 256, 3, 8); nl++;
    }
    printf("B. k-means: %d cases\n", nl);

    uint64_t rows = 0, multi = 0, full = 0;
    gq_thread_stats(&rows, &multi, &full);
    printf("gpu searches %llu: %llu rescored with near ties, %llu full double searches\n",
           (unsigned long long)rows, (unsigned long long)multi, (unsigned long long)full);
    printf("%d cases, %llu row comparisons, %lld s\n", g_cases,
           (unsigned long long)g_cmp, (long long)(time(NULL) - t0));
    gq_shutdown();
    if (g_fail) { printf("RESULT: FAIL (%d of %d cases)\n", g_fail, g_cases); return 1; }
    printf("RESULT: ALL PASS (%d cases, byte-identical)\n", g_cases);
    return 0;
}
