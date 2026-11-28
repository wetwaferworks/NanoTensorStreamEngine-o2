/* ln_headbench.c - time an int8 per-row lm_head GEMV on the host, the
 * candidate for keeping the head off the GPU so an expanded pack can fit.
 * Uses the zero-mutex nemo_pool; each row's dot product is a fixed-order
 * AVX2 loop, so the logits are bit-identical across 1, 2, 4 and 8 lanes.
 * Usage: ln_headbench --ref CKPT_DIR [--threads 8] [--reps 8]
 */
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
static double now_s(void) { LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c); return (double)c.QuadPart / (double)f.QuadPart; }
#else
#include <time.h>
static double now_s(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
#endif
#include "loader.h"
#include "pool.h"

enum { H = 4480, VOCAB = 131072 };

static uint64_t fnv(const void *p, size_t n) {
    const uint8_t *b = p; uint64_t h = 14695981039346656037ull;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

typedef struct { const int8_t *q; const float *sc, *x; float *y; } job;

static float dot_row(const int8_t *q, const float *x) {
    __m256 acc = _mm256_setzero_ps();
    int c = 0;
    for (; c + 16 <= H; c += 16) {
        __m128i v = _mm_loadu_si128((const __m128i *)(q + c));
        __m256i i16 = _mm256_cvtepi8_epi16(v);
        __m256i lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(i16));
        __m256i hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(i16, 1));
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(lo), _mm256_loadu_ps(x + c), acc);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(hi), _mm256_loadu_ps(x + c + 8), acc);
    }
    float t[8]; _mm256_storeu_ps(t, acc);
    float s = t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
    for (; c < H; c++) s += (float)q[c] * x[c];
    return s;
}
static void rows(void *ctx, int begin, int end) {
    job *j = ctx;
    for (int r = begin; r < end; r++) j->y[r] = j->sc[r] * dot_row(j->q + (size_t)r * H, j->x);
}

int main(int argc, char **argv) {
    const char *ref = NULL; int threads = 8, reps = 8;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--ref") && i + 1 < argc) ref = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else { fprintf(stderr, "usage: ln_headbench --ref CKPT_DIR [--threads 8] [--reps 8]\n"); return 2; }
    }
    if (!ref || threads < 1 || threads > 8) { fprintf(stderr, "need --ref and --threads 1..8\n"); return 2; }
    nemo_checkpoint cp;
    if (nemo_checkpoint_open(&cp, ref)) { fprintf(stderr, "cannot open %s\n", ref); return 1; }
    const nemo_tensor_info *t = nemo_find(&cp, "lm_head.weight");
    if (!t || !t->is_bf16 || t->count != (uint64_t)VOCAB * H) { fprintf(stderr, "lm_head missing or not bf16\n"); return 1; }
    int8_t *q = malloc((size_t)VOCAB * H);
    float *sc = malloc(sizeof(float) * VOCAB), *x = malloc(sizeof(float) * H);
    float *y = malloc(sizeof(float) * VOCAB), *y1 = malloc(sizeof(float) * VOCAB);
    uint16_t *row = malloc((size_t)H * 2);
    if (!q || !sc || !x || !y || !y1 || !row) { fprintf(stderr, "out of memory\n"); return 1; }
    FILE *f = fopen(t->shard_path, "rb");
    if (!f || fseeko(f, (off_t)t->data_off, SEEK_SET)) { fprintf(stderr, "lm_head read failed\n"); return 1; }
    double t0 = now_s();
    for (int r = 0; r < VOCAB; r++) {
        if (fread(row, 2, H, f) != H) { fprintf(stderr, "lm_head read failed\n"); return 1; }
        float mx = 0.0f;
        float tmp[H];
        for (int c = 0; c < H; c++) { uint32_t u = (uint32_t)row[c] << 16; memcpy(&tmp[c], &u, 4); float v = tmp[c] < 0 ? -tmp[c] : tmp[c]; if (v > mx) mx = v; }
        float s = mx > 0.0f ? mx / 127.0f : 1.0f;
        sc[r] = s;
        float inv = 1.0f / s;
        for (int c = 0; c < H; c++) {
            int v = (int)lrintf(tmp[c] * inv);
            if (v > 127) v = 127; if (v < -127) v = -127;
            q[(size_t)r * H + c] = (int8_t)v;
        }
    }
    fclose(f);
    fprintf(stderr, "loaded and quantized lm_head in %.1f s (560 MiB int8 + scales)\n", now_s() - t0);
    /* a deterministic, non-zero hidden state */
    for (int c = 0; c < H; c++) x[c] = ((c * 17) % 200 - 100) * 0.01f;

    int lanes[] = {1, 2, 4, 8};
    uint64_t href = 0;
    for (int li = 0; li < 4; li++) {
        if (lanes[li] > threads) break;
        nemo_pool p; memset(&p, 0, sizeof p);
        if (nemo_pool_init(&p, lanes[li])) { fprintf(stderr, "pool init %d failed\n", lanes[li]); return 1; }
        job j = { q, sc, x, y };
        nemo_pool_for(&p, VOCAB, rows, &j); /* warmup */
        double a = now_s();
        for (int r = 0; r < reps; r++) nemo_pool_for(&p, VOCAB, rows, &j);
        double ms = (now_s() - a) * 1000.0 / reps;
        uint64_t h = fnv(y, sizeof(float) * VOCAB);
        if (li == 0) { href = h; memcpy(y1, y, sizeof(float) * VOCAB); }
        int same = (h == href);
        /* 13.1 tok/s GPU decode is 76.3 ms/token; report the combined rate if layers stay that fast */
        double combined = 1000.0 / (76.3 + ms);
        printf("lm_head int8 GEMV on %d lane%s: %.2f ms, hash %s %016llx, would be ~%.1f tok/s with a 13.1 tok/s GPU body (real-world)\n",
               lanes[li], lanes[li] > 1 ? "s" : "", ms, same ? "matches-1-lane" : "DIFFERS", (unsigned long long)h, combined);
        nemo_pool_shutdown(&p);
        if (!same) return 1;
    }
    return 0;
}
