/*
 * nq2-streamtest.c - NanoQuatTwo four-loop streaming test harness.
 *
 * JD's architecture (2026-11-24, designed at the podium before bed):
 *   - the MAIN loop does all computation on the CPU (tensor math);
 *   - a DISC loop does nothing but lookups, grabbing the prefetch
 *     and predictive tensors needed next;
 *   - TWO loops (configurable) do the memory unpacks and hand the
 *     finished tensors forward to the main loop.
 * Consumption is in one fixed serial order no matter which unpack
 * loop finishes first: concurrent stages, serialized truth, the
 * Timing Handshake applied to streaming. The question the harness
 * answers with numbers: does the main loop ever wait, and what
 * end-to-end rate does the pipeline sustain from disc to CPU?
 *
 * Modes: --unpackers 0 runs the sequential reference (main loop
 * does read + unpack + compute itself) and produces the reference
 * accumulator every pipelined configuration must reproduce bit
 * for bit. Demand order is the engine's forward order over the
 * layer tensors (q, k, v, o, gate, up, down per layer); non-layer
 * tensors are resident in the real engine and excluded here.
 *
 * Usage:
 *   nq2-streamtest --pack S.nq2 --book S.nq2book --shards d [...]
 *                  [--unpackers N] [--lookahead N] [--layers N]
 *
 * Pure C99 + pthreads (bench tool; the engine carries the Windows
 * port of the same discipline). Architecture: JD. Implementation:
 * Elfie.
 */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <pthread.h>
#include <time.h>

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static uint32_t rd32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
/* Canonical shard order: by filename (the part after the last
 * '/' or '\\'), ties broken by the full path. The quantizer
 * writes packs in this order and the engines bind packs to
 * directory tensors in it; streamtest assigns each tensor's
 * block0 cumulatively in scan order, so its shard list is
 * sorted into the same order before scanning (Archie Round 3
 * P2-1 parenthetical, sustained in triage, 2026-11-25). */
static const char *base_name(const char *p)
{
    const char *s = strrchr(p, '/');
    const char *b = strrchr(p, '\\');
    if (b && (!s || b > s)) s = b;
    return s ? s + 1 : p;
}

static int cmp_shard_arg(const void *a, const void *b)
{
    const char *pa = *(const char *const *)a;
    const char *pb = *(const char *const *)b;
    int c = strcmp(base_name(pa), base_name(pb));
    return c ? c : strcmp(pa, pb);
}

static uint64_t rd64(const uint8_t *p)
{
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}
static float half_to_float(uint16_t h)
{
    int sign = (h >> 15) & 1, exp = (h >> 10) & 0x1f, man = h & 0x3ff;
    float v;
    if (exp == 0) v = (float)man * 5.9604644775390625e-08f;
    else if (exp == 31) v = man ? NAN : INFINITY;
    else v = ldexpf(1.0f + (float)man / 1024.0f, exp - 15);
    return sign ? -v : v;
}

/* ---------- tensor directory from shard headers -------------------- */

typedef struct { char *name; uint64_t count, block0, nblocks; } TDir;
static TDir g_dir[512];
static int g_ndir = 0;
static uint32_t g_block;
static uint64_t g_nblocks_total;

static void dir_add(const char *name, uint64_t count)
{
    if (g_ndir >= 512) return;
    TDir *t = &g_dir[g_ndir++];
    t->name = (char *)malloc(strlen(name) + 1);
    strcpy(t->name, name);
    t->count = count;
    t->nblocks = g_block ? count / g_block : 0;
    t->block0 = 0;
}

static int shard_scan(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    uint8_t lb[8];
    if (fread(lb, 1, 8, f) != 8) { fclose(f); return -1; }
    uint64_t hlen = rd64(lb);
    if (hlen == 0 || hlen > (1ull << 28)) { fclose(f); return -1; }
    char *hdr = (char *)malloc(hlen + 1);
    if (!hdr || fread(hdr, 1, hlen, f) != hlen) { fclose(f); return -1; }
    hdr[hlen] = 0;
    const char *p = hdr;
    while (*p) {
        while (*p && *p != '"') p++;
        if (!*p) break;
        const char *ks = ++p;
        while (*p && *p != '"') p++;
        size_t klen = (size_t)(p - ks);
        if (!*p) break;
        p++;
        while (*p && *p != '{') p++;
        if (!*p) break;
        const char *obj = p;
        int depth = 0;
        do {
            if (*p == '{') depth++;
            else if (*p == '}') depth--;
            p++;
        } while (*p && depth > 0);
        char key[512];
        if (klen >= sizeof(key)) continue;
        memcpy(key, ks, klen);
        key[klen] = 0;
        if (strcmp(key, "__metadata__") == 0) continue;
        const char *shp = strstr(obj, "\"shape\"");
        if (!shp) continue;
        const char *b = strchr(shp, '[');
        const char *e = b ? strchr(b, ']') : NULL;
        if (!b || !e) continue;
        char sbuf[256];
        size_t sl = (size_t)(e - b - 1);
        if (sl >= sizeof(sbuf)) continue;
        memcpy(sbuf, b + 1, sl);
        sbuf[sl] = 0;
        uint64_t count = 1;
        char *tok = strtok(sbuf, ", ");
        if (!tok) count = 0;
        while (tok) { count *= strtoull(tok, NULL, 10); tok = strtok(NULL, ", "); }
        dir_add(key, count);
    }
    free(hdr);
    fclose(f);
    return 0;
}

static const TDir *dir_find(const char *name)
{
    for (int i = 0; i < g_ndir; i++)
        if (strcmp(g_dir[i].name, name) == 0) return &g_dir[i];
    return NULL;
}

/* ---------- pipeline ------------------------------------------------ */

#define MAX_ORD 512
#define SLOTS 4

typedef struct {
    /* staged raw records (disc loop output) */
    uint8_t *stage_idx[MAX_ORD];
    uint16_t *stage_fr[MAX_ORD];
    int staged[MAX_ORD];
    /* materialized tensors (unpack output), slot = ord % SLOTS */
    float *slot_buf[SLOTS];
    int ready[MAX_ORD];
    int claimed[MAX_ORD];
    int consumed;              /* main loop progress (ordinal) */
    int fetched;               /* disc loop progress */
    int stop;
    pthread_mutex_t lock;
    pthread_cond_t cv;
    /* stats */
    double disc_busy, unpack_busy, main_stall, main_busy;
    uint64_t bytes_packed, weights;
} Pipe;

static Pipe P;
static const TDir *g_demand[MAX_ORD];
static int g_nord = 0;
static int g_lookahead = 4;
static FILE *g_stream;
static uint64_t g_header, g_nb_total;
static const float *g_book;
static uint32_t g_entries;
static const float *g_rows[256];
static int g_variant = 0;   /* 0 baseline, 1 rows+prefetch, 2 entry-major */
/* Block-level prefetch distance in the variant-1 expansion
 * loop: how far ahead of the block being expanded its book
 * row is pulled toward cache. A runtime tunable since the
 * prefetch adaptability item (2026-11-25): --prefetch-dist N
 * or NQ2_PREFETCH_DIST, default 2, the original hard-coded
 * value. A cache hint only; the accumulator cannot depend
 * on it. */
static int g_prefetch_dist = 2;
static __thread uint32_t *g_bucket = NULL;
static __thread uint64_t g_bucket_cap = 0;

/* Short-read policy (Codex Astra M-P8, re-raised as ChatGPT
 * Sol M8; remedy taken under JD's 2026-11-25 delegation):
 * a short read is a hard failure, never a zero-filled success.
 * The old policy let a pack cut by 40 trailing bytes publish
 * a stable-looking accumulator (138.46327352204358 instead of
 * the complete file's value) and exit 0. Now the first short
 * read records its shortfall here, stops the pipeline, and
 * main exits nonzero without printing an accumulator. */
static volatile int g_io_fail = 0;
static char g_io_msg[512];

static void io_fail(const char *plane, const char *tensor,
                    uint64_t want, uint64_t got)
{
    if (!g_io_fail) {
        snprintf(g_io_msg, sizeof(g_io_msg),
                 "nq2-streamtest: short read on the %s (tensor "
                 "%s): wanted %llu, got %llu; the pack is "
                 "incomplete, no accumulator is published\n",
                 plane, tensor,
                 (unsigned long long)want,
                 (unsigned long long)got);
    }
    g_io_fail = 1;
}

static int read_tensor_raw(int ord, uint8_t **idx_out, uint16_t **fr_out)
{
    const TDir *t = g_demand[ord];
    uint64_t nb = t->nblocks, b0 = t->block0;
    /* calloc, not malloc (CODE-REVIEW-2026-11-24, T8). The
     * zeroed buffers are initialization discipline only: since
     * the M-P8 remedy above, a short read fails the run before
     * any unread byte can be consumed. */
    uint8_t *idx = (uint8_t *)calloc(nb, 1);
    uint16_t *fr = (uint16_t *)calloc(nb, 4);
    size_t got;
    fseeko(g_stream, (off_t)(g_header + b0), SEEK_SET);
    got = fread(idx, 1, (size_t)nb, g_stream);
    if (got != nb) {
        io_fail("index plane", t->name, nb, (uint64_t)got);
        free(idx); free(fr);
        return -1;
    }
    fseeko(g_stream, (off_t)(g_header + g_nb_total + 2 * b0), SEEK_SET);
    got = fread(fr, 2, (size_t)nb, g_stream);
    if (got != nb) {
        io_fail("frame min plane", t->name, nb, (uint64_t)got);
        free(idx); free(fr);
        return -1;
    }
    fseeko(g_stream, (off_t)(g_header + 3 * g_nb_total + 2 * b0), SEEK_SET);
    got = fread(fr + nb, 2, (size_t)nb, g_stream);
    if (got != nb) {
        io_fail("frame scale plane", t->name, nb, (uint64_t)got);
        free(idx); free(fr);
        return -1;
    }
    *idx_out = idx;
    *fr_out = fr;
    return 0;
}

static void unpack_tensor(int ord, const uint8_t *idx, const uint16_t *fr,
                          float *dst)
{
    const TDir *t = g_demand[ord];
    uint64_t nb = t->nblocks;
    if (g_variant == 2 && g_entries <= 256) {
        /* Entry-major: bucket the blocks by prototype, then stream
         * each prototype once across all its blocks while it is
         * hot. Same values to the same addresses; only the visit
         * order changes. */
        if (nb > g_bucket_cap) {
            free(g_bucket);
            g_bucket = (uint32_t *)malloc(nb * sizeof(uint32_t));
            g_bucket_cap = nb;
        }
        uint32_t counts[256] = {0};
        uint32_t start[257];
        uint32_t pos[256];
        for (uint64_t b = 0; b < nb; b++) counts[idx[b]]++;
        start[0] = 0;
        for (uint32_t e = 0; e < g_entries; e++) {
            start[e + 1] = start[e] + counts[e];
            pos[e] = start[e];
        }
        for (uint64_t b = 0; b < nb; b++) g_bucket[pos[idx[b]]++] = (uint32_t)b;
        for (uint32_t e = 0; e < g_entries; e++) {
            const float *restrict src = g_rows[e];
            for (uint32_t j = start[e]; j < start[e + 1]; j++) {
                uint64_t b = g_bucket[j];
                float mn = half_to_float(fr[b]);
                float sc = half_to_float(fr[nb + b]);
                float *restrict o = dst + b * g_block;
                for (uint32_t i = 0; i < g_block; i++)
                    o[i] = mn + sc * src[i];
            }
        }
        return;
    }
    for (uint64_t b = 0; b < nb; b++) {
        float mn = half_to_float(fr[b]);
        float sc = half_to_float(fr[nb + b]);
        const float *proto;
        if (g_variant == 1) {
            if (b + (uint64_t)g_prefetch_dist < nb)
                __builtin_prefetch(g_rows[idx[b + (uint64_t)g_prefetch_dist]], 0, 1);
            proto = g_rows[idx[b]];
        } else {
            proto = g_book + (size_t)idx[b] * g_block;
        }
        float *out = dst + b * g_block;
        const float *restrict src = proto;
        float *restrict o = out;
        for (uint32_t i = 0; i < g_block; i++) o[i] = mn + sc * src[i];
    }
}

static void *disc_loop(void *unused)
{
    (void)unused;
    for (int ord = 0; ord < g_nord; ord++) {
        /* lookahead gate: never run more than g_lookahead ahead
         * of consumption; this is the prefetch + predictive grab */
        pthread_mutex_lock(&P.lock);
        while (!P.stop && ord - P.consumed >= g_lookahead)
            pthread_cond_wait(&P.cv, &P.lock);
        pthread_mutex_unlock(&P.lock);
        if (P.stop) return NULL;
        double t0 = now_s();
        uint8_t *idx; uint16_t *fr;
        if (read_tensor_raw(ord, &idx, &fr) != 0) {
            /* M-P8: stop the whole pipeline; main reports the
             * recorded shortfall and exits nonzero. */
            pthread_mutex_lock(&P.lock);
            P.stop = 1;
            pthread_cond_broadcast(&P.cv);
            pthread_mutex_unlock(&P.lock);
            return NULL;
        }
        double dt = now_s() - t0;
        pthread_mutex_lock(&P.lock);
        P.stage_idx[ord] = idx;
        P.stage_fr[ord] = fr;
        P.staged[ord] = 1;
        P.fetched = ord + 1;
        P.disc_busy += dt;
        P.bytes_packed += g_demand[ord]->nblocks * 5;
        pthread_cond_broadcast(&P.cv);
        pthread_mutex_unlock(&P.lock);
    }
    return NULL;
}

static void *unpack_loop(void *unused)
{
    (void)unused;
    for (;;) {
        pthread_mutex_lock(&P.lock);
        int ord = -1;
        for (;;) {
            if (P.stop) { pthread_mutex_unlock(&P.lock); return NULL; }
            /* claim the lowest staged, unclaimed ordinal whose
             * slot is free (its previous tenant consumed) */
            for (int o = 0; o < g_nord; o++) {
                if (P.staged[o] && !P.claimed[o] && !P.ready[o]
                        && o - P.consumed < SLOTS + g_lookahead) {
                    /* slot safety: the slot holds ord-SLOTS until
                     * consumed; require o - SLOTS < consumed or
                     * that ordinal already consumed */
                    if (o < SLOTS || P.consumed > o - SLOTS) { ord = o; break; }
                }
            }
            if (ord >= 0) break;
            if (P.consumed >= g_nord) { pthread_mutex_unlock(&P.lock); return NULL; }
            pthread_cond_wait(&P.cv, &P.lock);
        }
        P.claimed[ord] = 1;
        uint8_t *idx = P.stage_idx[ord];
        uint16_t *fr = P.stage_fr[ord];
        float *dst = P.slot_buf[ord % SLOTS];
        pthread_mutex_unlock(&P.lock);
        double t0 = now_s();
        unpack_tensor(ord, idx, fr, dst);
        double dt = now_s() - t0;
        free(idx); free(fr);
        pthread_mutex_lock(&P.lock);
        P.unpack_busy += dt;
        P.ready[ord] = 1;
        pthread_cond_broadcast(&P.cv);
        pthread_mutex_unlock(&P.lock);
    }
}

static double compute_tensor(const float *w, uint64_t n)
{
    double acc = 0.0;
    for (uint64_t i = 0; i + 1 < n; i++) acc += (double)w[i] * (double)w[i + 1];
    acc += (double)w[n - 1] * (double)w[0];
    return acc;
}

int main(int argc, char **argv)
{
    const char *pack = NULL, *bookp = NULL;
    const char *shards[8];
    int n_shards = 0, unpackers = 2, layers = 26;
    /* Environment layer for the prefetch distance; the CLI
     * flag below overrides it. Strict parse, loud on garbage,
     * same discipline as the Triforce engine. */
    {
        const char *v = getenv("NQ2_PREFETCH_DIST");
        if (v && *v) {
            char *end = NULL;
            long n = strtol(v, &end, 10);
            if (end == v || *end != '\0') {
                fprintf(stderr, "nq2-streamtest: NQ2_PREFETCH_DIST "
                        "'%s' is not a number\n", v);
                return 1;
            }
            g_prefetch_dist = (int)n;
        }
    }
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--pack") == 0 && i + 1 < argc) pack = argv[++i];
        else if (strcmp(argv[i], "--book") == 0 && i + 1 < argc) bookp = argv[++i];
        else if (strcmp(argv[i], "--shards") == 0) {
            while (i + 1 < argc && argv[i + 1][0] != '-' && n_shards < 8)
                shards[n_shards++] = argv[++i];
        }
        else if (strcmp(argv[i], "--unpackers") == 0 && i + 1 < argc) unpackers = atoi(argv[++i]);
        else if (strcmp(argv[i], "--variant") == 0 && i + 1 < argc) g_variant = atoi(argv[++i]);
        else if (strcmp(argv[i], "--lookahead") == 0 && i + 1 < argc) g_lookahead = atoi(argv[++i]);
        else if (strcmp(argv[i], "--prefetch-dist") == 0 && i + 1 < argc) g_prefetch_dist = atoi(argv[++i]);
        else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) layers = atoi(argv[++i]);
    }
    if (g_prefetch_dist < 0) {
        fprintf(stderr, "nq2-streamtest: --prefetch-dist must be >= 0\n");
        return 1;
    }
    if (!pack || !bookp || n_shards == 0) {
        fprintf(stderr, "usage: nq2-streamtest --pack S.nq2 --book S.nq2book "
                "--shards d [...] [--unpackers N] [--lookahead N] "
                "[--prefetch-dist N] [--layers N]\n");
        return 1;
    }
    /* Canonical shard order, matching the writer and the
     * engines, whatever order the shards were passed in. */
    qsort(shards, (size_t)n_shards, sizeof(shards[0]), cmp_shard_arg);
    /* pack header */
    g_stream = fopen(pack, "rb");
    if (!g_stream) { fprintf(stderr, "cannot open pack\n"); return 1; }
    setvbuf(g_stream, NULL, _IOFBF, 1 << 18);
    uint8_t hdr[32];
    if (fread(hdr, 1, 32, g_stream) != 32 || memcmp(hdr, "NQ2\0", 4) != 0) {
        fprintf(stderr, "bad pack\n"); return 1;
    }
    uint32_t version = rd32(hdr + 4);
    g_block = rd32(hdr + 8);
    g_entries = rd32(hdr + 12);
    g_nb_total = rd64(hdr + 24);
    g_header = version == 2 ? 36 : 32;
    if (version != 1) { fprintf(stderr, "streamtest takes v1 packs\n"); return 1; }
    /* M-P8, upfront: the stream must actually hold the bytes
     * its header promises (header + 1 index byte and 2 frame
     * halves of 2 bytes per block). A truncated pack fails
     * here, by name and shortfall, before any tensor is read;
     * the per-read checks in read_tensor_raw are the second
     * line of the same policy. Trailing bytes past the
     * promised extent are tolerated. */
    {
        fseeko(g_stream, 0, SEEK_END);
        off_t psz = ftello(g_stream);
        uint64_t want = g_header + 5 * g_nb_total;
        if (psz < 0 || (uint64_t)psz < want) {
            fprintf(stderr, "nq2-streamtest: pack file is %lld "
                    "bytes but its header promises %llu (header "
                    "%llu + 5 x %llu blocks); the pack is "
                    "incomplete, no accumulator is published\n",
                    (long long)psz, (unsigned long long)want,
                    (unsigned long long)g_header,
                    (unsigned long long)g_nb_total);
            return 1;
        }
        fseeko(g_stream, 0, SEEK_SET);
    }
    /* book (size kept in off_t: the old (long)ftello cast
     * narrows a book above 2 GiB on Windows; Codex Astra
     * review M-P7, 2026-11-25) */
    FILE *bf = fopen(bookp, "rb");
    if (!bf) { fprintf(stderr, "cannot open book\n"); return 1; }
    fseeko(bf, 0, SEEK_END);
    off_t bsz = ftello(bf);
    if (bsz < 12) { fprintf(stderr, "book too small\n"); return 1; }
    fseeko(bf, 0, SEEK_SET);
    uint8_t *bookf = (uint8_t *)malloc((size_t)bsz);
    if (!bookf || fread(bookf, 1, (size_t)bsz, bf) != (size_t)bsz) return 1;
    fclose(bf);
    g_book = (const float *)(bookf + 12);
    for (uint32_t e = 0; e < g_entries && e < 256; e++)
        g_rows[e] = g_book + (size_t)e * g_block;
    /* directory */
    for (int s = 0; s < n_shards; s++)
        if (shard_scan(shards[s]) != 0) { fprintf(stderr, "shard scan failed\n"); return 1; }
    uint64_t run = 0;
    for (int i = 0; i < g_ndir; i++) { g_dir[i].block0 = run; run += g_dir[i].nblocks; }
    g_nblocks_total = run;
    /* demand order: engine forward order over layer tensors */
    static const char *slots[7] = {
        "self_attn.q_proj.weight", "self_attn.k_proj.weight",
        "self_attn.v_proj.weight", "self_attn.o_proj.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight",
        "mlp.down_proj.weight"
    };
    for (int l = 0; l < layers; l++) {
        for (int s = 0; s < 7; s++) {
            char name[256];
            snprintf(name, sizeof(name), "encoder.layers.%d.%s", l, slots[s]);
            const TDir *t = dir_find(name);
            if (!t || t->nblocks == 0) { fprintf(stderr, "missing %s\n", name); return 1; }
            if (g_nord >= MAX_ORD) { fprintf(stderr, "demand exceeds %d ordinals\n", MAX_ORD); return 1; }
            g_demand[g_nord++] = t;
        }
    }
    uint64_t max_elems = 0, tot_w = 0, tot_packed = 0;
    for (int o = 0; o < g_nord; o++) {
        if (g_demand[o]->count > max_elems) max_elems = g_demand[o]->count;
        tot_w += g_demand[o]->count;
        tot_packed += g_demand[o]->nblocks * g_block;
    }
    {
        uint64_t tail_w = 0;
        int tail_n = 0;
        for (int o = 0; o < g_nord; o++) {
            uint64_t t = g_demand[o]->count
                       - g_demand[o]->nblocks * g_block;
            if (t) { tail_w += t; tail_n++; }
        }
        if (tail_n)
            printf("note: %d tensors have %llu tail elements past "
                   "the last whole block; the pack does not cover "
                   "them and the accumulator excludes them\n",
                   tail_n, (unsigned long long)tail_w);
    }
    memset(&P, 0, sizeof(P));
    pthread_mutex_init(&P.lock, NULL);
    pthread_cond_init(&P.cv, NULL);
    P.consumed = 0;
    for (int s = 0; s < SLOTS; s++)
        P.slot_buf[s] = (float *)malloc(max_elems * sizeof(float));
    P.weights = tot_w;

    printf("streamtest: %d tensors in forward order, %llu weights "
           "(%.2f GB expanded); packed extent %llu weights "
           "(%.2f GB)\n", g_nord,
           (unsigned long long)tot_w, (double)tot_w * 4.0 / 1e9,
           (unsigned long long)tot_packed,
           (double)tot_packed * 4.0 / 1e9);
    printf("config: unpackers %d, lookahead %d, slots %d, variant %d\n",
           unpackers, g_lookahead, SLOTS, g_variant);

    double acc = 0.0;
    double wall0 = now_s();
    if (unpackers == 0) {
        /* sequential reference: main does everything itself */
        float *buf = (float *)malloc(max_elems * sizeof(float));
        for (int ord = 0; ord < g_nord; ord++) {
            uint8_t *idx; uint16_t *fr;
            double t0 = now_s();
            if (read_tensor_raw(ord, &idx, &fr) != 0) {
                fprintf(stderr, "%s", g_io_msg);
                free(buf);
                return 1;
            }
            P.disc_busy += now_s() - t0;
            t0 = now_s();
            /* unpack into buf via a one-off call shape */
            {
                const TDir *t = g_demand[ord];
                uint64_t nb = t->nblocks;
                for (uint64_t b = 0; b < nb; b++) {
                    float mn = half_to_float(fr[b]);
                    float sc = half_to_float(fr[nb + b]);
                    const float *proto = g_book + (size_t)idx[b] * g_block;
                    float *o = buf + b * g_block;
                    for (uint32_t i = 0; i < g_block; i++) o[i] = mn + sc * proto[i];
                }
            }
            P.unpack_busy += now_s() - t0;
            t0 = now_s();
            /* Accumulate over exactly the packed extent: the
             * unpacker writes whole blocks only, so the raw
             * count's tail elements were never written (stale
             * or uninitialized memory). For block-aligned
             * tensors the two extents are identical (Codex
             * Astra review M-P6, 2026-11-25). */
            acc += compute_tensor(buf, g_demand[ord]->nblocks
                                       * g_block);
            P.main_busy += now_s() - t0;
            free(idx); free(fr);
            P.consumed = ord + 1;
        }
        free(buf);
    } else {
        pthread_t dth, uth[4];
        pthread_create(&dth, NULL, disc_loop, NULL);
        for (int u = 0; u < unpackers && u < 4; u++)
            pthread_create(&uth[u], NULL, unpack_loop, NULL);
        for (int ord = 0; ord < g_nord; ord++) {
            double tw = now_s();
            pthread_mutex_lock(&P.lock);
            while (!P.ready[ord] && !P.stop)
                pthread_cond_wait(&P.cv, &P.lock);
            int have = P.ready[ord];
            pthread_mutex_unlock(&P.lock);
            if (!have) break;   /* disc loop stopped: short read */
            P.main_stall += now_s() - tw;
            double t0 = now_s();
            /* Packed extent only, as in the serial path above
             * (M-P6): the slot buffer's tail past the last
             * whole block holds a previous tensor's values. */
            acc += compute_tensor(P.slot_buf[ord % SLOTS],
                                  g_demand[ord]->nblocks * g_block);
            P.main_busy += now_s() - t0;
            pthread_mutex_lock(&P.lock);
            P.consumed = ord + 1;
            pthread_cond_broadcast(&P.cv);
            pthread_mutex_unlock(&P.lock);
        }
        pthread_mutex_lock(&P.lock);
        P.stop = 1;
        pthread_cond_broadcast(&P.cv);
        pthread_mutex_unlock(&P.lock);
        pthread_join(dth, NULL);
        for (int u = 0; u < unpackers && u < 4; u++) pthread_join(uth[u], NULL);
    }
    if (g_io_fail) {
        fprintf(stderr, "%s", g_io_msg);
        return 1;
    }
    double wall = now_s() - wall0;
    /* The expanded rate is computed from the packed extent
     * (Sol M9): the raw count includes tail elements the pack
     * never produces and the pipeline never expands, so
     * dividing by it overstated the rate. The accumulator is
     * untouched: this is accounting only. */
    printf("wall: %.2f s; expanded rate: %.1f M weights/sec (%.0f MB/sec)\n",
           wall, (double)tot_packed / wall / 1e6,
           (double)tot_packed * 4.0 / wall / 1e6);
    printf("disc busy: %.2f s; unpack busy (summed): %.2f s; "
           "main busy: %.2f s; main stall: %.2f s (%.1f%% of wall)\n",
           P.disc_busy, P.unpack_busy, P.main_busy, P.main_stall,
           100.0 * P.main_stall / wall);
    printf("accumulator: %.17g\n", acc);
    return 0;
}
