/* nq2-quantize.c ... NanoQuatTwo quantizer, in pure C.
 *
 * Implemented by Elfie (E.L.F.I.E.): the quantizer and this code.
 * Direction and executive planning by JD.
 *
 * Takes a big model and turns it into a little model: an index
 * stream (.nq2) plus a memory file (.nq2book) that lives beside
 * it. At load time the weights are extracted back out of memory
 * by the streaming runner (nq2-run.c).
 *
 * How it works:
 *   - Weights travel in blocks of BLOCK consecutive values.
 *   - Each block is normalized into an affine frame: a minimum
 *     and a scale, stored as float16 (4 bytes per block).
 *   - A codebook of ENTRIES prototypical blocks is learned from
 *     the model's own blocks by k-means over a random sample.
 *   - Every block is then packed to one 8-bit index: the
 *     prototype it is nearest to, in normalized space.
 *
 * Input is Hugging Face safetensors shards, read directly: the
 * shard header is JSON, parsed by hand below (safetensors
 * headers are simple: no braces inside strings). dtypes BF16,
 * F16, and F32 are supported.
 *
 * Usage:
 *   nq2-quantize --out PREFIX shard1.safetensors [shard2 ...]
 *       [--block 256] [--entries 256] [--budget 250000000]
 *       [--filter layers.0.,layers.1.] [--sample-blocks 32768]
 *       [--iters 8]
 *
 * Plain hosted C99. No dependencies beyond the standard library.
 */

#define _POSIX_C_SOURCE 200809L   /* fseeko/ftello declarations */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#ifdef _WIN32
#include <process.h>    /* _getpid */
#include <windows.h>    /* threads (2026-11-26) */
#else
#include <unistd.h>     /* getpid */
#include <pthread.h>    /* threads (2026-11-26) */
#endif

/* Seek to a 64-bit file offset. Plain fseek takes a long, which
 * is 32 bits wide on Windows, and the shards this tool reads run
 * past 5 GB: use the 64-bit entry points where they exist. */
static int seek_to(FILE *f, uint64_t off)
{
#ifdef _WIN32
    return _fseeki64(f, (long long)off, SEEK_SET);
#else
    return fseeko(f, (off_t)off, SEEK_SET);
#endif
}

/* Current size of an open file, 64-bit clean, or 0 on failure.
 * Leaves the file positioned at the end; callers seek back. */
static uint64_t file_size(FILE *f)
{
#ifdef _WIN32
    if (_fseeki64(f, 0, SEEK_END) != 0) return 0;
    long long s = _ftelli64(f);
    return s < 0 ? 0 : (uint64_t)s;
#else
    if (fseeko(f, 0, SEEK_END) != 0) return 0;
    off_t s = ftello(f);
    return s < 0 ? 0 : (uint64_t)s;
#endif
}

/* ---------------- small utilities ---------------- */

/* Thread-local so parallel per-tensor workers each replay their own
 * tensor's exact serial draws (see per_tensor_parallel). */
static __thread uint64_t rng_state = 0x2545F4914F6CDD1Dull;
static uint64_t rng_next(void)
{
    uint64_t x = rng_state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    rng_state = x;
    return x * 0x2545F4914F6CDD1Dull;
}
static double dsqrt(double x)  /* Newton; avoids libm entirely */
{
    if (x <= 0.0) return 0.0;
    double y = x > 1.0 ? x : 1.0;
    for (int i = 0; i < 60; i++) y = 0.5 * (y + x / y);
    return y;
}

/* ---------------- native threads, no mutexes ----------------
 * Added 2026-11-26 (co-implemented by Cortana at JD's direction;
 * the quantizer itself is Elfie's). Win32 threads on Windows,
 * pthreads elsewhere. Work is handed out as disjoint chunks via a
 * lock-free counter; nothing is ever locked. */
static int g_threads = 1;   /* --threads N; main() sets logical CPUs if 0 */
static int g_learn_threads = 0; /* 0 = g_threads; 1 inside per-tensor workers */

static int cpu_count(void)
{
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwNumberOfProcessors > 0 ? (int)si.dwNumberOfProcessors : 1;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
#endif
}

static uint64_t atomic_next(uint64_t *p)
{
#if defined(__GNUC__)
    return __atomic_fetch_add(p, 1, __ATOMIC_RELAXED);
#else
    return (uint64_t)InterlockedIncrement64((volatile LONG64 *)p) - 1;
#endif
}

typedef void (*WorkFn)(int tid, void *shared);
typedef struct { WorkFn fn; void *shared; int tid; } WorkArg;

#ifdef _WIN32
static DWORD WINAPI work_tramp(LPVOID p)
{
    WorkArg *a = (WorkArg *)p;
    a->fn(a->tid, a->shared);
    return 0;
}
#else
static void *work_tramp(void *p)
{
    WorkArg *a = (WorkArg *)p;
    a->fn(a->tid, a->shared);
    return NULL;
}
#endif

/* Run fn on n workers (worker 0 is the caller) and join them all.
 * Workers claim chunks dynamically, so a thread that fails to start
 * costs speed, never correctness. */
static void run_parallel(int n, WorkFn fn, void *shared)
{
    if (n <= 1) { fn(0, shared); return; }
    WorkArg *args = (WorkArg *)calloc((size_t)n, sizeof(WorkArg));
#ifdef _WIN32
    HANDLE *th = (HANDLE *)calloc((size_t)n, sizeof(HANDLE));
#else
    pthread_t *th = (pthread_t *)calloc((size_t)n, sizeof(pthread_t));
    int *ok = (int *)calloc((size_t)n, sizeof(int));
#endif
    if (!args || !th) { fprintf(stderr, "out of memory\n"); exit(1); }
    for (int t = 1; t < n; t++) {
        args[t].fn = fn; args[t].shared = shared; args[t].tid = t;
#ifdef _WIN32
        th[t] = CreateThread(NULL, 0, work_tramp, &args[t], 0, NULL);
#else
        ok[t] = (pthread_create(&th[t], NULL, work_tramp, &args[t]) == 0);
#endif
    }
    fn(0, shared);
    for (int t = 1; t < n; t++) {
#ifdef _WIN32
        if (th[t]) { WaitForSingleObject(th[t], INFINITE); CloseHandle(th[t]); }
#else
        if (ok[t]) pthread_join(th[t], NULL);
#endif
    }
    free(args); free(th);
#ifndef _WIN32
    free(ok);
#endif
}

static uint16_t float_to_half(float f)
{
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int exp = (int)((x >> 23) & 0xffu) - 112;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        int shift = 14 - exp;
        uint32_t half = mant >> shift;
        /* round to nearest, ties to even: the mask must cover
         * only the bits BELOW the round bit, or every tie (and
         * every value with the round bit set) rounds up */
        if ((mant >> (shift - 1)) & 1u
                && ((half & 1u) || (mant & ((1u << (shift - 1)) - 1))))
            half++;
        return (uint16_t)(sign | half);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0));
    if (mant & 0x1000u && ((mant & 0x2000u) || (mant & 0xfffu))) mant += 0x2000u;
    if (mant & 0x800000u) { mant = 0; exp++; if (exp >= 31) return (uint16_t)(sign | 0x7c00u); }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

static float bf16_to_float(uint16_t h)
{
    uint32_t bits = (uint32_t)h << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static float f16_to_float(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (!mant) bits = sign;
        else {
            exp = 1;
            while (!(mant & 0x400u)) { mant <<= 1; exp--; }
            mant &= 0x3ffu;
            bits = sign | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) bits = sign | 0x7f800000u | (mant << 13);
    else bits = sign | ((exp + 112) << 23) | (mant << 13);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

/* ---------------- safetensors reading ---------------- */

enum { DT_BF16, DT_F16, DT_F32 };

/* --skip-unsupported (SuperVL work, 2026-11-27): tensors whose dtype is
 * not BF16, F16 or F32 (for example the I64 summary_idxs of the RADIO
 * tower) are left out of the pack with a notice instead of rejecting the
 * whole model. Off by default: the old refusal stands. */
static int g_skip_unsupported = 0;
static int g_n_skipped = 0;

typedef struct {
    char *name;
    int dtype;
    uint64_t count;        /* number of weights              */
    uint64_t begin, end;   /* byte offsets within the shard  */
} TensorInfo;

typedef struct {
    char *path;
    FILE *file;
    uint64_t data_start;   /* where tensor bytes begin       */
    TensorInfo *tensors;
    int n_tensors;
} Shard;

/* Find the closing quote of a JSON string starting at s[0]=='"',
 * honoring backslash escapes. Returns index of the closing quote. */
static long json_string_end(const char *s, long len)
{
    for (long i = 1; i < len; i++) {
        if (s[i] == '\\') { i++; continue; }
        if (s[i] == '"') return i;
    }
    return -1;
}

/* Parse one shard header. The header is a flat JSON object whose
 * values are either the __metadata__ object or a tensor object
 * holding "dtype" and "data_offsets". We walk the top-level keys
 * and, for each object value, pull those two fields out by name. */
static int shard_open(Shard *sh, const char *path)
{
    memset(sh, 0, sizeof(*sh));
    sh->path = (char *)malloc(strlen(path) + 1);
    strcpy(sh->path, path);
    sh->file = fopen(path, "rb");
    if (!sh->file) { fprintf(stderr, "cannot open %s\n", path); return -1; }
    uint8_t lenbuf[8];
    if (fread(lenbuf, 1, 8, sh->file) != 8) return -1;
    uint64_t hlen = 0;
    for (int i = 0; i < 8; i++) hlen |= (uint64_t)lenbuf[i] << (8 * i);
    if (hlen == 0 || hlen > (1ull << 28)) return -1;
    char *hdr = (char *)malloc(hlen + 1);
    if (!hdr) return -1;
    if (fread(hdr, 1, hlen, sh->file) != hlen) { free(hdr); return -1; }
    hdr[hlen] = '\0';
    sh->data_start = 8 + hlen;
    uint64_t fsize = file_size(sh->file);
    if (fsize == 0 || seek_to(sh->file, sh->data_start) != 0) {
        free(hdr);
        return -1;
    }

    int cap = 64;
    sh->tensors = (TensorInfo *)malloc(sizeof(TensorInfo) * cap);
    long pos = 0, n = (long)hlen;
    while (pos < n) {
        if (hdr[pos] != '"') { pos++; continue; }
        long kend = json_string_end(hdr + pos, n - pos);
        if (kend < 0) break;
        long klen = kend - 1;
        /* key is hdr[pos+1 .. pos+klen]; expect ':' then a value */
        long p = pos + kend + 1;
        while (p < n && (hdr[p] == ' ' || hdr[p] == ':' || hdr[p] == '\t')) p++;
        if (p >= n || hdr[p] != '{') { pos = p; continue; }
        /* find this object's end by brace counting */
        long q = p, depth = 0, obj_end = -1;
        while (q < n) {
            if (hdr[q] == '"') {
                long e = json_string_end(hdr + q, n - q);
                if (e < 0) break;
                q += e + 1;
                continue;
            }
            if (hdr[q] == '{') depth++;
            if (hdr[q] == '}') { depth--; if (depth == 0) { obj_end = q; break; } }
            q++;
        }
        if (obj_end < 0) break;
        /* tensor object? it has "dtype" inside */
        char key[256];
        if (klen > 250) klen = 250;
        memcpy(key, hdr + pos + 1, klen);
        key[klen] = '\0';
        if (strcmp(key, "__metadata__") != 0) {
            const char *obj = hdr + p;
            long olen = obj_end - p + 1;
            const char *dt = strstr(obj, "\"dtype\"");
            const char *doff = strstr(obj, "\"data_offsets\"");
            if (dt && doff && (dt - obj) < olen && (doff - obj) < olen) {
                TensorInfo *t;
                if (sh->n_tensors == cap) {
                    cap *= 2;
                    sh->tensors = (TensorInfo *)realloc(
                        sh->tensors, sizeof(TensorInfo) * cap);
                }
                t = &sh->tensors[sh->n_tensors];
                t->name = (char *)malloc(strlen(key) + 1);
                strcpy(t->name, key);
                const char *colon = strchr(dt + 7, ':');
                const char *q1 = colon ? strchr(colon, '"') : NULL;
                const char *q2 = q1 ? strchr(q1 + 1, '"') : NULL;
                char dts[8] = {0};
                if (q1 && q2 && q2 - q1 - 1 >= 1 && q2 - q1 - 1 <= 7)
                    memcpy(dts, q1 + 1, q2 - q1 - 1);
                if (strcmp(dts, "BF16") == 0) t->dtype = DT_BF16;
                else if (strcmp(dts, "F16") == 0) t->dtype = DT_F16;
                else if (strcmp(dts, "F32") == 0) t->dtype = DT_F32;
                else if (g_skip_unsupported) {
                    fprintf(stderr, "skip: tensor %s has dtype '%s' "
                            "(not a weight dtype; left out of the "
                            "pack)\n", key, dts);
                    g_n_skipped++;
                    free(t->name);
                    pos = obj_end + 1;
                    continue;
                }
                else {
                    /* Unsupported dtypes are rejected, never
                     * reinterpreted: an I32 tensor's bit patterns
                     * are not F32 weights, and packing them as if
                     * they were silently corrupts the model and
                     * its rate accounting (Codex Astra review
                     * M-P4, 2026-11-25). */
                    fprintf(stderr, "reject: tensor %s has "
                            "unsupported dtype '%s' (only BF16, "
                            "F16, F32 pack)\n", key, dts);
                    free(hdr);
                    return -1;
                }
                const char *br = strchr(doff, '[');
                t->begin = br ? strtoull(br + 1, NULL, 10) : 0;
                const char *comma = br ? strchr(br, ',') : NULL;
                t->end = comma ? strtoull(comma + 1, NULL, 10) : 0;
                /* Span validation (Archie P0-4): end < begin would
                 * underflow the count below, and a span running
                 * past the end of the file would read out of the
                 * shard. This tool parses no shape, so the count
                 * comes from the span; the file is the authority
                 * the span is checked against. */
                if (t->end < t->begin || t->end > fsize
                    || sh->data_start + t->end > fsize) {
                    fprintf(stderr, "reject: tensor %s has "
                            "data_offsets [%llu, %llu) outside a "
                            "%llu-byte shard\n", key,
                            (unsigned long long)t->begin,
                            (unsigned long long)t->end,
                            (unsigned long long)fsize);
                    free(hdr);
                    return -1;
                }
                uint64_t esz = t->dtype == DT_F32 ? 4 : 2;
                t->count = (t->end - t->begin) / esz;
                sh->n_tensors++;
            }
        }
        pos = obj_end + 1;
    }
    free(hdr);
    return 0;
}

/* Read count weights starting at weight index start from a tensor,
 * converting to float. Reads the raw bytes in one fread. */
/* Per-thread shard handles for the parallel per-tensor path; the main
 * thread (t_fh == NULL) keeps using sh->file as before. */
static Shard *g_shards;
static int g_n_shards;
static __thread FILE **t_fh;
static FILE *tensor_fh(Shard *sh)
{
    if (!t_fh) return sh->file;
    int i = (int)(sh - g_shards);
    if (!t_fh[i]) t_fh[i] = fopen(sh->path, "rb");
    return t_fh[i];
}

static int tensor_read(Shard *sh, TensorInfo *t, uint64_t start,
                       uint64_t count, float *dst)
{
    uint64_t esz = t->dtype == DT_F32 ? 4 : 2;
    uint64_t byte_off = sh->data_start + t->begin + start * esz;
    uint8_t *raw = (uint8_t *)malloc(count * esz);
    if (!raw) return -1;
    FILE *fh = tensor_fh(sh);
    if (!fh) { free(raw); return -1; }
    if (seek_to(fh, byte_off) != 0) { free(raw); return -1; }
    if (fread(raw, esz, count, fh) != count) { free(raw); return -1; }
    for (uint64_t i = 0; i < count; i++) {
        if (t->dtype == DT_F32) {
            memcpy(&dst[i], raw + i * 4, 4);
        } else {
            uint16_t h = (uint16_t)(raw[i * 2] | (raw[i * 2 + 1] << 8));
            dst[i] = t->dtype == DT_BF16 ? bf16_to_float(h) : f16_to_float(h);
        }
    }
    free(raw);
    return 0;
}

/* ---------------- scope and block walking ---------------- */

typedef struct {
    Shard *shard;
    TensorInfo *tensor;
} ScopeItem;

static ScopeItem *g_scope;
static int g_n_scope;
static uint64_t g_total_weights;   /* weights that land in blocks */
static uint64_t g_total_blocks;

static int name_in_filter(const char *name, const char *filter)
{
    if (!filter || !*filter) return 1;
    char buf[1024];
    strncpy(buf, filter, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
    for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
        if (strstr(name, tok)) return 1;
    }
    return 0;
}

typedef void (*BlockFn)(const float *raw_block, void *ctx);

/* Walk every block of every in-scope tensor, slab by slab, and
 * hand each raw block to fn. Tail weights that do not fill a
 * block are not packed (counted and reported by main). */
static void walk_blocks(uint32_t block, BlockFn fn, void *ctx)
{
    const uint64_t SLAB = 4096;   /* blocks per read slab */
    float *buf = (float *)malloc(SLAB * block * sizeof(float));
    if (!buf) { fprintf(stderr, "out of memory\n"); exit(1); }
    for (int s = 0; s < g_n_scope; s++) {
        Shard *sh = g_scope[s].shard;
        TensorInfo *t = g_scope[s].tensor;
        uint64_t tb = t->count / block;
        for (uint64_t base = 0; base < tb; base += SLAB) {
            uint64_t want = tb - base;
            if (want > SLAB) want = SLAB;
            if (tensor_read(sh, t, base * block, want * block, buf) != 0) {
                fprintf(stderr, "read failed on %s\n", t->name);
                free(buf);
                exit(1);
            }
            for (uint64_t b = 0; b < want; b++)
                fn(buf + b * block, ctx);
        }
    }
    free(buf);
}

/* Normalization mode: 0 = min-max (default; normalized values in
 * [0,1], frame = min + range), 1 = RMS (no offset; normalized =
 * raw / block RMS, frame stored as min 0 + scale RMS, so the stream
 * format and its reconstruction min + scale * prototype are
 * unchanged and existing readers handle RMS packs unmodified). */
static int g_norm_rms = 0;
/* PQ book learning runs one k-means per subvector; the
 * per-iteration line is noise at that multiplicity, so the
 * PQ path quiets it and prints one summary per tensor. */
static int g_kmeans_verbose = 1;

/* Normalize one block into out; also returns the block's frame
 * (minimum and scale) through mn_out / sc_out. */
static void normalize_block(const float *raw, uint32_t block,
                            float *out, float *mn_out, float *sc_out)
{
    /* A zero-range (minmax) or zero-RMS block gets a zero scale
     * and a zero normalized vector (Codex Astra review H3,
     * 2026-11-25). The old substitution of scale 1.0 let the
     * nearest shared centroid manufacture nonzero weights out
     * of an all-zero block at decode. With scale 0 the existing
     * reconstruction mn + 0 * prototype yields exactly mn: the
     * zero vector for RMS, and for minmax the constant itself,
     * preserved to the precision of the stored offset. No
     * format change: readers already multiply by the scale. */
    if (g_norm_rms) {
        double s = 0.0;
        for (uint32_t i = 0; i < block; i++) s += (double)raw[i] * raw[i];
        float sc = (float)dsqrt(s / block);
        if (sc == 0.0f) {
            for (uint32_t i = 0; i < block; i++) out[i] = 0.0f;
        } else {
            for (uint32_t i = 0; i < block; i++) out[i] = raw[i] / sc;
        }
        *mn_out = 0.0f;
        *sc_out = sc;
        return;
    }
    float mn = raw[0], mx = raw[0];
    for (uint32_t i = 1; i < block; i++) {
        if (raw[i] < mn) mn = raw[i];
        if (raw[i] > mx) mx = raw[i];
    }
    float sc = mx - mn;
    if (sc == 0.0f) {
        for (uint32_t i = 0; i < block; i++) out[i] = 0.0f;
    } else {
        for (uint32_t i = 0; i < block; i++) out[i] = (raw[i] - mn) / sc;
    }
    *mn_out = mn;
    *sc_out = sc;
}

/* Nearest prototype by squared error. Four centroids are scored per
 * pass over x, each in its own accumulator, so the four dependent
 * double-add chains overlap in the core. Every centroid's sum is
 * still accumulated in the original order i = 0..block-1 and the
 * winner is chosen in ascending k with a strict <, so the result is
 * bit-identical to the one-at-a-time loop (2026-11-26). */
static int nearest_centroid(const float *cent, uint32_t entries,
                            uint32_t block, const float *x)
{
    int best = 0;
    double best_err = 1e30;
    uint32_t k = 0;
    for (; k + 4 <= entries; k += 4) {
        const float *c0 = cent + (size_t)k * block;
        const float *c1 = c0 + block;
        const float *c2 = c1 + block;
        const float *c3 = c2 + block;
        double e0 = 0.0, e1 = 0.0, e2 = 0.0, e3 = 0.0;
        for (uint32_t i = 0; i < block; i++) {
            double xi = (double)x[i];
            double d0 = xi - c0[i];
            double d1 = xi - c1[i];
            double d2 = xi - c2[i];
            double d3 = xi - c3[i];
            e0 += d0 * d0;
            e1 += d1 * d1;
            e2 += d2 * d2;
            e3 += d3 * d3;
        }
        if (e0 < best_err) { best_err = e0; best = (int)k; }
        if (e1 < best_err) { best_err = e1; best = (int)(k + 1); }
        if (e2 < best_err) { best_err = e2; best = (int)(k + 2); }
        if (e3 < best_err) { best_err = e3; best = (int)(k + 3); }
    }
    for (; k < entries; k++) {
        const float *c = cent + (size_t)k * block;
        double err = 0.0;
        for (uint32_t i = 0; i < block; i++) {
            double d = (double)x[i] - c[i];
            err += d * d;
        }
        if (err < best_err) { best_err = err; best = (int)k; }
    }
    return best;
}

/* Threaded assignment of sample blocks to prototypes (k-means). */
typedef struct {
    const float *cent;
    uint32_t entries, block;
    const float *x;
    uint64_t n;
    uint8_t *out;
    uint64_t next;            /* lock-free chunk counter */
} AssignJob;

#define ASSIGN_CHUNK 64u

static void assign_worker(int tid, void *vj)
{
    AssignJob *j = (AssignJob *)vj;
    uint64_t nch = (j->n + ASSIGN_CHUNK - 1) / ASSIGN_CHUNK;
    (void)tid;
    for (;;) {
        uint64_t c = atomic_next(&j->next);
        if (c >= nch) break;
        uint64_t b0 = c * ASSIGN_CHUNK, b1 = b0 + ASSIGN_CHUNK;
        if (b1 > j->n) b1 = j->n;
        for (uint64_t b = b0; b < b1; b++)
            j->out[b] = (uint8_t)nearest_centroid(j->cent, j->entries,
                                  j->block, j->x + b * j->block);
    }
}

/* Walk the blocks of a single scope tensor (used by the
 * per-tensor books path, which samples, learns, and packs one
 * tensor at a time). */
static void walk_one_tensor(int scope_idx, uint32_t block,
                            BlockFn fn, void *ctx)
{
    const uint64_t SLAB = 4096;
    float *buf = (float *)malloc(SLAB * block * sizeof(float));
    if (!buf) { fprintf(stderr, "out of memory\n"); exit(1); }
    Shard *sh = g_scope[scope_idx].shard;
    TensorInfo *t = g_scope[scope_idx].tensor;
    uint64_t tb = t->count / block;
    for (uint64_t base = 0; base < tb; base += SLAB) {
        uint64_t want = tb - base;
        if (want > SLAB) want = SLAB;
        if (tensor_read(sh, t, base * block, want * block, buf) != 0) {
            fprintf(stderr, "read failed on %s\n", t->name);
            free(buf);
            exit(1);
        }
        for (uint64_t b = 0; b < want; b++)
            fn(buf + b * block, ctx);
    }
    free(buf);
}

/* Learn a codebook by k-means (Lloyd's algorithm) over a sample
 * of normalized blocks. cent must hold entries * block floats. */
static void learn_book(const float *sample, uint64_t sample_n,
                       uint32_t block, uint32_t entries, int iters,
                       float *cent)
{
    /* Precondition guard: the initializer below divides by
     * sample_n (Codex Astra review M-P3, 2026-11-25). */
    if (sample_n == 0) {
        fprintf(stderr, "learn_book: empty training sample\n");
        exit(1);
    }
    for (uint32_t k = 0; k < entries; k++) {
        uint64_t src = rng_next() % sample_n;
        memcpy(cent + (size_t)k * block, sample + src * block,
               block * sizeof(float));
    }
    double *sums = (double *)calloc((size_t)entries * block, sizeof(double));
    uint64_t *counts = (uint64_t *)calloc(entries, sizeof(uint64_t));
    uint8_t *assign = (uint8_t *)malloc((size_t)sample_n);
    if (!sums || !counts || !assign) { fprintf(stderr, "out of memory\n"); exit(1); }
    /* Threaded assignment (2026-11-26): the nearest-centroid search is
     * the cost; it writes one byte per sample block. The accumulation
     * below stays serial in sample order, so the double sums, and the
     * book, are bit-identical to the single-threaded original. */
    AssignJob job;
    memset(&job, 0, sizeof(job));
    job.cent = cent; job.entries = entries; job.block = block;
    job.x = sample; job.n = sample_n; job.out = assign;
    int lt = g_learn_threads ? g_learn_threads : g_threads;
    int nt = sample_n >= 512 ? lt : 1;
    if (nt < 1) nt = 1;
    for (int it = 0; it < iters; it++) {
        memset(sums, 0, (size_t)entries * block * sizeof(double));
        memset(counts, 0, entries * sizeof(uint64_t));
        job.next = 0;
        run_parallel(nt, assign_worker, &job);
        for (uint64_t b = 0; b < sample_n; b++) {
            const float *x = sample + b * block;
            int k = assign[b];
            counts[k]++;
            double *sum = sums + (size_t)k * block;
            for (uint32_t i = 0; i < block; i++) sum[i] += x[i];
        }
        double shift = 0.0;
        for (uint32_t k = 0; k < entries; k++) {
            if (!counts[k]) continue;
            float *c = cent + (size_t)k * block;
            for (uint32_t i = 0; i < block; i++) {
                double nv = sums[(size_t)k * block + i] / (double)counts[k];
                double d = nv - c[i];
                if (d < 0) d = -d;
                if (d > shift) shift = d;
                c[i] = (float)nv;
            }
        }
        if (g_kmeans_verbose)
            printf("  kmeans iter %d: max centroid shift %.5f\n", it, shift);
    }
    free(sums);
    free(counts);
    free(assign);
}

/* ---------------- pass 1: reservoir sampling ---------------- */

typedef struct {
    uint32_t block;
    float *sample;        /* sample_cap * block, normalized */
    uint64_t sample_cap;
    uint64_t sample_n;
    uint64_t seen;
    float *tmp;
} SampleCtx;

static void sample_offer(SampleCtx *ctx, const float *normalized)
{
    if (ctx->sample_n < ctx->sample_cap) {
        memcpy(ctx->sample + ctx->sample_n * ctx->block, normalized,
               ctx->block * sizeof(float));
        ctx->sample_n++;
    } else {
        uint64_t j = rng_next() % (ctx->seen + 1);
        if (j < ctx->sample_cap)
            memcpy(ctx->sample + j * ctx->block, normalized,
                   ctx->block * sizeof(float));
    }
    ctx->seen++;
}

static void sample_fn(const float *raw, void *vctx)
{
    SampleCtx *ctx = (SampleCtx *)vctx;
    float mn, sc;
    normalize_block(raw, ctx->block, ctx->tmp, &mn, &sc);
    sample_offer(ctx, ctx->tmp);
}

/* ---------------- pass 2: packing ---------------- */

typedef struct {
    uint32_t block, entries;
    const float *cent;
    uint8_t *idx;         /* g_total_blocks */
    uint16_t *mn_h;       /* g_total_blocks */
    uint16_t *sc_h;       /* g_total_blocks */
    uint64_t cursor;
    double err2, ref2;
    float *tmp;
    /* First block whose serialized frame is not finite in
     * float16 (Sol H8); the callers refuse publication. */
    int frame_bad;
    float bad_value;
    uint64_t bad_block;
} PackCtx;

static void pack_fn(const float *raw, void *vctx)
{
    PackCtx *ctx = (PackCtx *)vctx;
    float mn, sc;
    normalize_block(raw, ctx->block, ctx->tmp, &mn, &sc);
    int k = nearest_centroid(ctx->cent, ctx->entries, ctx->block, ctx->tmp);
    uint64_t at = ctx->cursor++;
    ctx->idx[at] = (uint8_t)k;
    ctx->mn_h[at] = float_to_half(mn);
    ctx->sc_h[at] = float_to_half(sc);
    /* The frame is what the format actually stores. A finite
     * weight past the float16 range (or a non-finite source
     * value) converts to an Inf/NaN half pattern, and the
     * block would expand to NaNs while every tool downstream
     * reports success (ChatGPT Sol review H8, 2026-11-25).
     * Record the first offender; the callers check after the
     * walk and refuse to publish. */
    if (!ctx->frame_bad
            && (((ctx->mn_h[at] & 0x7c00u) == 0x7c00u)
                || ((ctx->sc_h[at] & 0x7c00u) == 0x7c00u))) {
        ctx->frame_bad = 1;
        ctx->bad_block = at;
        ctx->bad_value = ((ctx->mn_h[at] & 0x7c00u) == 0x7c00u)
                         ? mn : sc;
    }
    /* Measure fidelity the way the runner will see it: frames
     * rounded to float16, prototype from the book. */
    float rmn = f16_to_float(ctx->mn_h[at]);
    float rsc = f16_to_float(ctx->sc_h[at]);
    const float *c = ctx->cent + (size_t)k * ctx->block;
    for (uint32_t i = 0; i < ctx->block; i++) {
        double rec = (double)rmn + (double)rsc * c[i];
        double d = (double)raw[i] - rec;
        ctx->err2 += d * d;
        ctx->ref2 += (double)raw[i] * raw[i];
    }
}

/* ---------------- threaded global-book passes ----------------
 * Added 2026-11-26 (co-implemented by Cortana at JD's direction).
 * Output is byte-identical for any thread count:
 *   - sampling keeps the single-threaded RNG; it replays the
 *     reservoir decisions first (cheap, no I/O), then reads only
 *     the blocks that survive into the sample,
 *   - packing hands disjoint chunks of the global block range to
 *     workers; every block writes only its own idx/mn/sc slots,
 *   - fidelity sums are per chunk, reduced in chunk order after
 *     the join.
 * No mutexes: the only shared mutable word is a lock-free chunk
 * counter. Each worker opens its own FILE per shard. */

static Shard *g_shards;
static int g_n_shards;

typedef struct { uint64_t gidx; uint64_t slot; } SampEvt;

static void sample_global_fast(SampleCtx *ctx)
{
    uint64_t N = g_total_blocks, cap = ctx->sample_cap;
    size_t ecap = 1u << 20, en = 0;
    SampEvt *ev = (SampEvt *)malloc(ecap * sizeof(SampEvt));
    if (!ev) { fprintf(stderr, "out of memory sampling\n"); exit(1); }
    /* Replay sample_offer(): same RNG calls in the same order. */
    for (uint64_t seen = 0; seen < N; seen++) {
        uint64_t slot;
        if (seen < cap) slot = seen;
        else {
            uint64_t j = rng_next() % (seen + 1);
            if (j >= cap) continue;
            slot = j;
        }
        if (en == ecap) {
            ecap *= 2;
            ev = (SampEvt *)realloc(ev, ecap * sizeof(SampEvt));
            if (!ev) { fprintf(stderr, "out of memory sampling\n"); exit(1); }
        }
        ev[en].gidx = seen; ev[en].slot = slot; en++;
    }
    ctx->sample_n = N < cap ? N : cap;
    ctx->seen = N;
    float *raw = (float *)malloc(ctx->block * sizeof(float));
    if (!raw) { fprintf(stderr, "out of memory sampling\n"); exit(1); }
    int s = 0;
    uint64_t off = 0;
    for (size_t e = 0; e < en; e++) {
        while (s < g_n_scope
               && ev[e].gidx >= off + g_scope[s].tensor->count / ctx->block) {
            off += g_scope[s].tensor->count / ctx->block;
            s++;
        }
        if (s >= g_n_scope) { fprintf(stderr, "sampling index out of range\n"); exit(1); }
        if (tensor_read(g_scope[s].shard, g_scope[s].tensor,
                        (ev[e].gidx - off) * ctx->block, ctx->block, raw) != 0) {
            fprintf(stderr, "read failed on %s\n", g_scope[s].tensor->name);
            exit(1);
        }
        float mn, sc;
        normalize_block(raw, ctx->block,
                        ctx->sample + ev[e].slot * ctx->block, &mn, &sc);
    }
    free(raw);
    free(ev);
}

static int tensor_read_fh(FILE *f, const Shard *sh, const TensorInfo *t,
                          uint64_t start, uint64_t count, float *dst,
                          uint8_t *raw)
{
    uint64_t esz = t->dtype == DT_F32 ? 4 : 2;
    uint64_t byte_off = sh->data_start + t->begin + start * esz;
    if (seek_to(f, byte_off) != 0) return -1;
    if (fread(raw, esz, count, f) != count) return -1;
    for (uint64_t i = 0; i < count; i++) {
        if (t->dtype == DT_F32) {
            memcpy(&dst[i], raw + i * 4, 4);
        } else {
            uint16_t h = (uint16_t)(raw[i * 2] | (raw[i * 2 + 1] << 8));
            dst[i] = t->dtype == DT_BF16 ? bf16_to_float(h) : f16_to_float(h);
        }
    }
    return 0;
}

#define PACK_CHUNK 2048u

typedef struct { int s; uint32_t n; uint64_t b0; uint64_t g0; } PackChunk;

typedef struct {
    PackCtx *pc;
    PackChunk *chunks;
    uint64_t nchunks;
    uint64_t next;            /* lock-free chunk counter */
    double *cerr, *cref;      /* per-chunk fidelity partials */
    uint64_t *cbad;           /* per-chunk first bad frame (block+1) */
    float *cbadv;
    time_t t0;
} PackJob;

static void pack_worker(int tid, void *vj)
{
    PackJob *j = (PackJob *)vj;
    PackCtx *pc = j->pc;
    uint32_t block = pc->block;
    (void)tid;
    float *buf = (float *)malloc((size_t)PACK_CHUNK * block * sizeof(float));
    uint8_t *raw = (uint8_t *)malloc((size_t)PACK_CHUNK * block * 4);
    float *tmp = (float *)malloc(block * sizeof(float));
    FILE **fh = (FILE **)calloc((size_t)g_n_shards, sizeof(FILE *));
    if (!buf || !raw || !tmp || !fh) { fprintf(stderr, "out of memory packing\n"); exit(1); }
    for (;;) {
        uint64_t c = atomic_next(&j->next);
        if (c >= j->nchunks) break;
        PackChunk *ch = &j->chunks[c];
        Shard *sh = g_scope[ch->s].shard;
        TensorInfo *t = g_scope[ch->s].tensor;
        int si = (int)(sh - g_shards);
        if (!fh[si]) {
            fh[si] = fopen(sh->path, "rb");
            if (!fh[si]) { fprintf(stderr, "cannot open %s\n", sh->path); exit(1); }
        }
        if (tensor_read_fh(fh[si], sh, t, ch->b0 * block,
                           (uint64_t)ch->n * block, buf, raw) != 0) {
            fprintf(stderr, "read failed on %s\n", t->name);
            exit(1);
        }
        double e2 = 0.0, r2 = 0.0;
        for (uint32_t b = 0; b < ch->n; b++) {
            const float *rw = buf + (size_t)b * block;
            float mn, sc;
            normalize_block(rw, block, tmp, &mn, &sc);
            int k = nearest_centroid(pc->cent, pc->entries, block, tmp);
            uint64_t at = ch->g0 + b;
            pc->idx[at] = (uint8_t)k;
            pc->mn_h[at] = float_to_half(mn);
            pc->sc_h[at] = float_to_half(sc);
            if (!j->cbad[c]
                    && (((pc->mn_h[at] & 0x7c00u) == 0x7c00u)
                        || ((pc->sc_h[at] & 0x7c00u) == 0x7c00u))) {
                j->cbad[c] = at + 1;
                j->cbadv[c] = ((pc->mn_h[at] & 0x7c00u) == 0x7c00u) ? mn : sc;
            }
            float rmn = f16_to_float(pc->mn_h[at]);
            float rsc = f16_to_float(pc->sc_h[at]);
            const float *cc = pc->cent + (size_t)k * block;
            for (uint32_t i = 0; i < block; i++) {
                double rec = (double)rmn + (double)rsc * cc[i];
                double d = (double)rw[i] - rec;
                e2 += d * d;
                r2 += (double)rw[i] * rw[i];
            }
        }
        j->cerr[c] = e2;
        j->cref[c] = r2;
        if (c != 0 && (c % 8192u) == 0) {
            double el = difftime(time(NULL), j->t0);
            double frac = (double)c / (double)j->nchunks;
            printf("  pass 2: %llu / %llu chunks (%.1f%%), elapsed %.0f s, eta %.0f s\n",
                   (unsigned long long)c, (unsigned long long)j->nchunks,
                   frac * 100.0, el, frac > 0 ? el * (1.0 - frac) / frac : 0.0);
            fflush(stdout);
        }
    }
    for (int i = 0; i < g_n_shards; i++) if (fh[i]) fclose(fh[i]);
    free(fh); free(tmp); free(raw); free(buf);
}

static void pack_parallel(PackCtx *pc)
{
    uint64_t nch = 0;
    for (int s = 0; s < g_n_scope; s++) {
        uint64_t tb = g_scope[s].tensor->count / pc->block;
        nch += (tb + PACK_CHUNK - 1) / PACK_CHUNK;
    }
    PackJob job;
    memset(&job, 0, sizeof(job));
    job.pc = pc;
    job.nchunks = nch;
    job.chunks = (PackChunk *)malloc((nch ? nch : 1) * sizeof(PackChunk));
    job.cerr = (double *)calloc(nch ? nch : 1, sizeof(double));
    job.cref = (double *)calloc(nch ? nch : 1, sizeof(double));
    job.cbad = (uint64_t *)calloc(nch ? nch : 1, sizeof(uint64_t));
    job.cbadv = (float *)calloc(nch ? nch : 1, sizeof(float));
    if (!job.chunks || !job.cerr || !job.cref || !job.cbad || !job.cbadv) {
        fprintf(stderr, "out of memory packing\n");
        exit(1);
    }
    uint64_t ci = 0, g0 = 0;
    for (int s = 0; s < g_n_scope; s++) {
        uint64_t tb = g_scope[s].tensor->count / pc->block;
        for (uint64_t b0 = 0; b0 < tb; b0 += PACK_CHUNK) {
            uint64_t n = tb - b0;
            if (n > PACK_CHUNK) n = PACK_CHUNK;
            job.chunks[ci].s = s;
            job.chunks[ci].n = (uint32_t)n;
            job.chunks[ci].b0 = b0;
            job.chunks[ci].g0 = g0 + b0;
            ci++;
        }
        g0 += tb;
    }
    job.t0 = time(NULL);
    run_parallel(g_threads, pack_worker, &job);
    /* Deterministic reduction in chunk order. */
    for (uint64_t c = 0; c < nch; c++) {
        pc->err2 += job.cerr[c];
        pc->ref2 += job.cref[c];
        if (!pc->frame_bad && job.cbad[c]) {
            pc->frame_bad = 1;
            pc->bad_block = job.cbad[c] - 1;
            pc->bad_value = job.cbadv[c];
        }
    }
    pc->cursor = g_total_blocks;
    free(job.chunks); free(job.cerr); free(job.cref);
    free(job.cbad); free(job.cbadv);
}
/* ---------------- residual stage (experimental) ----------------
 * A second book trained on the first stage's residuals, measured
 * only: the written stream stays single-stage, and the report
 * carries the two-stage fidelity and the rate it would cost
 * (a second 5-byte record and a second book per tensor).
 * Residuals are normalized minmax regardless of the stage-1 norm;
 * they are a different population from the weights. */

static int g_residual = 0;

static void minmax_normalize(const float *raw, uint32_t block,
                             float *out, float *mn_out, float *sc_out)
{
    float mn = raw[0], mx = raw[0];
    for (uint32_t i = 1; i < block; i++) {
        if (raw[i] < mn) mn = raw[i];
        if (raw[i] > mx) mx = raw[i];
    }
    float sc = mx - mn;
    if (sc == 0.0f) sc = 1.0f;
    for (uint32_t i = 0; i < block; i++) out[i] = (raw[i] - mn) / sc;
    *mn_out = mn;
    *sc_out = sc;
}

static void recon_stage1(const float *cent, uint32_t entries,
                         uint32_t block, const float *raw,
                         float *tmp, float *recon)
{
    float mn, sc;
    normalize_block(raw, block, tmp, &mn, &sc);
    int k = nearest_centroid(cent, entries, block, tmp);
    float rmn = f16_to_float(float_to_half(mn));
    float rsc = f16_to_float(float_to_half(sc));
    const float *c = cent + (size_t)k * block;
    for (uint32_t i = 0; i < block; i++)
        recon[i] = rmn + rsc * c[i];
}

typedef struct {
    uint32_t block, entries;
    const float *cent1;
    SampleCtx inner;
    float *tmp, *recon, *resid, *ntmp;
} ResSampleCtx;

static void res_sample_fn(const float *raw, void *vctx)
{
    ResSampleCtx *ctx = (ResSampleCtx *)vctx;
    recon_stage1(ctx->cent1, ctx->entries, ctx->block, raw,
                 ctx->tmp, ctx->recon);
    for (uint32_t i = 0; i < ctx->block; i++)
        ctx->resid[i] = raw[i] - ctx->recon[i];
    float mn, sc;
    minmax_normalize(ctx->resid, ctx->block, ctx->ntmp, &mn, &sc);
    sample_offer(&ctx->inner, ctx->ntmp);
}

typedef struct {
    uint32_t block, entries;
    const float *cent1, *cent2;
    double err2, ref2;
    float *tmp, *recon, *resid, *ntmp;
} ResFidCtx;

static void res_fid_fn(const float *raw, void *vctx)
{
    ResFidCtx *ctx = (ResFidCtx *)vctx;
    recon_stage1(ctx->cent1, ctx->entries, ctx->block, raw,
                 ctx->tmp, ctx->recon);
    for (uint32_t i = 0; i < ctx->block; i++)
        ctx->resid[i] = raw[i] - ctx->recon[i];
    float mn, sc;
    minmax_normalize(ctx->resid, ctx->block, ctx->ntmp, &mn, &sc);
    int k = nearest_centroid(ctx->cent2, ctx->entries, ctx->block,
                             ctx->ntmp);
    float rmn = f16_to_float(float_to_half(mn));
    float rsc = f16_to_float(float_to_half(sc));
    const float *c = ctx->cent2 + (size_t)k * ctx->block;
    for (uint32_t i = 0; i < ctx->block; i++) {
        double rec = (double)ctx->recon[i]
                   + (double)rmn + (double)rsc * c[i];
        double d = (double)raw[i] - rec;
        ctx->err2 += d * d;
        ctx->ref2 += (double)raw[i] * raw[i];
    }
}


/* ---------------- format v3: pack identity (H3b) ----------------
 * Format v3 answers Codex Astra H3b as JD decided it
 * (2026-11-25): generation pairing as the floor, plus a
 * per-tensor digest manifest.
 *
 * Generation pairing: both v3 headers carry a 64-bit
 * generation id. The id is not random: it is an FNV-1a digest
 * of the pair's own content (geometry, tensor names, book
 * bytes, index and frame bytes, raw-section bytes), computed
 * before the headers are written. Two files packed together
 * share it by construction; a stream and a book from
 * different pack runs disagree, and every v3 reader refuses
 * the pair with a plain message. Because the id is
 * content-derived, re-packing the same model with the same
 * options reproduces the same id: packs stay reproducible.
 *
 * Manifest: <prefix>.nq2manifest, a text sidecar listing,
 * per tensor, its block range and two FNV-1a digests: one
 * over the tensor's stored stream bytes (indices, then
 * frame minima, then frame scales; the raw bytes for an
 * fp16-raw tensor) and one over its book's stored floats.
 * Readers verify both at open when the manifest is present.
 *
 * Legacy packs (v1/v2) carry no identity; readers bind them
 * positionally, as before, and say so in a notice.
 *
 * v3-only writer options: --pq D (product quantization),
 * --budgets (per-tensor block sizes by sensitivity), and
 * --embed-fp16 (the embedding stored as raw float16). All
 * three need per-tensor books. */

static int g_format = 3;      /* --format 1|2|3 */
static int g_pq_d = 0;        /* --pq D: subvector dim, 0 = flat */
static int g_budgets = 0;     /* --budgets */
static int g_embed_fp16 = 0;  /* --embed-fp16 */
/* Generalised raw float16 storage (SuperVL work, 2026-11-27).
 *   --raw-names a,b   tensors whose name contains any listed substring
 *   --raw-small N     tensors with N weights or fewer, or with a weight
 *                     count that is not a whole number of blocks (a
 *                     block-packed tensor would silently lose its tail)
 * Both use the same raw float16 section as --embed-fp16 and need
 * per-tensor books, format 3. */
static const char *g_raw_names = NULL;
static uint64_t g_raw_small = 0;
static int g_raw_any = 0;     /* any raw option given */

static int name_matches_any(const char *name, const char *list)
{
    const char *p = list;
    while (p && *p) {
        const char *e = strchr(p, ',');
        size_t n = e ? (size_t)(e - p) : strlen(p);
        if (n > 0 && n < 255) {
            char tok[256];
            memcpy(tok, p, n); tok[n] = 0;
            if (strstr(name, tok)) return 1;
        }
        if (!e) break;
        p = e + 1;
    }
    return 0;
}

static int tensor_is_raw(const TensorInfo *ti, uint32_t block)
{
    if (g_embed_fp16 && strstr(ti->name, "embed_tokens")) return 1;
    if (g_raw_names && name_matches_any(ti->name, g_raw_names)) return 1;
    if (g_raw_small && (ti->count <= g_raw_small
                        || ti->count % block != 0)) return 1;
    return 0;
}

#define NQ2_FNV_SEED 14695981039346656037ull
static uint64_t fnv1a(const void *data, size_t len, uint64_t h)
{
    const uint8_t *p = (const uint8_t *)data;
    for (size_t i = 0; i < len; i++) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ull;
    }
    return h;
}
static uint64_t fnv1a_u32(uint32_t v, uint64_t h)
{
    uint8_t b[4];
    b[0] = (uint8_t)v; b[1] = (uint8_t)(v >> 8);
    b[2] = (uint8_t)(v >> 16); b[3] = (uint8_t)(v >> 24);
    return fnv1a(b, 4, h);
}

/* One tensor's packed output, kept whole so the budget pass
 * can re-pack a tensor at a different block size and so the
 * v3 writer digests exactly the bytes it writes. */
typedef struct {
    float *book;          /* entries * block floats (PQ: same count) */
    uint8_t *idx;         /* nblocks * idxw bytes */
    uint16_t *mn_h, *sc_h;
    uint64_t nblocks;
    uint32_t block;
    uint16_t *raw;        /* fp16-raw storage, or NULL */
    uint64_t raw_count;
    double err2, ref2;
    int frame_bad;
    float bad_value;
    uint64_t bad_block;
} TOut;

static void tout_free(TOut *o)
{
    free(o->book); free(o->idx); free(o->mn_h); free(o->sc_h);
    free(o->raw);
    memset(o, 0, sizeof(*o));
}

/* PQ packing: each normalized block is split into nsub
 * subvectors of dim d; each subvector is assigned to its own
 * sub-book (the nq2-pq probe's scheme, K entries per sub-book,
 * frame unchanged). Reconstruction the runner will see:
 * mn + sc * subbook[sub][idx[sub]][i], measured here the same
 * way pack_fn measures the flat path. */
typedef struct {
    uint32_t block, entries, d, nsub;
    const float *books;   /* nsub * entries * d */
    uint8_t *idx;         /* nblocks * nsub */
    uint16_t *mn_h, *sc_h;
    uint64_t cursor;
    double err2, ref2;
    float *tmp;
    int frame_bad;
    float bad_value;
    uint64_t bad_block;
} PqPackCtx;

static void pq_pack_fn(const float *raw, void *vctx)
{
    PqPackCtx *ctx = (PqPackCtx *)vctx;
    float mn, sc;
    normalize_block(raw, ctx->block, ctx->tmp, &mn, &sc);
    uint64_t at = ctx->cursor++;
    ctx->mn_h[at] = float_to_half(mn);
    ctx->sc_h[at] = float_to_half(sc);
    if (!ctx->frame_bad
            && (((ctx->mn_h[at] & 0x7c00u) == 0x7c00u)
                || ((ctx->sc_h[at] & 0x7c00u) == 0x7c00u))) {
        ctx->frame_bad = 1;
        ctx->bad_block = at;
        ctx->bad_value = ((ctx->mn_h[at] & 0x7c00u) == 0x7c00u)
                         ? mn : sc;
    }
    float rmn = f16_to_float(ctx->mn_h[at]);
    float rsc = f16_to_float(ctx->sc_h[at]);
    for (uint32_t sub = 0; sub < ctx->nsub; sub++) {
        const float *bk = ctx->books
            + (size_t)sub * ctx->entries * ctx->d;
        int k = nearest_centroid(bk, ctx->entries, ctx->d,
                                 ctx->tmp + sub * ctx->d);
        ctx->idx[at * ctx->nsub + sub] = (uint8_t)k;
        const float *c = bk + (size_t)k * ctx->d;
        for (uint32_t i = 0; i < ctx->d; i++) {
            double rec = (double)rmn + (double)rsc * c[i];
            double dd = (double)raw[sub * ctx->d + i] - rec;
            ctx->err2 += dd * dd;
            ctx->ref2 += (double)raw[sub * ctx->d + i]
                       * (double)raw[sub * ctx->d + i];
        }
    }
}

/* Sample, learn, and pack one scope tensor at the given
 * block size into o (flat or PQ according to g_pq_d).
 * Returns 0 on success; on a non-finite frame the output is
 * still produced and o->frame_bad is set, as pack_fn does. */
static int pack_one_tensor(int s, uint32_t block, uint32_t entries,
                           int iters, uint64_t sample_cap, TOut *o)
{
    TensorInfo *ti = g_scope[s].tensor;
    uint64_t tb = ti->count / block;
    memset(o, 0, sizeof(*o));
    o->block = block;
    o->nblocks = tb;
    o->book = (float *)calloc((size_t)entries * block, sizeof(float));
    if (!o->book) { fprintf(stderr, "out of memory\n"); return 1; }
    if (tb == 0) return 0;
    uint64_t cap = sample_cap;
    if (cap > tb) cap = tb;
    SampleCtx sctx;
    memset(&sctx, 0, sizeof(sctx));
    sctx.block = block;
    sctx.sample_cap = cap;
    sctx.sample = (float *)malloc(cap * block * sizeof(float));
    sctx.tmp = (float *)malloc(block * sizeof(float));
    if (!sctx.sample || !sctx.tmp) {
        fprintf(stderr, "out of memory sampling\n");
        return 1;
    }
    walk_one_tensor(s, block, sample_fn, &sctx);
    if (g_pq_d) {
        /* One k-means per subvector position, over the
         * sampled blocks' subvectors. */
        uint32_t d = (uint32_t)g_pq_d, nsub = block / d;
        float *sub_sample = (float *)malloc(
            (size_t)sctx.sample_n * d * sizeof(float));
        if (!sub_sample) { fprintf(stderr, "out of memory\n"); return 1; }
        int verbose_save = g_kmeans_verbose;   /* parallel workers all save 0 */
        g_kmeans_verbose = 0;
        for (uint32_t sub = 0; sub < nsub; sub++) {
            for (uint64_t b = 0; b < sctx.sample_n; b++)
                memcpy(sub_sample + b * d,
                       sctx.sample + b * block + sub * d,
                       d * sizeof(float));
            learn_book(sub_sample, sctx.sample_n, d, entries, iters,
                       o->book + (size_t)sub * entries * d);
        }
        g_kmeans_verbose = verbose_save;
        free(sub_sample);
        PqPackCtx pctx;
        memset(&pctx, 0, sizeof(pctx));
        pctx.block = block; pctx.entries = entries;
        pctx.d = d; pctx.nsub = nsub;
        pctx.books = o->book;
        pctx.idx = (uint8_t *)malloc(tb * nsub);
        pctx.mn_h = (uint16_t *)malloc(tb * 2);
        pctx.sc_h = (uint16_t *)malloc(tb * 2);
        pctx.tmp = (float *)malloc(block * sizeof(float));
        if (!pctx.idx || !pctx.mn_h || !pctx.sc_h || !pctx.tmp) {
            fprintf(stderr, "out of memory packing\n");
            return 1;
        }
        walk_one_tensor(s, block, pq_pack_fn, &pctx);
        o->idx = pctx.idx; o->mn_h = pctx.mn_h; o->sc_h = pctx.sc_h;
        o->err2 = pctx.err2; o->ref2 = pctx.ref2;
        o->frame_bad = pctx.frame_bad;
        o->bad_value = pctx.bad_value;
        o->bad_block = pctx.bad_block;
        free(pctx.tmp);
    } else {
        learn_book(sctx.sample, sctx.sample_n, block, entries,
                   iters, o->book);
        PackCtx pctx;
        memset(&pctx, 0, sizeof(pctx));
        pctx.block = block;
        pctx.entries = entries;
        pctx.cent = o->book;
        pctx.idx = (uint8_t *)malloc(tb);
        pctx.mn_h = (uint16_t *)malloc(tb * 2);
        pctx.sc_h = (uint16_t *)malloc(tb * 2);
        pctx.tmp = (float *)malloc(block * sizeof(float));
        if (!pctx.idx || !pctx.mn_h || !pctx.sc_h || !pctx.tmp) {
            fprintf(stderr, "out of memory packing\n");
            return 1;
        }
        walk_one_tensor(s, block, pack_fn, &pctx);
        o->idx = pctx.idx; o->mn_h = pctx.mn_h; o->sc_h = pctx.sc_h;
        o->err2 = pctx.err2; o->ref2 = pctx.ref2;
        o->frame_bad = pctx.frame_bad;
        o->bad_value = pctx.bad_value;
        o->bad_block = pctx.bad_block;
        free(pctx.tmp);
    }
    free(sctx.sample); free(sctx.tmp);
    return 0;
}

/* Store one scope tensor as raw float16 (--embed-fp16): no
 * book, no blocks; fidelity is the fp16 round-trip itself. */
static int store_one_tensor_fp16(int s, TOut *o)
{
    TensorInfo *ti = g_scope[s].tensor;
    Shard *sh = g_scope[s].shard;
    memset(o, 0, sizeof(*o));
    o->raw_count = ti->count;
    o->raw = (uint16_t *)malloc(ti->count * sizeof(uint16_t));
    if (!o->raw) { fprintf(stderr, "out of memory\n"); return 1; }
    const uint64_t SLAB = 1ull << 20;
    float *buf = (float *)malloc(SLAB * sizeof(float));
    if (!buf) { fprintf(stderr, "out of memory\n"); return 1; }
    for (uint64_t base = 0; base < ti->count; base += SLAB) {
        uint64_t want = ti->count - base;
        if (want > SLAB) want = SLAB;
        if (tensor_read(sh, ti, base, want, buf) != 0) {
            fprintf(stderr, "read failed on %s\n", ti->name);
            free(buf);
            return 1;
        }
        for (uint64_t i = 0; i < want; i++) {
            uint16_t h = float_to_half(buf[i]);
            o->raw[base + i] = h;
            double d = (double)buf[i] - (double)f16_to_float(h);
            o->err2 += d * d;
            o->ref2 += (double)buf[i] * (double)buf[i];
        }
    }
    free(buf);
    return 0;
}

/* The generation id over a set of per-tensor outputs:
 * geometry first, then per tensor its name, book bytes,
 * index bytes, frame bytes, and raw bytes, in scope order. */
static uint64_t generation_of(uint32_t block, uint32_t entries,
                              uint32_t flags, const TOut *outs)
{
    uint64_t h = NQ2_FNV_SEED;
    h = fnv1a_u32(block, h);
    h = fnv1a_u32(entries, h);
    h = fnv1a_u32((uint32_t)g_n_scope, h);
    h = fnv1a_u32(flags, h);
    h = fnv1a_u32((uint32_t)g_pq_d, h);
    uint32_t idxw = g_pq_d ? block / (uint32_t)g_pq_d : 1;
    for (int s = 0; s < g_n_scope; s++) {
        const char *nm = g_scope[s].tensor->name;
        h = fnv1a(nm, strlen(nm), h);
        if (outs[s].book)
            h = fnv1a(outs[s].book,
                      (size_t)entries * outs[s].block * sizeof(float), h);
        if (outs[s].nblocks)
            h = fnv1a(outs[s].idx,
                      (size_t)outs[s].nblocks * idxw, h);
        if (outs[s].nblocks) {
            h = fnv1a(outs[s].mn_h, (size_t)outs[s].nblocks * 2, h);
            h = fnv1a(outs[s].sc_h, (size_t)outs[s].nblocks * 2, h);
        }
        if (outs[s].raw)
            h = fnv1a(outs[s].raw, (size_t)outs[s].raw_count * 2, h);
    }
    return h;
}

static uint64_t seg_digest_of(const TOut *o, uint32_t idxw)
{
    uint64_t h = NQ2_FNV_SEED;
    if (o->raw) return fnv1a(o->raw, (size_t)o->raw_count * 2, h);
    h = fnv1a(o->idx, (size_t)o->nblocks * idxw, h);
    h = fnv1a(o->mn_h, (size_t)o->nblocks * 2, h);
    h = fnv1a(o->sc_h, (size_t)o->nblocks * 2, h);
    return h;
}

static uint64_t book_digest_of(const float *book, uint32_t entries,
                               uint32_t block)
{
    if (!book) return NQ2_FNV_SEED;
    return fnv1a(book, (size_t)entries * block * sizeof(float),
                 NQ2_FNV_SEED);
}

/* Write the manifest beside a published pair. Returns 0 on
 * success. block0s are cumulative block offsets in scope
 * order. */
static int write_manifest(const char *out_prefix, uint64_t gen,
                          const TOut *outs, const uint64_t *block0s,
                          uint32_t idxw, uint32_t entries)
{
    char pathbuf[1088];
    snprintf(pathbuf, sizeof(pathbuf), "%s.nq2manifest", out_prefix);
    FILE *mf = fopen(pathbuf, "w");
    if (!mf) { fprintf(stderr, "cannot write %s\n", pathbuf); return 1; }
    fprintf(mf, "NQ2MANIFEST 1\n");
    fprintf(mf, "generation %016llx\n", (unsigned long long)gen);
    fprintf(mf, "tensors %d\n", g_n_scope);
    for (int s = 0; s < g_n_scope; s++) {
        fprintf(mf, "tensor %s %llu %llu %u %016llx %016llx %s\n",
                g_scope[s].tensor->name,
                (unsigned long long)block0s[s],
                (unsigned long long)outs[s].nblocks,
                outs[s].block,
                (unsigned long long)seg_digest_of(&outs[s], idxw),
                (unsigned long long)book_digest_of(outs[s].book,
                                                   entries,
                                                   outs[s].block),
                outs[s].raw ? "raw" : "packed");
    }
    fprintf(mf, "end\n");
    if (fclose(mf) != 0) {
        fprintf(stderr, "write failed for %s\n", pathbuf);
        return 1;
    }
    printf("manifest: %s\n", pathbuf);
    return 0;
}

/* ---------------- main ---------------- */

/* Checked write for the artifact writers (Codex Astra review
 * H6, 2026-11-25): a short fwrite is a failure the caller must
 * propagate, never a detail to ignore. The writers stage each
 * artifact under a temporary name, verify every write and the
 * final close, and rename the pair into place only when both
 * are complete, so a rejected write publishes neither a half
 * pair nor a success report, and a failed re-pack leaves the
 * previous pair untouched. */
static int wr(FILE *f, const void *p, size_t sz, size_t n)
{
    return fwrite(p, sz, n, f) == n ? 0 : -1;
}

/* Publish a fully written staging file over its destination.
 * POSIX rename() replaces an existing destination atomically.
 * Windows rename() (MoveFileA without MOVEFILE_REPLACE_EXISTING)
 * fails with EEXIST when the destination already exists, so a
 * re-pack over an existing prefix could never publish (Archie
 * Round 3 P1-WIN, 2026-11-25). On Windows the destination is
 * removed first; that leaves a small non-atomic window where
 * the destination is briefly absent, the accepted platform
 * trade-off. The H6 guarantee is unchanged in the part that
 * matters: staging writes are fully checked before this is
 * called, so a failed write still publishes nothing. */
static int publish_file(const char *tmp, const char *dst)
{
#ifdef _WIN32
    remove(dst); /* Windows rename() fails with EEXIST if dst exists */
#endif
    return rename(tmp, dst);
}

/* Per-writer identity for staging and backup names (Sol H4):
 * two quantizer runs aimed at the same prefix must not share
 * staging files, or each can publish the other's half pair. */
static long writer_pid(void)
{
#ifdef _WIN32
    return (long)_getpid();
#else
    return (long)getpid();
#endif
}

static int path_exists(const char *p)
{
    FILE *f = fopen(p, "rb");
    if (f) { fclose(f); return 1; }
    return 0;
}

/* Publish a staged pair (book, stream) as one commit (Sol
 * H4). Publishing the book and then the stream with no commit
 * point can leave a new book beside an old stream, and the
 * readers validate the pair on geometry only, so a mixed
 * generation of the same geometry reads silently. Within the
 * current format the commit is a rename sequence: the old
 * pair is first moved aside to per-writer backup names, then
 * both staging files are renamed in; if the second rename
 * fails, the new book is unpublished and the backups are
 * restored, so the prefix names either the whole old pair or
 * the whole new pair, never a mixture. If a rollback rename
 * itself fails, the backups are left in place and named in
 * the error. True generation pairing inside the files needs
 * a format change and is flagged for JD with the H3b design
 * options, not invented here. */
static int publish_pair(const char *book_tmp, const char *stream_tmp,
                        const char *book_dst, const char *stream_dst)
{
    char book_bak[1088], stream_bak[1088];
    long pid = writer_pid();
    snprintf(book_bak, sizeof(book_bak), "%s.bak-%ld",
             book_dst, pid);
    snprintf(stream_bak, sizeof(stream_bak), "%s.bak-%ld",
             stream_dst, pid);
    int have_book_bak = 0, have_stream_bak = 0;
    remove(book_bak);
    remove(stream_bak);
    if (rename(book_dst, book_bak) == 0) have_book_bak = 1;
    else if (path_exists(book_dst)) {
        fprintf(stderr, "cannot move %s aside for the pair "
                "publish; nothing published\n", book_dst);
        return -1;
    }
    if (rename(stream_dst, stream_bak) == 0) have_stream_bak = 1;
    else if (path_exists(stream_dst)) {
        fprintf(stderr, "cannot move %s aside for the pair "
                "publish; nothing published\n", stream_dst);
        if (have_book_bak) rename(book_bak, book_dst);
        return -1;
    }
    if (publish_file(book_tmp, book_dst) != 0) {
        if (have_book_bak && rename(book_bak, book_dst) != 0)
            fprintf(stderr, "rollback failed: previous book "
                    "left at %s\n", book_bak);
        if (have_stream_bak && rename(stream_bak, stream_dst) != 0)
            fprintf(stderr, "rollback failed: previous stream "
                    "left at %s\n", stream_bak);
        return -1;
    }
    if (publish_file(stream_tmp, stream_dst) != 0) {
        remove(book_dst); /* unpublish the new book */
        if (have_book_bak && rename(book_bak, book_dst) != 0)
            fprintf(stderr, "rollback failed: previous book "
                    "left at %s\n", book_bak);
        if (have_stream_bak && rename(stream_bak, stream_dst) != 0)
            fprintf(stderr, "rollback failed: previous stream "
                    "left at %s\n", stream_bak);
        return -1;
    }
    if (have_book_bak) remove(book_bak);
    if (have_stream_bak) remove(stream_bak);
    return 0;
}

/* Canonical shard order: by filename (the part after the last
 * '/' or '\\'), ties broken by the full path. The readers (all
 * four inference engines, the runner, the error probe) bind
 * the packed stream to directory tensors in this order, so the
 * writer must serialize in it too, whatever order the shards
 * were passed in (Codex Astra review H1, 2026-11-25). Both
 * separators count: a Windows invocation spelled with native
 * backslashes must sort by the same basenames as one spelled
 * with slashes (Archie Round 3 P2-1, 2026-11-25). */
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

/* ---------------- parallel per-tensor books (2026-11-26) ----------------
 * Co-implemented by Cortana at JD's direction; the quantizer is Elfie's.
 * Used for --books per-tensor --format 3 without --pq/--budgets/--residual
 * when --threads > 1. Output is byte-identical to the serial path:
 *   - each tensor's RNG start state is precomputed from geometry (the
 *     reservoir draws tb - cap values, learn_book draws `entries`), so
 *     every worker reproduces the exact serial draws for its tensor;
 *   - every byte's file offset is known before packing, so workers write
 *     their tensor straight into disjoint regions of the staged files;
 *   - the generation id is the same in-order FNV-1a, computed by reading
 *     the staged files back in scope order after the join, then patched
 *     into both headers;
 *   - per-tensor error sums are reduced in scope order.
 * No mutexes: tensors are claimed through one lock-free counter. At most
 * one tensor per worker is in memory, so RAM stays bounded (the serial
 * writer holds every book at once: about 11 GB for a 43k-tensor model). */

static uint64_t xs_step(uint64_t x)
{
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    return x;
}

typedef struct {
    uint32_t block, entries;
    int iters;
    uint64_t cap;
    const char *book_tmp, *stream_tmp;
    int *is_raw;
    uint64_t *nb, *rng0, *book_off, *idx_off, *mn_off, *sc_off, *raw_off;
    double *err2, *ref2;
    uint64_t *segd, *bookd;
    int *frame_bad;  uint64_t *bad_block; float *bad_value;
    int *fail;
    int *tbad;            /* per worker */
    uint32_t idxw;        /* index bytes per block: 1 flat, block/d for PQ */
    uint64_t next;        /* lock-free tensor counter */
    uint64_t done;        /* progress only (atomic) */
    time_t t0;
} PtJob;

static int pt_write(FILE *f, uint64_t off, const void *p, size_t sz, size_t n)
{
    if (!n) return 0;
    if (seek_to(f, off) != 0) return -1;
    return fwrite(p, sz, n, f) == n ? 0 : -1;
}

static void pt_worker(int tid, void *vj)
{
    PtJob *j = (PtJob *)vj;
    t_fh = (FILE **)calloc((size_t)g_n_shards, sizeof(FILE *));
    FILE *bf = fopen(j->book_tmp, "r+b");
    FILE *sf = fopen(j->stream_tmp, "r+b");
    if (!t_fh || !bf || !sf) { j->tbad[tid] = 1; goto out; }
    for (;;) {
        uint64_t s64 = atomic_next(&j->next);
        if (s64 >= (uint64_t)g_n_scope) break;
        int s = (int)s64;
        TOut o;
        rng_state = j->rng0[s];
        if (j->is_raw[s]) {
            if (store_one_tensor_fp16(s, &o) != 0) { j->fail[s] = 1; continue; }
            if (pt_write(sf, j->raw_off[s], o.raw, 2, (size_t)o.raw_count) != 0)
                j->tbad[tid] = 1;
            j->segd[s] = seg_digest_of(&o, 1);
            j->bookd[s] = book_digest_of(NULL, j->entries, j->block);
        } else {
            if (pack_one_tensor(s, j->block, j->entries, j->iters, j->cap, &o) != 0) {
                j->fail[s] = 1; continue;
            }
            if (pt_write(bf, j->book_off[s], o.book, sizeof(float),
                         (size_t)j->entries * j->block) != 0) j->tbad[tid] = 1;
            if (o.nblocks) {
                if (pt_write(sf, j->idx_off[s], o.idx, 1, (size_t)o.nblocks * j->idxw) != 0
                    || pt_write(sf, j->mn_off[s], o.mn_h, 2, (size_t)o.nblocks) != 0
                    || pt_write(sf, j->sc_off[s], o.sc_h, 2, (size_t)o.nblocks) != 0)
                    j->tbad[tid] = 1;
            }
            j->segd[s] = seg_digest_of(&o, j->idxw);
            j->bookd[s] = book_digest_of(o.book, j->entries, j->block);
            j->frame_bad[s] = o.frame_bad;
            j->bad_block[s] = o.bad_block;
            j->bad_value[s] = o.bad_value;
        }
        j->err2[s] = o.err2;
        j->ref2[s] = o.ref2;
        tout_free(&o);
        uint64_t d = atomic_next(&j->done) + 1;
        if ((d % 500u) == 0 || d == (uint64_t)g_n_scope) {
            double el = difftime(time(NULL), j->t0);
            double fr = (double)d / (double)g_n_scope;
            printf("  tensors %llu / %d (%.1f%%), elapsed %.0f s, eta %.0f s\n",
                   (unsigned long long)d, g_n_scope, fr * 100.0, el,
                   fr > 0 ? el * (1.0 - fr) / fr : 0.0);
            fflush(stdout);
        }
    }
out:
    if (bf && fflush(bf) != 0) j->tbad[tid] = 1;
    if (sf && fflush(sf) != 0) j->tbad[tid] = 1;
    if (bf && fclose(bf) != 0) j->tbad[tid] = 1;
    if (sf && fclose(sf) != 0) j->tbad[tid] = 1;
    if (t_fh) {
        for (int i = 0; i < g_n_shards; i++) if (t_fh[i]) fclose(t_fh[i]);
        free(t_fh);
        t_fh = NULL;
    }
}

static int fnv_range(FILE *f, uint64_t off, uint64_t len, uint64_t *h,
                     uint8_t *buf, size_t bufsz)
{
    if (!len) return 0;
    if (seek_to(f, off) != 0) return -1;
    while (len) {
        size_t want = len > bufsz ? bufsz : (size_t)len;
        if (fread(buf, 1, want, f) != want) return -1;
        *h = fnv1a(buf, want, *h);
        len -= want;
    }
    return 0;
}

static int per_tensor_parallel(const char *out_prefix, uint32_t block,
                               uint32_t entries, int iters,
                               uint64_t tensor_sample_cap)
{
    int n = g_n_scope;
    PtJob j;
    memset(&j, 0, sizeof(j));
    j.block = block; j.entries = entries; j.iters = iters; j.cap = tensor_sample_cap;
#define PT_ALLOC(p, T) do { p = (T *)calloc((size_t)(n ? n : 1), sizeof(T)); if (!p) { fprintf(stderr, "out of memory\n"); return 1; } } while (0)
    PT_ALLOC(j.is_raw, int);   PT_ALLOC(j.nb, uint64_t);   PT_ALLOC(j.rng0, uint64_t);
    PT_ALLOC(j.book_off, uint64_t); PT_ALLOC(j.idx_off, uint64_t);
    PT_ALLOC(j.mn_off, uint64_t);   PT_ALLOC(j.sc_off, uint64_t); PT_ALLOC(j.raw_off, uint64_t);
    PT_ALLOC(j.err2, double); PT_ALLOC(j.ref2, double);
    PT_ALLOC(j.segd, uint64_t); PT_ALLOC(j.bookd, uint64_t);
    PT_ALLOC(j.frame_bad, int); PT_ALLOC(j.bad_block, uint64_t); PT_ALLOC(j.bad_value, float);
    PT_ALLOC(j.fail, int);
    uint64_t *block0s; PT_ALLOC(block0s, uint64_t);
#undef PT_ALLOC
    j.tbad = (int *)calloc((size_t)g_threads + 1, sizeof(int));
    if (!j.tbad) { fprintf(stderr, "out of memory\n"); return 1; }

    /* Geometry, totals, and each tensor's RNG start state. */
    uint64_t tot_blocks = 0, tot_weights = 0, tot_raw = 0, raw_acc = 0;
    int n_books = 0, n_raw = 0;
    uint64_t st = rng_state;
    for (int s = 0; s < n; s++) {
        TensorInfo *ti = g_scope[s].tensor;
        j.is_raw[s] = tensor_is_raw(ti, block);
        j.rng0[s] = st;
        if (j.is_raw[s]) {
            j.nb[s] = 0;
            tot_raw += ti->count;
            n_raw++;
        } else {
            uint64_t tb = ti->count / block;
            j.nb[s] = tb;
            n_books++;
            uint64_t cap = tensor_sample_cap < tb ? tensor_sample_cap : tb;
            uint64_t nsub_rng = g_pq_d ? block / (uint32_t)g_pq_d : 1;
            uint64_t adv = (tb > cap ? tb - cap : 0)
                         + (tb ? nsub_rng * entries : 0);
            for (uint64_t k = 0; k < adv; k++) st = xs_step(st);
        }
        block0s[s] = tot_blocks;
        tot_blocks += j.nb[s];
        tot_weights += j.nb[s] * block;
    }
    rng_state = st;
    uint32_t flags = (tot_raw ? 4u : 0u) | (g_pq_d ? 1u : 0u);
    uint64_t idxw = g_pq_d ? block / (uint32_t)g_pq_d : 1;
    j.idxw = (uint32_t)idxw;
    uint64_t bookbytes = (uint64_t)entries * block * 4;
    uint64_t book_hdr = 28 + (g_pq_d ? 4 : 0);
    uint64_t book_size = book_hdr + (uint64_t)n_books * bookbytes;
    uint64_t H = 36 + (uint64_t)n * 8 + 8 + 4 + (g_pq_d ? 4 : 0) + ((flags & 4u) ? 4 + (uint64_t)n_raw * 4 : 0);
    uint64_t stream_size = H + tot_blocks * (idxw + 4) + tot_raw * 2;
    {
        uint64_t bk = 0;
        for (int s = 0; s < n; s++) {
            if (!j.is_raw[s]) { j.book_off[s] = book_hdr + bk * bookbytes; bk++; }
            j.idx_off[s] = H + block0s[s] * idxw;
            j.mn_off[s] = H + tot_blocks * idxw + block0s[s] * 2;
            j.sc_off[s] = H + tot_blocks * (idxw + 2) + block0s[s] * 2;
            if (j.is_raw[s]) {
                j.raw_off[s] = H + tot_blocks * (idxw + 4) + raw_acc * 2;
                raw_acc += g_scope[s].tensor->count;
            }
        }
    }

    /* Stage both files: headers with a zero generation id, sized in full. */
    char book_tmp[1088], stream_tmp[1088];
    snprintf(book_tmp, sizeof(book_tmp), "%s.nq2book.%ld.tmp", out_prefix, writer_pid());
    snprintf(stream_tmp, sizeof(stream_tmp), "%s.nq2.%ld.tmp", out_prefix, writer_pid());
    int bad = 0;
    uint64_t gen = 0;
    uint32_t version = 3, nb32 = (uint32_t)n_books, nseg = (uint32_t)n;
    {
        FILE *bf = fopen(book_tmp, "wb");
        if (!bf) { fprintf(stderr, "cannot write %s\n", book_tmp); return 1; }
        bad |= wr(bf, "NQ2B", 1, 4);
        bad |= wr(bf, &version, 4, 1);
        bad |= wr(bf, &block, 4, 1);
        bad |= wr(bf, &entries, 4, 1);
        bad |= wr(bf, &nb32, 4, 1);
        bad |= wr(bf, &gen, 8, 1);
        if (g_pq_d) { uint32_t d32 = (uint32_t)g_pq_d; bad |= wr(bf, &d32, 4, 1); }
        if (book_size > book_hdr) { uint8_t z = 0; bad |= seek_to(bf, book_size - 1); bad |= wr(bf, &z, 1, 1); }
        if (fclose(bf) != 0) bad = 1;
        FILE *sf = fopen(stream_tmp, "wb");
        if (!sf) { fprintf(stderr, "cannot write %s\n", stream_tmp); remove(book_tmp); return 1; }
        bad |= wr(sf, "NQ2\0", 1, 4);
        bad |= wr(sf, &version, 4, 1);
        bad |= wr(sf, &block, 4, 1);
        bad |= wr(sf, &entries, 4, 1);
        bad |= wr(sf, &tot_weights, 8, 1);
        bad |= wr(sf, &tot_blocks, 8, 1);
        bad |= wr(sf, &nseg, 4, 1);
        for (int s = 0; s < n; s++) bad |= wr(sf, &j.nb[s], 8, 1);
        bad |= wr(sf, &gen, 8, 1);
        bad |= wr(sf, &flags, 4, 1);
        if (flags & 1u) { uint32_t d32 = (uint32_t)g_pq_d; bad |= wr(sf, &d32, 4, 1); }
        if (flags & 4u) {
            uint32_t nr = (uint32_t)n_raw;
            bad |= wr(sf, &nr, 4, 1);
            for (int s = 0; s < n; s++)
                if (j.is_raw[s]) { uint32_t ix = (uint32_t)s; bad |= wr(sf, &ix, 4, 1); }
        }
        if (stream_size > H) { uint8_t z = 0; bad |= seek_to(sf, stream_size - 1); bad |= wr(sf, &z, 1, 1); }
        if (fclose(sf) != 0) bad = 1;
    }
    if (bad) { fprintf(stderr, "staging failed; nothing published\n"); remove(book_tmp); remove(stream_tmp); return 1; }

    /* Pack every tensor in parallel. */
    j.book_tmp = book_tmp; j.stream_tmp = stream_tmp;
    g_learn_threads = 1;      /* no nested k-means threads inside workers */
    g_kmeans_verbose = 0;
    j.t0 = time(NULL);
    printf("parallel per-tensor books: %d tensors on %d threads\n", n, g_threads);
    fflush(stdout);
    run_parallel(g_threads, pt_worker, &j);
    printf("per-tensor pass (%d threads): %lld s\n", g_threads, (long long)(time(NULL) - j.t0));
    fflush(stdout);

    for (int t = 0; t <= g_threads; t++) if (j.tbad[t]) bad = 1;
    for (int s = 0; s < n; s++) {
        TensorInfo *ti = g_scope[s].tensor;
        if (j.fail[s]) { fprintf(stderr, "tensor %s failed; nothing published\n", ti->name); bad = 1; break; }
        if (j.frame_bad[s]) {
            fprintf(stderr, "tensor %s, block %llu: frame value %g is not "
                    "representable in float16; refusing to publish a pack "
                    "with non-finite frames (Sol H8)\n", ti->name,
                    (unsigned long long)j.bad_block[s], (double)j.bad_value[s]);
            bad = 1; break;
        }
    }
    if (bad) { remove(book_tmp); remove(stream_tmp); return 1; }
    for (int s = 0; s < n; s++) {
        TensorInfo *ti = g_scope[s].tensor;
        printf("tensor %d/%d %s: %llu blocks\n", s + 1, n, ti->name,
               (unsigned long long)(ti->count / block));
        if (j.is_raw[s])
            printf("  stored as raw float16 (no book, no blocks); round-trip relative L2 %.4f\n",
                   j.ref2[s] > 0 ? dsqrt(j.err2[s] / j.ref2[s]) : 0.0);
        else
            printf("  tensor relative L2: %.4f\n",
                   j.ref2[s] > 0 ? dsqrt(j.err2[s] / j.ref2[s]) : 0.0);
    }
    double tot_err2 = 0.0, tot_ref2 = 0.0;
    for (int s = 0; s < n; s++) { tot_err2 += j.err2[s]; tot_ref2 += j.ref2[s]; }
    double rel_l2 = tot_ref2 > 0 ? dsqrt(tot_err2 / tot_ref2) : 0.0;
    printf("packed %llu blocks in %d per-tensor books; relative L2 error %.4f\n",
           (unsigned long long)tot_blocks, n_books, rel_l2);

    /* Generation id: the serial digest, in scope order, from the staged bytes. */
    {
        size_t bufsz = 1u << 22;
        uint8_t *buf = (uint8_t *)malloc(bufsz);
        FILE *bf = fopen(book_tmp, "rb"), *sf = fopen(stream_tmp, "rb");
        if (!buf || !bf || !sf) { fprintf(stderr, "cannot re-read staged files\n"); remove(book_tmp); remove(stream_tmp); return 1; }
        uint64_t h = NQ2_FNV_SEED;
        h = fnv1a_u32(block, h);
        h = fnv1a_u32(entries, h);
        h = fnv1a_u32((uint32_t)n, h);
        h = fnv1a_u32(flags, h);
        h = fnv1a_u32((uint32_t)g_pq_d, h);
        int rb = 0;
        time_t tg = time(NULL);
        for (int s = 0; s < n && !rb; s++) {
            const char *nm = g_scope[s].tensor->name;
            h = fnv1a(nm, strlen(nm), h);
            if (!j.is_raw[s]) rb |= fnv_range(bf, j.book_off[s], bookbytes, &h, buf, bufsz);
            if (j.nb[s]) {
                rb |= fnv_range(sf, j.idx_off[s], j.nb[s] * idxw, &h, buf, bufsz);
                rb |= fnv_range(sf, j.mn_off[s], j.nb[s] * 2, &h, buf, bufsz);
                rb |= fnv_range(sf, j.sc_off[s], j.nb[s] * 2, &h, buf, bufsz);
            }
            if (j.is_raw[s]) rb |= fnv_range(sf, j.raw_off[s], g_scope[s].tensor->count * 2, &h, buf, bufsz);
        }
        fclose(bf); fclose(sf); free(buf);
        if (rb) { fprintf(stderr, "re-read of staged files failed; nothing published\n"); remove(book_tmp); remove(stream_tmp); return 1; }
        gen = h;
        printf("generation digest pass: %lld s\n", (long long)(time(NULL) - tg));
        bf = fopen(book_tmp, "r+b"); sf = fopen(stream_tmp, "r+b");
        if (!bf || !sf) { fprintf(stderr, "cannot patch staged headers\n"); remove(book_tmp); remove(stream_tmp); return 1; }
        bad |= pt_write(bf, 20, &gen, 8, 1);
        bad |= pt_write(sf, 36 + (uint64_t)n * 8, &gen, 8, 1);
        if (fclose(bf) != 0) bad = 1;
        if (fclose(sf) != 0) bad = 1;
        if (bad) { fprintf(stderr, "header patch failed; nothing published\n"); remove(book_tmp); remove(stream_tmp); return 1; }
    }

    {
        char book_dst[1024], stream_dst[1024];
        snprintf(book_dst, sizeof(book_dst), "%s.nq2book", out_prefix);
        snprintf(stream_dst, sizeof(stream_dst), "%s.nq2", out_prefix);
        if (publish_pair(book_tmp, stream_tmp, book_dst, stream_dst) != 0) {
            fprintf(stderr, "cannot publish %s pair\n", out_prefix);
            remove(book_tmp); remove(stream_tmp);
            return 1;
        }
    }
    {
        char pathbuf[1088];
        snprintf(pathbuf, sizeof(pathbuf), "%s.nq2manifest", out_prefix);
        FILE *mf = fopen(pathbuf, "w");
        if (!mf) { fprintf(stderr, "cannot write %s\n", pathbuf); return 1; }
        fprintf(mf, "NQ2MANIFEST 1\n");
        fprintf(mf, "generation %016llx\n", (unsigned long long)gen);
        fprintf(mf, "tensors %d\n", n);
        for (int s = 0; s < n; s++)
            fprintf(mf, "tensor %s %llu %llu %u %016llx %016llx %s\n",
                    g_scope[s].tensor->name, (unsigned long long)block0s[s],
                    (unsigned long long)j.nb[s], block,
                    (unsigned long long)j.segd[s], (unsigned long long)j.bookd[s],
                    j.is_raw[s] ? "raw" : "packed");
        fprintf(mf, "end\n");
        if (fclose(mf) != 0) {
            fprintf(stderr, "the pair is published but its manifest is not; the pack carries its generation pairing only\n");
            return 1;
        }
        printf("manifest: %s\n", pathbuf);
    }

    uint64_t book_bytes = book_size;
    uint64_t stream_bytes = stream_size;
    uint64_t denom = tot_weights + tot_raw;
    double rate = denom ? (double)stream_bytes * 8.0 / (double)denom : 0.0;
    if (g_n_skipped) printf("skipped %d non-weight tensors (--skip-unsupported)\n", g_n_skipped);
    double crate = denom ? (double)(stream_bytes + book_bytes) * 8.0 / (double)denom : 0.0;
    char pathbuf[1024];
    snprintf(pathbuf, sizeof(pathbuf), "%s.report.txt", out_prefix);
    FILE *rf = fopen(pathbuf, "w");
    if (!rf) { fprintf(stderr, "cannot write %s\n", pathbuf); return 1; }
    fprintf(rf, "NanoQuatTwo quantization report (per-tensor books, format v%u)\n", 3u);
    fprintf(rf, "tensors packed: %d (one book each)\n", n);
    fprintf(rf, "weights packed: %llu\n", (unsigned long long)tot_weights);
    if (tot_raw)
        fprintf(rf, "weights stored as raw float16: %llu\n", (unsigned long long)tot_raw);
    fprintf(rf, "block size: %u, entries per book: %u\n", block, entries);
    fprintf(rf, "generation id: %016llx (book and stream; manifest %s.nq2manifest)\n",
            (unsigned long long)gen, out_prefix);
    fprintf(rf, "normalization: %s\n", g_norm_rms ? "rms (no offset)" : "minmax");
    fprintf(rf, "index stream (.nq2): %llu bytes\n", (unsigned long long)stream_bytes);
    fprintf(rf, "memory file (.nq2book): %llu bytes\n", (unsigned long long)book_bytes);
    fprintf(rf, "stream rate: %.4f bits per weight (stream only)\n", rate);
    fprintf(rf, "complete rate: %.4f bits per weight (stream + book, two-part accounting)\n", crate);
    fprintf(rf, "relative L2 error on real weights: %.4f\n", rel_l2);
    fprintf(rf, "per-tensor relative L2:\n");
    for (int s = 0; s < n; s++)
        fprintf(rf, "  %s: %.4f\n", g_scope[s].tensor->name,
                j.ref2[s] > 0 ? dsqrt(j.err2[s] / j.ref2[s]) : 0.0);
    if (fclose(rf) != 0) { fprintf(stderr, "write failed for %s\n", pathbuf); return 1; }
    printf("index stream: %llu bytes; memory file: %llu bytes\n",
           (unsigned long long)stream_bytes, (unsigned long long)book_bytes);
    printf("rate: %.4f bits per weight (stream only); complete %.4f (stream + book)\n", rate, crate);
    return 0;
}
int main(int argc, char **argv)
{
    const char *out_prefix = NULL;
    const char *filter = "";
    uint32_t block = 256, entries = 256;
    uint64_t budget = 250000000ull;
    uint64_t sample_cap = 32768;
    uint64_t tensor_sample_cap = 8192;
    int books_mode = 0;   /* 0 = one global book (v1), 1 = per-tensor books (v2) */
    int iters = 8;
    const char **shard_paths = (const char **)malloc(sizeof(char *) * (size_t)(argc + 1));
    if (!shard_paths) { fprintf(stderr, "out of memory\n"); return 1; }
#ifdef _WIN32
    _setmaxstdio(2048);   /* one open FILE per shard; the CRT default is 512 */
#endif
    int n_shard_paths = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_prefix = argv[++i];
        else if (strcmp(argv[i], "--block") == 0 && i + 1 < argc) block = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--entries") == 0 && i + 1 < argc) entries = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--budget") == 0 && i + 1 < argc) budget = strtoull(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--filter") == 0 && i + 1 < argc) filter = argv[++i];
        else if (strcmp(argv[i], "--sample-blocks") == 0 && i + 1 < argc) sample_cap = strtoull(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--tensor-sample-blocks") == 0 && i + 1 < argc) tensor_sample_cap = strtoull(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--books") == 0 && i + 1 < argc) { const char *m = argv[++i]; if (strcmp(m, "per-tensor") == 0) books_mode = 1; else if (strcmp(m, "global") == 0) books_mode = 0; else fprintf(stderr, "warning: unknown --books mode '%s', using global\n", m); }
        else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iters = (int)strtol(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--norm") == 0 && i + 1 < argc) { const char *m = argv[++i]; if (strcmp(m, "rms") == 0) g_norm_rms = 1; else if (strcmp(m, "minmax") == 0) g_norm_rms = 0; else fprintf(stderr, "warning: unknown --norm mode '%s', using minmax\n", m); }
        else if (strcmp(argv[i], "--residual") == 0) g_residual = 1;
        else if (strcmp(argv[i], "--format") == 0 && i + 1 < argc) g_format = (int)strtol(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--pq") == 0 && i + 1 < argc) g_pq_d = (int)strtol(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--budgets") == 0) g_budgets = 1;
        else if (strcmp(argv[i], "--embed-fp16") == 0) g_embed_fp16 = 1;
        else if (strcmp(argv[i], "--raw-names") == 0 && i + 1 < argc) { g_raw_names = argv[++i]; g_raw_any = 1; }
        else if (strcmp(argv[i], "--raw-small") == 0 && i + 1 < argc) { g_raw_small = strtoull(argv[++i], NULL, 10); g_raw_any = 1; }
        else if (strcmp(argv[i], "--skip-unsupported") == 0) g_skip_unsupported = 1;
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) g_threads = (int)strtol(argv[++i], NULL, 10);
        else if (argv[i][0] != '-') shard_paths[n_shard_paths++] = argv[i];   /* no fixed cap: sized from argc */
        else { fprintf(stderr, "bad argument: %s\n", argv[i]); return 1; }
    }
    if (g_threads <= 0) g_threads = cpu_count();
    if (g_threads > 256) g_threads = 256;
    if (!out_prefix || n_shard_paths == 0) {
        fprintf(stderr,
            "usage: nq2-quantize --out PREFIX shard.safetensors [more shards]\n"
            "         [--block 256] [--entries 256] [--budget N]\n"
            "         [--filter a,b] [--sample-blocks N] [--iters N]\n"
            "         [--threads N]   (default: logical CPUs; output identical for any N)\n"
            "         [--raw-names a,b] [--raw-small N] [--skip-unsupported]\n");
        return 1;
    }
    if (entries > 256) {
        fprintf(stderr, "entries above 256 do not fit an 8-bit index\n");
        return 1;
    }
    if (entries == 0 || block == 0) {
        fprintf(stderr, "block and entries must both be at least 1\n");
        return 1;
    }
    /* The writer must not accept geometry its readers refuse:
     * every reader rejects block above 4096 (Codex Astra
     * review M-P2, 2026-11-25). */
    if (block > 4096) {
        fprintf(stderr, "block above 4096 exceeds the geometry "
                "limit every reader enforces\n");
        return 1;
    }
    /* The sample caps are validated here, before the books-mode
     * branch: the zero-cap check used to sit in the global path
     * only, after per-tensor mode had already trained on an
     * empty sample (Codex Astra review M-P3, 2026-11-25). */
    if (sample_cap == 0 || tensor_sample_cap == 0) {
        fprintf(stderr, "--sample-blocks and "
                "--tensor-sample-blocks must be at least 1\n");
        return 1;
    }
    /* Format v3 option discipline: the v3-only options need
     * per-tensor books (their layouts are per-tensor), and a
     * legacy format cannot express them. */
    if (g_format != 1 && g_format != 2 && g_format != 3) {
        fprintf(stderr, "--format must be 1, 2, or 3\n");
        return 1;
    }
    if (g_embed_fp16) g_raw_any = 1;
    if ((g_pq_d || g_budgets || g_raw_any) && books_mode != 1) {
        fprintf(stderr, "--pq, --budgets, and the raw options need "
                "--books per-tensor\n");
        return 1;
    }
    if ((g_pq_d || g_budgets || g_raw_any) && g_format != 3) {
        fprintf(stderr, "--pq, --budgets, and the raw options are "
                "format v3 options\n");
        return 1;
    }
    if (books_mode == 1 && g_format == 1) {
        fprintf(stderr, "per-tensor books are format 2 or 3\n");
        return 1;
    }
    if (books_mode == 0 && g_format == 2) {
        fprintf(stderr, "the global-book layout is format 1 or 3\n");
        return 1;
    }
    if (g_pq_d && (g_pq_d != 4 && g_pq_d != 8)) {
        fprintf(stderr, "--pq subvector dim must be 4 or 8\n");
        return 1;
    }
    if (g_pq_d && block != 256) {
        fprintf(stderr, "--pq is defined at block 256 (the frame "
                "the probe measured)\n");
        return 1;
    }
    if (g_pq_d && g_budgets) {
        fprintf(stderr, "--pq and --budgets are not supported "
                "together (one index width per pack)\n");
        return 1;
    }
    if (g_budgets && g_residual) {
        fprintf(stderr, "--budgets and --residual are not "
                "supported together\n");
        return 1;
    }
    /* Serialize in canonical order regardless of argv order
     * (H1; see cmp_shard_arg above). */
    qsort(shard_paths, (size_t)n_shard_paths,
          sizeof(shard_paths[0]), cmp_shard_arg);

    /* Open shards and build the scope: whole tensors, in order,
     * while the running weight total is under the budget. */
    Shard *shards = (Shard *)calloc(n_shard_paths, sizeof(Shard));
    g_shards = shards; g_n_shards = n_shard_paths;
    int scope_cap = 256;
    g_scope = (ScopeItem *)malloc(sizeof(ScopeItem) * scope_cap);
    uint64_t scope_weights = 0;
    for (int s = 0; s < n_shard_paths; s++) {
        if (shard_open(&shards[s], shard_paths[s]) != 0) return 1;
        for (int t = 0; t < shards[s].n_tensors; t++) {
            TensorInfo *ti = &shards[s].tensors[t];
            if (!name_in_filter(ti->name, filter)) continue;
            if (scope_weights >= budget) continue;
            if (g_n_scope == scope_cap) {
                scope_cap *= 2;
                g_scope = (ScopeItem *)realloc(g_scope, sizeof(ScopeItem) * scope_cap);
            }
            g_scope[g_n_scope].shard = &shards[s];
            g_scope[g_n_scope].tensor = ti;
            g_n_scope++;
            scope_weights += ti->count;
            g_total_blocks += ti->count / block;
        }
    }
    g_total_weights = g_total_blocks * block;
    printf("scope: %d tensors, %llu weights in %llu blocks of %u\n",
           g_n_scope, (unsigned long long)g_total_weights,
           (unsigned long long)g_total_blocks, block);
    if (g_total_blocks == 0) { fprintf(stderr, "nothing in scope\n"); return 1; }

    if (books_mode == 1) {
        /* Per-tensor books. Each tensor is sampled, learned,
         * and packed with its own codebook, so every book
         * speaks only its own tensor's language.
         * Format 2 (legacy, --format 2) writes the v2 layout:
         * a tensor table of blocks per tensor at one shared
         * block size. Format 3 (the default) keeps that layout
         * and adds the H3b identity: the generation id in both
         * headers and the per-tensor digest manifest beside
         * the pair. Format 3 is also the only format that can
         * express --pq (per-tensor sub-books), --budgets
         * (per-tensor block sizes), and --embed-fp16 (a raw
         * float16 section in place of one tensor's blocks). */
        if (g_budgets && block < 128) {
            fprintf(stderr, "--budgets needs block >= 128 to "
                    "have room to step down\n");
            return 1;
        }
        /* Flat v3 per-tensor books on more than one thread: the
         * parallel, byte-identical, RAM-bounded path. */
        if (g_format == 3 && !g_budgets && !g_residual
                && g_threads > 1)
            return per_tensor_parallel(out_prefix, block, entries, iters,
                                       tensor_sample_cap);
        uint32_t idxw = g_pq_d ? block / (uint32_t)g_pq_d : 1;
        TOut *outs = (TOut *)calloc((size_t)g_n_scope, sizeof(TOut));
        double *terr = (double *)calloc((size_t)g_n_scope, sizeof(double));
        double *tref = (double *)calloc((size_t)g_n_scope, sizeof(double));
        uint32_t *tblock_sz = (uint32_t *)malloc(
            sizeof(uint32_t) * (size_t)g_n_scope);
        double res_err2 = 0.0;
        if (!outs || !terr || !tref || !tblock_sz) {
            fprintf(stderr, "out of memory\n");
            return 1;
        }
        for (int s = 0; s < g_n_scope; s++) tblock_sz[s] = block;

        /* First pass at the base block: every tensor packed
         * with its own book (or stored raw, for the embedding
         * under --embed-fp16). The budget pass below re-packs
         * the sensitive tensors at smaller blocks. */
        for (int s = 0; s < g_n_scope; s++) {
            TensorInfo *ti = g_scope[s].tensor;
            printf("tensor %d/%d %s: %llu blocks\n", s + 1, g_n_scope,
                   ti->name, (unsigned long long)(ti->count / block));
            if (tensor_is_raw(ti, block)) {
                if (store_one_tensor_fp16(s, &outs[s]) != 0) return 1;
                outs[s].block = block;
                printf("  stored as raw float16 (no book, no "
                       "blocks); round-trip relative L2 %.4f\n",
                       outs[s].ref2 > 0
                           ? dsqrt(outs[s].err2 / outs[s].ref2) : 0.0);
            } else {
                if (pack_one_tensor(s, block, entries, iters,
                                    tensor_sample_cap, &outs[s]) != 0)
                    return 1;
                if (outs[s].frame_bad) {
                    fprintf(stderr, "tensor %s, block %llu: frame "
                            "value %g is not representable in "
                            "float16; refusing to publish a pack "
                            "with non-finite frames (Sol H8)\n",
                            ti->name,
                            (unsigned long long)outs[s].bad_block,
                            (double)outs[s].bad_value);
                    return 1;
                }
                printf("  tensor relative L2: %.4f\n",
                       outs[s].ref2 > 0
                           ? dsqrt(outs[s].err2 / outs[s].ref2) : 0.0);
                if (g_residual && outs[s].nblocks) {
                    /* Residual stage (experimental, measured
                     * only), as before: a second book trained
                     * on this tensor's stage-1 residuals. */
                    ResSampleCtx rctx;
                    memset(&rctx, 0, sizeof(rctx));
                    rctx.block = block;
                    rctx.entries = entries;
                    rctx.cent1 = outs[s].book;
                    rctx.inner.block = block;
                    uint64_t cap = tensor_sample_cap;
                    if (cap > outs[s].nblocks) cap = outs[s].nblocks;
                    rctx.inner.sample_cap = cap;
                    rctx.inner.sample = (float *)malloc(
                        cap * block * sizeof(float));
                    rctx.inner.tmp = (float *)malloc(
                        block * sizeof(float));
                    rctx.tmp = (float *)malloc(block * sizeof(float));
                    rctx.recon = (float *)malloc(block * sizeof(float));
                    rctx.resid = (float *)malloc(block * sizeof(float));
                    rctx.ntmp = (float *)malloc(block * sizeof(float));
                    if (!rctx.inner.sample || !rctx.tmp || !rctx.recon
                            || !rctx.resid || !rctx.ntmp) {
                        fprintf(stderr, "out of memory (residual)\n");
                        return 1;
                    }
                    walk_one_tensor(s, block, res_sample_fn, &rctx);
                    float *cent2 = (float *)malloc(
                        (size_t)entries * block * sizeof(float));
                    if (!cent2) {
                        fprintf(stderr, "out of memory (residual)\n");
                        return 1;
                    }
                    learn_book(rctx.inner.sample, rctx.inner.sample_n,
                               block, entries, iters, cent2);
                    free(rctx.inner.sample); free(rctx.inner.tmp);
                    ResFidCtx fctx;
                    memset(&fctx, 0, sizeof(fctx));
                    fctx.block = block;
                    fctx.entries = entries;
                    fctx.cent1 = outs[s].book;
                    fctx.cent2 = cent2;
                    fctx.tmp = rctx.tmp;
                    fctx.recon = rctx.recon;
                    fctx.resid = rctx.resid;
                    fctx.ntmp = rctx.ntmp;
                    walk_one_tensor(s, block, res_fid_fn, &fctx);
                    res_err2 += fctx.err2;
                    printf("  residual stage: two-stage relative "
                           "L2 %.4f\n",
                           fctx.ref2 > 0
                               ? dsqrt(fctx.err2 / fctx.ref2) : 0.0);
                    free(cent2);
                    free(rctx.tmp); free(rctx.recon);
                    free(rctx.resid); free(rctx.ntmp);
                }
            }
            terr[s] = outs[s].err2;
            tref[s] = outs[s].ref2;
        }

        if (g_budgets) {
            /* Sensitivity weighting (format v3): rank tensors
             * by their share of the total squared error at the
             * base block (the concentration the errprobe
             * measured: the top 10 of 241 tensors held 54.6
             * percent of the squared error in the frozen
             * pack). Tensors carrying the first half of the
             * squared error are re-packed at block/4, the next
             * 30 percent at block/2, the rest keep the base
             * block; books are re-learned at the new size. */
            double tot_err = 0.0;
            for (int s = 0; s < g_n_scope; s++) tot_err += outs[s].err2;
            int *order = (int *)malloc(sizeof(int) * (size_t)g_n_scope);
            if (!order) { fprintf(stderr, "out of memory\n"); return 1; }
            for (int s = 0; s < g_n_scope; s++) order[s] = s;
            for (int a = 1; a < g_n_scope; a++) {
                int v = order[a], b = a - 1;
                while (b >= 0 && outs[order[b]].err2 < outs[v].err2) {
                    order[b + 1] = order[b];
                    b--;
                }
                order[b + 1] = v;
            }
            uint32_t q_blk = block / 4, h_blk = block / 2;
            if (q_blk < 32) q_blk = 32;
            if (h_blk < 32) h_blk = 32;
            double cum = 0.0;
            int n_q = 0, n_h = 0;
            for (int oi = 0; oi < g_n_scope; oi++) {
                int s = order[oi];
                uint32_t want = block;
                if (tot_err > 0 && !outs[s].raw && outs[s].nblocks) {
                    double share_before = cum / tot_err;
                    if (share_before < 0.50) want = q_blk;
                    else if (share_before < 0.80) want = h_blk;
                }
                cum += outs[s].err2;
                tblock_sz[s] = want;
                if (want == q_blk && want != block) n_q++;
                else if (want == h_blk && want != block) n_h++;
            }
            free(order);
            for (int s = 0; s < g_n_scope; s++) {
                if (tblock_sz[s] == block || outs[s].raw) continue;
                printf("budget: %s re-packed at block %u (was "
                       "%u); squared-error share %.4f\n",
                       g_scope[s].tensor->name, tblock_sz[s], block,
                       tot_err > 0 ? outs[s].err2 / tot_err : 0.0);
                tout_free(&outs[s]);
                if (pack_one_tensor(s, tblock_sz[s], entries, iters,
                                    tensor_sample_cap, &outs[s]) != 0)
                    return 1;
                if (outs[s].frame_bad) {
                    fprintf(stderr, "tensor %s, block %llu: frame "
                            "value %g is not representable in "
                            "float16; refusing to publish (Sol "
                            "H8)\n", g_scope[s].tensor->name,
                            (unsigned long long)outs[s].bad_block,
                            (double)outs[s].bad_value);
                    return 1;
                }
                terr[s] = outs[s].err2;
                tref[s] = outs[s].ref2;
                printf("  tensor relative L2 at block %u: %.4f\n",
                       tblock_sz[s],
                       tref[s] > 0 ? dsqrt(terr[s] / tref[s]) : 0.0);
            }
            printf("budgets: %d tensors at block %u, %d at block "
                   "%u, %d at block %u\n",
                   n_q, q_blk, n_h, h_blk,
                   g_n_scope - n_q - n_h, block);
        }

        /* Totals over what will actually be written. */
        uint64_t tot_blocks = 0, tot_weights = 0, tot_raw = 0;
        uint64_t *block0s = (uint64_t *)malloc(
            sizeof(uint64_t) * (size_t)g_n_scope);
        int n_books_stored = 0;
        if (!block0s) { fprintf(stderr, "out of memory\n"); return 1; }
        for (int s = 0; s < g_n_scope; s++) {
            block0s[s] = tot_blocks;
            tot_blocks += outs[s].nblocks;
            tot_weights += outs[s].nblocks * outs[s].block;
            if (outs[s].raw) tot_raw += outs[s].raw_count;
            else n_books_stored++;
        }
        double tot_err2 = 0.0, tot_ref2 = 0.0;
        for (int s = 0; s < g_n_scope; s++) {
            tot_err2 += terr[s];
            tot_ref2 += tref[s];
        }
        double rel_l2 = tot_ref2 > 0 ? dsqrt(tot_err2 / tot_ref2) : 0.0;
        printf("packed %llu blocks in %d per-tensor books; "
               "relative L2 error %.4f\n",
               (unsigned long long)tot_blocks, n_books_stored, rel_l2);

        uint32_t flags = 0;
        if (g_pq_d) flags |= 1u;
        if (g_budgets) flags |= 2u;
        if (tot_raw) flags |= 4u;
        uint64_t gen = 0;
        if (g_format == 3)
            gen = generation_of(block, entries, flags, outs);

        char pathbuf[1024], book_tmp[1088], stream_tmp[1088];
        int bad = 0;
        snprintf(book_tmp, sizeof(book_tmp), "%s.nq2book.%ld.tmp",
                 out_prefix, writer_pid());
        FILE *bf = fopen(book_tmp, "wb");
        if (!bf) { fprintf(stderr, "cannot write %s\n", book_tmp); return 1; }
        if (g_format == 3) {
            uint32_t version = 3;
            uint32_t nb32 = (uint32_t)n_books_stored;
            bad |= wr(bf, "NQ2B", 1, 4);
            bad |= wr(bf, &version, 4, 1);
            bad |= wr(bf, &block, 4, 1);
            bad |= wr(bf, &entries, 4, 1);
            bad |= wr(bf, &nb32, 4, 1);
            bad |= wr(bf, &gen, 8, 1);
            if (flags & 1u) {
                uint32_t d32 = (uint32_t)g_pq_d;
                bad |= wr(bf, &d32, 4, 1);
            }
        } else {
            uint32_t version = 2;
            uint32_t nb32 = (uint32_t)g_n_scope;
            bad |= wr(bf, "NQ2B", 1, 4);
            bad |= wr(bf, &version, 4, 1);
            bad |= wr(bf, &block, 4, 1);
            bad |= wr(bf, &entries, 4, 1);
            bad |= wr(bf, &nb32, 4, 1);
        }
        for (int s = 0; s < g_n_scope; s++) {
            if (outs[s].raw) continue;
            bad |= wr(bf, outs[s].book, sizeof(float),
                      (size_t)entries * outs[s].block);
        }
        if (fclose(bf) != 0) bad = 1;
        if (bad) {
            fprintf(stderr, "write failed for %s.nq2book; "
                    "nothing published\n", out_prefix);
            remove(book_tmp);
            return 1;
        }

        snprintf(stream_tmp, sizeof(stream_tmp), "%s.nq2.%ld.tmp",
                 out_prefix, writer_pid());
        FILE *sf = fopen(stream_tmp, "wb");
        if (!sf) {
            fprintf(stderr, "cannot write %s\n", stream_tmp);
            remove(book_tmp);
            return 1;
        }
        if (g_format == 3) {
            uint32_t version = 3;
            uint32_t nseg = (uint32_t)g_n_scope;
            bad |= wr(sf, "NQ2\0", 1, 4);
            bad |= wr(sf, &version, 4, 1);
            bad |= wr(sf, &block, 4, 1);
            bad |= wr(sf, &entries, 4, 1);
            bad |= wr(sf, &tot_weights, 8, 1);
            bad |= wr(sf, &tot_blocks, 8, 1);
            bad |= wr(sf, &nseg, 4, 1);
            for (int s = 0; s < g_n_scope; s++)
                bad |= wr(sf, &outs[s].nblocks, 8, 1);
            bad |= wr(sf, &gen, 8, 1);
            bad |= wr(sf, &flags, 4, 1);
            if (flags & 1u) {
                uint32_t d32 = (uint32_t)g_pq_d;
                bad |= wr(sf, &d32, 4, 1);
            }
            if (flags & 2u)
                for (int s = 0; s < g_n_scope; s++)
                    bad |= wr(sf, &tblock_sz[s], 4, 1);
            if (flags & 4u) {
                uint32_t n_raw = 0;
                for (int s = 0; s < g_n_scope; s++)
                    if (outs[s].raw) n_raw++;
                bad |= wr(sf, &n_raw, 4, 1);
                for (int s = 0; s < g_n_scope; s++)
                    if (outs[s].raw) {
                        uint32_t ix = (uint32_t)s;
                        bad |= wr(sf, &ix, 4, 1);
                    }
            }
        } else {
            uint32_t version = 2;
            uint32_t nb32 = (uint32_t)g_n_scope;
            uint64_t tw = tot_weights, tb = tot_blocks;
            bad |= wr(sf, "NQ2\0", 1, 4);
            bad |= wr(sf, &version, 4, 1);
            bad |= wr(sf, &block, 4, 1);
            bad |= wr(sf, &entries, 4, 1);
            bad |= wr(sf, &tw, 8, 1);
            bad |= wr(sf, &tb, 8, 1);
            bad |= wr(sf, &nb32, 4, 1);
            for (int s = 0; s < g_n_scope; s++)
                bad |= wr(sf, &outs[s].nblocks, 8, 1);
        }
        for (int s = 0; s < g_n_scope; s++)
            if (outs[s].nblocks)
                bad |= wr(sf, outs[s].idx, 1,
                          (size_t)outs[s].nblocks * idxw);
        for (int s = 0; s < g_n_scope; s++)
            if (outs[s].nblocks)
                bad |= wr(sf, outs[s].mn_h, 2, (size_t)outs[s].nblocks);
        for (int s = 0; s < g_n_scope; s++)
            if (outs[s].nblocks)
                bad |= wr(sf, outs[s].sc_h, 2, (size_t)outs[s].nblocks);
        if (flags & 4u)
            for (int s = 0; s < g_n_scope; s++)
                if (outs[s].raw)
                    bad |= wr(sf, outs[s].raw, 2,
                              (size_t)outs[s].raw_count);
        if (fclose(sf) != 0) bad = 1;
        if (bad) {
            fprintf(stderr, "write failed for %s.nq2; "
                    "nothing published\n", out_prefix);
            remove(stream_tmp);
            remove(book_tmp);
            return 1;
        }
        /* Both artifacts are complete on disc; publish the
         * pair as one commit (publish_pair, Sol H4). */
        {
            char book_dst[1024], stream_dst[1024];
            snprintf(book_dst, sizeof(book_dst), "%s.nq2book",
                     out_prefix);
            snprintf(stream_dst, sizeof(stream_dst), "%s.nq2",
                     out_prefix);
            if (publish_pair(book_tmp, stream_tmp,
                             book_dst, stream_dst) != 0) {
                fprintf(stderr, "cannot publish %s pair\n",
                        out_prefix);
                remove(book_tmp);
                remove(stream_tmp);
                return 1;
            }
        }
        if (g_format == 3
                && write_manifest(out_prefix, gen, outs, block0s,
                                  idxw, entries) != 0) {
            fprintf(stderr, "the pair is published but its "
                    "manifest is not; the pack carries its "
                    "generation pairing only\n");
            return 1;
        }

        uint64_t book_floats = 0;
        for (int s = 0; s < g_n_scope; s++)
            if (!outs[s].raw)
                book_floats += (uint64_t)entries * outs[s].block;
        uint64_t book_bytes = (g_format == 3 ? 28 : 20)
            + ((flags & 1u) ? 4 : 0) + book_floats * 4;
        uint64_t stream_header = (g_format == 3)
            ? 48 + (uint64_t)g_n_scope * 8
              + ((flags & 1u) ? 4 : 0)
              + ((flags & 2u) ? (uint64_t)g_n_scope * 4 : 0)
              + ((flags & 4u) ? 4 + (uint64_t)(g_n_scope) * 0 : 0)
            : 36 + (uint64_t)g_n_scope * 8;
        if ((flags & 4u) && g_format == 3) {
            uint64_t n_raw = 0;
            for (int s = 0; s < g_n_scope; s++)
                if (outs[s].raw) n_raw++;
            stream_header += 4 + n_raw * 4 - 4;
        }
        uint64_t stream_bytes = stream_header
            + tot_blocks * idxw + tot_blocks * 4 + tot_raw * 2;
        uint64_t denom = tot_weights + tot_raw;
        double rate = denom
            ? (double)stream_bytes * 8.0 / (double)denom : 0.0;
        double crate = denom
            ? (double)(stream_bytes + book_bytes) * 8.0
              / (double)denom : 0.0;
        snprintf(pathbuf, sizeof(pathbuf), "%s.report.txt", out_prefix);
        FILE *rf = fopen(pathbuf, "w");
        if (!rf) {
            fprintf(stderr, "cannot write %s\n", pathbuf);
            return 1;
        }
        {
            fprintf(rf, "NanoQuatTwo quantization report "
                    "(per-tensor books, format v%u)\n",
                    g_format == 3 ? 3u : 2u);
            fprintf(rf, "tensors packed: %d (one book each)\n",
                    g_n_scope);
            fprintf(rf, "weights packed: %llu\n",
                    (unsigned long long)tot_weights);
            if (tot_raw)
                fprintf(rf, "weights stored as raw float16: %llu\n",
                        (unsigned long long)tot_raw);
            fprintf(rf, "block size: %u, entries per book: %u\n",
                    block, entries);
            if (g_pq_d)
                fprintf(rf, "product quantization: subvector dim "
                        "%d, %u sub-books per tensor\n",
                        g_pq_d, block / (uint32_t)g_pq_d);
            if (g_budgets)
                fprintf(rf, "per-tensor bit budgets: squared-error "
                        "shares decide block size (block/4 for the "
                        "first 50 percent of squared error, block/2 "
                        "for the next 30 percent)\n");
            if (g_format == 3)
                fprintf(rf, "generation id: %016llx (book and "
                        "stream; manifest %s.nq2manifest)\n",
                        (unsigned long long)gen, out_prefix);
            fprintf(rf, "normalization: %s\n",
                    g_norm_rms ? "rms (no offset)" : "minmax");
            fprintf(rf, "index stream (.nq2): %llu bytes\n",
                    (unsigned long long)stream_bytes);
            fprintf(rf, "memory file (.nq2book): %llu bytes\n",
                    (unsigned long long)book_bytes);
            fprintf(rf, "stream rate: %.4f bits per weight "
                    "(stream only)\n", rate);
            fprintf(rf, "complete rate: %.4f bits per weight "
                    "(stream + book, two-part accounting)\n", crate);
            fprintf(rf, "relative L2 error on real weights: %.4f\n",
                    rel_l2);
            if (g_residual) {
                double two_l2 = tot_ref2 > 0
                    ? dsqrt(res_err2 / tot_ref2) : 0.0;
                uint64_t stream2 = stream_bytes
                    + tot_blocks * (idxw + 4);
                uint64_t book2 = book_bytes
                    + (uint64_t)n_books_stored * entries * block * 4;
                double crate2 = denom
                    ? (double)(stream2 + book2) * 8.0 / (double)denom
                    : 0.0;
                fprintf(rf, "residual stage (experimental, measured "
                        "only; the written stream is "
                        "single-stage):\n");
                fprintf(rf, "  two-stage relative L2: %.4f\n", two_l2);
                fprintf(rf, "  two-stage complete rate would be: "
                        "%.4f bits per weight\n", crate2);
            }
            fprintf(rf, "per-tensor relative L2:\n");
            for (int s = 0; s < g_n_scope; s++)
                fprintf(rf, "  %s: %.4f\n", g_scope[s].tensor->name,
                        tref[s] > 0 ? dsqrt(terr[s] / tref[s]) : 0.0);
            if (fclose(rf) != 0) {
                fprintf(stderr, "write failed for %s\n", pathbuf);
                return 1;
            }
        }
        printf("index stream: %llu bytes; memory file: %llu bytes\n",
               (unsigned long long)stream_bytes,
               (unsigned long long)book_bytes);
        printf("rate: %.4f bits per weight (stream only); "
               "complete %.4f (stream + book)\n", rate, crate);
        return 0;
    }

    /* Pass 1: sample normalized blocks from across the scope.
     * (The sample caps were validated before the books-mode
     * branch; see M-P3 at the argument checks.) */
    if (sample_cap > g_total_blocks) sample_cap = g_total_blocks;
    SampleCtx sctx;
    memset(&sctx, 0, sizeof(sctx));
    sctx.block = block;
    sctx.sample_cap = sample_cap;
    sctx.sample = (float *)malloc(sample_cap * block * sizeof(float));
    sctx.tmp = (float *)malloc(block * sizeof(float));
    if (!sctx.sample || !sctx.tmp) { fprintf(stderr, "out of memory sampling\n"); return 1; }
    { time_t t_a = time(NULL); sample_global_fast(&sctx); printf("pass 1 (sampling): %lld s\n", (long long)(time(NULL) - t_a)); fflush(stdout); }
    printf("sampled %llu blocks for k-means\n",
           (unsigned long long)sctx.sample_n);

    /* Learn the book: k-means (Lloyd) on the sample. */
    float *cent = (float *)malloc((size_t)entries * block * sizeof(float));
    float *tmp = (float *)malloc(block * sizeof(float));
    if (!cent || !tmp) { fprintf(stderr, "out of memory learning\n"); return 1; }
    { time_t t_a = time(NULL); learn_book(sctx.sample, sctx.sample_n, block, entries, iters, cent); printf("k-means (%d threads): %lld s\n", g_threads, (long long)(time(NULL) - t_a)); fflush(stdout); }
    free(sctx.sample); free(sctx.tmp);

    /* Pass 2: pack every block and measure true fidelity. */
    PackCtx pctx;
    memset(&pctx, 0, sizeof(pctx));
    pctx.block = block;
    pctx.entries = entries;
    pctx.cent = cent;
    pctx.idx = (uint8_t *)malloc(g_total_blocks);
    pctx.mn_h = (uint16_t *)malloc(g_total_blocks * 2);
    pctx.sc_h = (uint16_t *)malloc(g_total_blocks * 2);
    if (!pctx.idx || !pctx.mn_h || !pctx.sc_h) {
        fprintf(stderr, "out of memory packing\n");
        return 1;
    }
    pctx.tmp = tmp;
    { time_t t_a = time(NULL); pack_parallel(&pctx); printf("pass 2 (pack, %d threads): %lld s\n", g_threads, (long long)(time(NULL) - t_a)); fflush(stdout); }
    if (pctx.frame_bad) {
        fprintf(stderr, "block %llu: frame value %g is not "
                "representable in float16; refusing to publish "
                "a pack with non-finite frames (Sol H8)\n",
                (unsigned long long)pctx.bad_block,
                (double)pctx.bad_value);
        return 1;
    }
    double rel_l2 = pctx.ref2 > 0 ? dsqrt(pctx.err2 / pctx.ref2) : 0.0;
    printf("packed %llu blocks; relative L2 error %.4f\n",
           (unsigned long long)pctx.cursor, rel_l2);

    if (g_format == 3) {
        /* Format v3, global book: the v1 layout plus the
         * segment table, the generation id in both headers,
         * and the digest manifest (H3b). The book stays
         * shared; every tensor's manifest line carries the
         * shared book's digest. */
        TOut *outs = (TOut *)calloc((size_t)g_n_scope, sizeof(TOut));
        uint64_t *block0s = (uint64_t *)malloc(
            sizeof(uint64_t) * (size_t)g_n_scope);
        if (!outs || !block0s) {
            fprintf(stderr, "out of memory\n");
            return 1;
        }
        uint64_t off = 0;
        for (int s = 0; s < g_n_scope; s++) {
            uint64_t tb = g_scope[s].tensor->count / block;
            block0s[s] = off;
            outs[s].book = cent;
            outs[s].idx = pctx.idx + off;
            outs[s].mn_h = pctx.mn_h + off;
            outs[s].sc_h = pctx.sc_h + off;
            outs[s].nblocks = tb;
            outs[s].block = block;
            off += tb;
        }
        uint64_t gen = generation_of(block, entries, 0, outs);
        char pathbuf[1024], book_tmp[1088], stream_tmp[1088];
        int bad = 0;
        uint32_t version = 3, one = 1, flags = 0;
        snprintf(book_tmp, sizeof(book_tmp), "%s.nq2book.%ld.tmp",
                 out_prefix, writer_pid());
        FILE *bf = fopen(book_tmp, "wb");
        if (!bf) { fprintf(stderr, "cannot write %s\n", book_tmp); return 1; }
        bad |= wr(bf, "NQ2B", 1, 4);
        bad |= wr(bf, &version, 4, 1);
        bad |= wr(bf, &block, 4, 1);
        bad |= wr(bf, &entries, 4, 1);
        bad |= wr(bf, &one, 4, 1);
        bad |= wr(bf, &gen, 8, 1);
        bad |= wr(bf, cent, sizeof(float), (size_t)entries * block);
        if (fclose(bf) != 0) bad = 1;
        if (bad) {
            fprintf(stderr, "write failed for %s.nq2book; "
                    "nothing published\n", out_prefix);
            remove(book_tmp);
            return 1;
        }
        snprintf(stream_tmp, sizeof(stream_tmp), "%s.nq2.%ld.tmp",
                 out_prefix, writer_pid());
        FILE *sf = fopen(stream_tmp, "wb");
        if (!sf) {
            fprintf(stderr, "cannot write %s\n", stream_tmp);
            remove(book_tmp);
            return 1;
        }
        {
            uint32_t nseg = (uint32_t)g_n_scope;
            bad |= wr(sf, "NQ2\0", 1, 4);
            bad |= wr(sf, &version, 4, 1);
            bad |= wr(sf, &block, 4, 1);
            bad |= wr(sf, &entries, 4, 1);
            bad |= wr(sf, &g_total_weights, 8, 1);
            bad |= wr(sf, &g_total_blocks, 8, 1);
            bad |= wr(sf, &nseg, 4, 1);
            for (int s = 0; s < g_n_scope; s++)
                bad |= wr(sf, &outs[s].nblocks, 8, 1);
            bad |= wr(sf, &gen, 8, 1);
            bad |= wr(sf, &flags, 4, 1);
        }
        bad |= wr(sf, pctx.idx, 1, g_total_blocks);
        bad |= wr(sf, pctx.mn_h, 2, g_total_blocks);
        bad |= wr(sf, pctx.sc_h, 2, g_total_blocks);
        if (fclose(sf) != 0) bad = 1;
        if (bad) {
            fprintf(stderr, "write failed for %s.nq2; "
                    "nothing published\n", out_prefix);
            remove(stream_tmp);
            remove(book_tmp);
            return 1;
        }
        {
            char book_dst[1024], stream_dst[1024];
            snprintf(book_dst, sizeof(book_dst), "%s.nq2book",
                     out_prefix);
            snprintf(stream_dst, sizeof(stream_dst), "%s.nq2",
                     out_prefix);
            if (publish_pair(book_tmp, stream_tmp,
                             book_dst, stream_dst) != 0) {
                fprintf(stderr, "cannot publish %s pair\n",
                        out_prefix);
                remove(book_tmp);
                remove(stream_tmp);
                return 1;
            }
        }
        if (write_manifest(out_prefix, gen, outs, block0s, 1,
                           entries) != 0) {
            fprintf(stderr, "the pair is published but its "
                    "manifest is not; the pack carries its "
                    "generation pairing only\n");
            return 1;
        }
        uint64_t book_bytes = 28 + (uint64_t)entries * block * 4;
        uint64_t stream_bytes = 48 + (uint64_t)g_n_scope * 8
                                + g_total_blocks * 5;
        double rate = (double)stream_bytes * 8.0
                      / (double)g_total_weights;
        double crate = (double)(stream_bytes + book_bytes) * 8.0
                       / (double)g_total_weights;
        snprintf(pathbuf, sizeof(pathbuf), "%s.report.txt", out_prefix);
        FILE *rf = fopen(pathbuf, "w");
        if (!rf) {
            fprintf(stderr, "cannot write %s\n", pathbuf);
            return 1;
        }
        {
            fprintf(rf, "NanoQuatTwo quantization report "
                    "(format v3, global book)\n");
            fprintf(rf, "tensors packed: %d\n", g_n_scope);
            fprintf(rf, "weights packed: %llu\n",
                    (unsigned long long)g_total_weights);
            fprintf(rf, "block size: %u, entries: %u\n",
                    block, entries);
            fprintf(rf, "generation id: %016llx (book and stream; "
                    "manifest %s.nq2manifest)\n",
                    (unsigned long long)gen, out_prefix);
            fprintf(rf, "normalization: %s\n",
                    g_norm_rms ? "rms (no offset)" : "minmax");
            fprintf(rf, "index stream (.nq2): %llu bytes\n",
                    (unsigned long long)stream_bytes);
            fprintf(rf, "memory file (.nq2book): %llu bytes\n",
                    (unsigned long long)book_bytes);
            fprintf(rf, "stream rate: %.4f bits per weight "
                    "(stream only)\n", rate);
            fprintf(rf, "complete rate: %.4f bits per weight "
                    "(stream + book, two-part accounting)\n", crate);
            fprintf(rf, "relative L2 error on real weights: "
                    "%.4f\n", rel_l2);
            if (fclose(rf) != 0) {
                fprintf(stderr, "write failed for %s\n", pathbuf);
                return 1;
            }
        }
        printf("index stream: %llu bytes; memory file: %llu "
               "bytes\n", (unsigned long long)stream_bytes,
               (unsigned long long)book_bytes);
        printf("rate: %.4f bits per weight (stream only); "
               "complete %.4f (stream + book)\n", rate, crate);
        return 0;
    }

    /* Write the memory file and the index stream, staged and
     * checked (H6; see wr() above). */
    char pathbuf[1024], book_tmp[1088], stream_tmp[1088];
    int bad = 0;
    snprintf(book_tmp, sizeof(book_tmp), "%s.nq2book.%ld.tmp",
             out_prefix, writer_pid());
    FILE *bf = fopen(book_tmp, "wb");
    if (!bf) { fprintf(stderr, "cannot write %s\n", book_tmp); return 1; }
    bad |= wr(bf, "NQ2B", 1, 4);
    bad |= wr(bf, &block, 4, 1);
    bad |= wr(bf, &entries, 4, 1);
    bad |= wr(bf, cent, sizeof(float), (size_t)entries * block);
    if (fclose(bf) != 0) bad = 1;
    if (bad) {
        fprintf(stderr, "write failed for %s.nq2book; "
                "nothing published\n", out_prefix);
        remove(book_tmp);
        return 1;
    }

    snprintf(stream_tmp, sizeof(stream_tmp), "%s.nq2.%ld.tmp",
             out_prefix, writer_pid());
    FILE *sf = fopen(stream_tmp, "wb");
    if (!sf) {
        fprintf(stderr, "cannot write %s\n", stream_tmp);
        remove(book_tmp);
        return 1;
    }
    uint32_t version = 1;
    bad |= wr(sf, "NQ2\0", 1, 4);
    bad |= wr(sf, &version, 4, 1);
    bad |= wr(sf, &block, 4, 1);
    bad |= wr(sf, &entries, 4, 1);
    bad |= wr(sf, &g_total_weights, 8, 1);
    bad |= wr(sf, &g_total_blocks, 8, 1);
    bad |= wr(sf, pctx.idx, 1, g_total_blocks);
    bad |= wr(sf, pctx.mn_h, 2, g_total_blocks);
    bad |= wr(sf, pctx.sc_h, 2, g_total_blocks);
    if (fclose(sf) != 0) bad = 1;
    if (bad) {
        fprintf(stderr, "write failed for %s.nq2; "
                "nothing published\n", out_prefix);
        remove(stream_tmp);
        remove(book_tmp);
        return 1;
    }
    /* Publish the pair as one commit (publish_pair, Sol H4). */
    {
        char book_dst[1024], stream_dst[1024];
        snprintf(book_dst, sizeof(book_dst), "%s.nq2book",
                 out_prefix);
        snprintf(stream_dst, sizeof(stream_dst), "%s.nq2",
                 out_prefix);
        if (publish_pair(book_tmp, stream_tmp,
                         book_dst, stream_dst) != 0) {
            fprintf(stderr, "cannot publish %s pair\n", out_prefix);
            remove(book_tmp);
            remove(stream_tmp);
            return 1;
        }
    }

    uint64_t book_bytes = 12 + (uint64_t)entries * block * 4;
    uint64_t stream_bytes = 32 + g_total_blocks * 5;
    double rate = (double)stream_bytes * 8.0 / (double)g_total_weights;
    double crate = (double)(stream_bytes + book_bytes) * 8.0
                   / (double)g_total_weights;
    snprintf(pathbuf, sizeof(pathbuf), "%s.report.txt", out_prefix);
    FILE *rf = fopen(pathbuf, "w");
    if (!rf) {
        fprintf(stderr, "cannot write %s\n", pathbuf);
        return 1;
    }
    {
        fprintf(rf, "NanoQuatTwo quantization report\n");
        fprintf(rf, "tensors packed: %d\n", g_n_scope);
        fprintf(rf, "weights packed: %llu\n", (unsigned long long)g_total_weights);
        fprintf(rf, "block size: %u, entries: %u\n", block, entries);
        fprintf(rf, "normalization: %s\n", g_norm_rms ? "rms (no offset)" : "minmax");
        fprintf(rf, "index stream (.nq2): %llu bytes\n", (unsigned long long)stream_bytes);
        fprintf(rf, "memory file (.nq2book): %llu bytes\n", (unsigned long long)book_bytes);
        fprintf(rf, "stream rate: %.4f bits per weight (stream only)\n", rate);
        fprintf(rf, "complete rate: %.4f bits per weight (stream + book, two-part accounting)\n", crate);
        fprintf(rf, "relative L2 error on real weights: %.4f\n", rel_l2);
        if (fclose(rf) != 0) {
            fprintf(stderr, "write failed for %s\n", pathbuf);
            return 1;
        }
    }
    printf("index stream: %llu bytes; memory file: %llu bytes\n",
           (unsigned long long)stream_bytes, (unsigned long long)book_bytes);
    printf("rate: %.4f bits per weight (stream only); "
           "complete %.4f (stream + book)\n", rate, crate);
    return 0;
}
