/* nq2-infer.c ... NanoQuatTwo inference engine.
 *
 * Implemented by Elfie (E.L.F.I.E.): the engine and this code.
 * Direction and executive planning by JD.
 *
 * A pure C99 inference engine for continuous-diffusion language
 * models of the Nemotron-Labs-Sigma family, built to run on
 * NanoQuatTwo packed models (.nq2 + .nq2book) as well as on the
 * original safetensors weights. The architecture and sampling
 * procedure follow NVIDIA's published reference implementation
 * (modeling_nemotron_labs_sigma.py, NVIDIA Open Model License):
 * blockwise denoising of 16-dimensional continuous embeddings
 * with a KV-cached clean prefix, self-conditioning, optional
 * classifier-free guidance, and DDIM/DDPM transitions.
 *
 * Streaming design (JD's prefetch ruling, 2026-11-23): weights
 * flow disc -> memory -> CPU through a two-deep pipeline. The
 * forward pass consumes tensors in a fixed order; while one
 * tensor computes, the next is already materializing, and the
 * one after waits on disc. At most two weight tensors are ever
 * open at once, however large the model grows. The two big
 * matrices (token embedding, diffusion head) are never wholly
 * materialized: embeddings gather rows, the head multiplies in
 * row tiles.
 *
 * Thread-smart (JD's ruling): a worker pool sizes itself to the
 * machine's cores and partitions every matmul by rows; a
 * dedicated prefetch thread keeps the pipeline fed. Buffer sizes
 * are derived from the model's own tensor shapes at load, never
 * hardcoded.
 *
 * Scope of this first engine: token IDs in, token IDs out.
 * The text tokenizer is the next piece, on its own.
 *
 * Build: cc -O2 -std=c99 -Wall -Wextra -pthread -o nq2-infer \
 *          tools/nq2-infer.c -lm
 */

#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64 /* 64-bit off_t on Windows too */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <dirent.h> /* opendir/readdir directory scan (Archie P0-3) */

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif
#define NQ2HS_ENGINE_POOL
#include "nq2-handshake.h" /* Tensor Handshake: zero mutex */

#include <stdarg.h>
#include "nq2-dc-kernel.h"   /* the shared DTC kernel (review P1-1) */

/* ================= threads ================= */

static int cpu_count(void)
{
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (int)si.dwNumberOfProcessors;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
#endif
}

/* Physical RAM in bytes, or 0 when the platform will not say.
 * Used only to derive the default pipeline depth (prefetch
 * adaptability item, 2026-11-25): a device property, read
 * once at startup, never a results input. */
static uint64_t device_ram_bytes(void)
{
#ifdef _WIN32
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof ms;
    if (GlobalMemoryStatusEx(&ms)) return (uint64_t)ms.ullTotalPhys;
    return 0;
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long pgsz = sysconf(_SC_PAGESIZE);
    if (pages > 0 && pgsz > 0) return (uint64_t)pages * (uint64_t)pgsz;
    return 0;
#endif
}

/* The compute pool lives in nq2-handshake.h (Tensor Handshake,
 * zero mutex): pool_init / pool_run / pool_shutdown and g_pool,
 * same contract and partitions as before. */

/* ================= small math ================= */

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;

static void rng_seed(uint64_t s) { g_rng = s ? s : 0x9E3779B97F4A7C15ull; }

static uint64_t rng_next(void)
{
    uint64_t x = g_rng;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    g_rng = x;
    return x * 0x2545F4914F6CDD1Dull;
}

/* Standard normal by Box-Muller; spare cached per call chain
 * (single-threaded use: the sampler only). */
static double randn(void)
{
    static int have_spare = 0;
    static double spare = 0.0;
    if (have_spare) { have_spare = 0; return spare; }
    double u1, u2;
    do { u1 = (double)(rng_next() >> 11) * (1.0 / 9007199254740992.0); }
    while (u1 <= 0.0);
    u2 = (double)(rng_next() >> 11) * (1.0 / 9007199254740992.0);
    double r = sqrt(-2.0 * log(u1));
    double t = 2.0 * 3.14159265358979323846 * u2;
    spare = r * sin(t);
    have_spare = 1;
    return r * cos(t);
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
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while (!(mant & 0x400u)) { mant <<= 1; exp--; }
            mant &= 0x3ffu;
            bits = sign | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static float silu_f(float x) { return x / (1.0f + expf(-x)); }

/* y = x @ W^T, W row-major [rows][cols], x [n][cols], y [n][rows].
 * Rows of y are partitioned across the pool. */
typedef struct {
    const float *x;
    const float *w;
    float *y;
    int n, rows, cols;
} MatmulCtx;

static void matmul_work(void *vctx, int worker, int nw)
{
    MatmulCtx *c = (MatmulCtx *)vctx;
    int r0 = (int)((int64_t)c->rows * worker / nw);
    int r1 = (int)((int64_t)c->rows * (worker + 1) / nw);
    for (int r = r0; r < r1; r++) {
        const float *wrow = c->w + (size_t)r * c->cols;
        for (int i = 0; i < c->n; i++) {
            const float *xrow = c->x + (size_t)i * c->cols;
            double acc = 0.0;
            for (int k = 0; k < c->cols; k++) acc += (double)xrow[k] * wrow[k];
            c->y[(size_t)i * c->rows + r] = (float)acc;
        }
    }
}

static void matmul(const float *x, const float *w, float *y,
                   int n, int rows, int cols)
{
    MatmulCtx c;
    c.x = x; c.w = w; c.y = y; c.n = n; c.rows = rows; c.cols = cols;
    pool_run(matmul_work, &c);
}

/* RMSNorm over rows of x in place-ish (out may alias x). */
static void rmsnorm(float *out, const float *x, const float *w,
                    int n, int dim, float eps)
{
    for (int i = 0; i < n; i++) {
        const float *row = x + (size_t)i * dim;
        float *orow = out + (size_t)i * dim;
        double ss = 0.0;
        for (int k = 0; k < dim; k++) ss += (double)row[k] * row[k];
        double inv = 1.0 / sqrt(ss / dim + eps);
        for (int k = 0; k < dim; k++) orow[k] = (float)(row[k] * inv) * w[k];
    }
}

/* ================= weight sources =================
 *
 * Two backends behind one range-read API:
 *   raw:    the original safetensors shards (bf16/f16/f32)
 *   packed: a NanoQuatTwo .nq2 stream + .nq2book pair
 *
 * The packed format (v1/v2) carries no tensor names or shapes,
 * so the packed backend rebuilds its directory from the model
 * directory's safetensors headers, matching tensors to stream
 * segments in the same order the quantizer packed them. That
 * gap is recorded for a future format v3.
 */

enum { DT_BF16 = 0, DT_F16 = 1, DT_F32 = 2 };

typedef struct {
    char name[256];
    uint64_t count;
    int rows, cols;
    int shard;              /* raw: shard index */
    uint64_t byte_off;      /* raw: absolute offset of the data */
    int dtype;              /* raw */
    uint64_t block0;        /* packed: first stream block */
    uint64_t nblocks;       /* packed: blocks this tensor owns */
} TEnt;

#define MAX_TENSORS 1024
static TEnt g_dir[MAX_TENSORS];
static int g_ndir = 0;

static TEnt *dir_find(const char *name)
{
    for (int i = 0; i < g_ndir; i++)
        if (strcmp(g_dir[i].name, name) == 0) return &g_dir[i];
    return NULL;
}

/* ---- raw safetensors ---- */

#define MAX_SHARDS 8
static FILE *g_shards[MAX_SHARDS];
static char *g_shard_paths[MAX_SHARDS];
static int g_nshards = 0;

static char *dup_str(const char *s)
{
    size_t n = strlen(s) + 1;
    char *p = (char *)malloc(n);
    if (p) memcpy(p, s, n);
    return p;
}

static int js_string_end(const char *s, long n, long pos)
{
    if (pos >= n || s[pos] != '"') return -1;
    pos++;
    while (pos < n) {
        if (s[pos] == '\\') { pos += 2; continue; }
        if (s[pos] == '"') return (int)(pos + 1);
        pos++;
    }
    return -1;
}

static int shard_open_raw(const char *path, int shard_idx)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    /* File size, for the data_offsets span check below. */
    uint64_t fsize;
#ifdef _WIN32
    if (_fseeki64(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    { long long s = _ftelli64(f); if (s < 0) { fclose(f); return -1; } fsize = (uint64_t)s; }
    if (_fseeki64(f, 0, SEEK_SET) != 0) { fclose(f); return -1; }
#else
    if (fseeko(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    { off_t s = ftello(f); if (s < 0) { fclose(f); return -1; } fsize = (uint64_t)s; }
    if (fseeko(f, 0, SEEK_SET) != 0) { fclose(f); return -1; }
#endif
    uint8_t lenbuf[8];
    if (fread(lenbuf, 1, 8, f) != 8) { fclose(f); return -1; }
    uint64_t hlen = 0;
    for (int i = 0; i < 8; i++) hlen |= (uint64_t)lenbuf[i] << (8 * i);
    if (hlen == 0 || hlen > (1ull << 28)) { fclose(f); return -1; }
    char *hdr = (char *)malloc(hlen + 1);
    if (!hdr) { fclose(f); return -1; }
    if (fread(hdr, 1, hlen, f) != hlen) { free(hdr); fclose(f); return -1; }
    hdr[hlen] = '\0';
    uint64_t data_start = 8 + hlen;
    g_shards[shard_idx] = f;
    /* Retain the path so the pipe can open its own private
     * handle per shard at pipe_init (the fetcher must never
     * share a seek position with the main thread's reads). */
    g_shard_paths[shard_idx] = dup_str(path);

    long n = (long)hlen, pos = 0;
    while (pos < n) {
        while (pos < n && hdr[pos] != '"') pos++;
        if (pos >= n) break;
        int ke = js_string_end(hdr, n, pos);
        if (ke < 0) break;
        long klen = ke - pos - 2;
        char key[256];
        if (klen > 250) klen = 250;
        memcpy(key, hdr + pos + 1, klen);
        key[klen] = '\0';
        long p = ke;
        while (p < n && (hdr[p] == ' ' || hdr[p] == ':' || hdr[p] == '\t')) p++;
        if (p >= n || hdr[p] != '{') { pos = ke; continue; }
        const char *obj = hdr + p;
        long depth = 0, q = p;
        int instr = 0;
        while (q < n) {
            char c = hdr[q];
            if (instr) {
                if (c == '\\') { q += 2; continue; }
                if (c == '"') instr = 0;
            } else if (c == '"') instr = 1;
            else if (c == '{' || c == '[') depth++;
            else if (c == '}' || c == ']') { depth--; if (depth == 0) break; }
            q++;
        }
        long olen = q - p + 1;
        pos = q + 1;
        if (strcmp(key, "__metadata__") == 0) continue;
        if (g_ndir >= MAX_TENSORS) { free(hdr); return -1; }
        TEnt *t = &g_dir[g_ndir];
        memset(t, 0, sizeof(*t));
        memcpy(t->name, key, strlen(key) + 1);
        t->shard = shard_idx;
        const char *dt = strstr(obj, "\"dtype\"");
        if (dt && (long)(dt - obj) < olen) {
            const char *colon = strchr(dt + 7, ':');
            const char *q1 = colon ? strchr(colon, '"') : NULL;
            const char *q2 = q1 ? strchr(q1 + 1, '"') : NULL;
            char dts[8] = {0};
            if (q1 && q2 && q2 - q1 - 1 >= 1 && q2 - q1 - 1 <= 7)
                memcpy(dts, q1 + 1, q2 - q1 - 1);
            if (strcmp(dts, "BF16") == 0) t->dtype = DT_BF16;
            else if (strcmp(dts, "F16") == 0) t->dtype = DT_F16;
            else if (strcmp(dts, "F32") == 0) t->dtype = DT_F32;
            else {
                /* Unsupported dtypes are rejected, never
                 * reinterpreted as F32 (Codex Astra review
                 * M-P4, 2026-11-25; the packer does the same). */
                fprintf(stderr, "reject: tensor %s has "
                        "unsupported dtype '%s' (only BF16, "
                        "F16, F32 load)\n", key, dts);
                free(hdr);
                return -1;
            }
        }
        const char *sh = strstr(obj, "\"shape\"");
        if (sh && (long)(sh - obj) < olen) {
            const char *lb = strchr(sh, '[');
            const char *rb = lb ? strchr(lb, ']') : NULL;
            if (lb && rb) {
                char buf[128];
                long bl = rb - lb - 1;
                if (bl > 120) bl = 120;
                memcpy(buf, lb + 1, bl);
                buf[bl] = '\0';
                uint64_t dims[4];
                int nd = 0;
                char *tok = strtok(buf, ", ");
                while (tok && nd < 4) {
                    dims[nd++] = strtoull(tok, NULL, 10);
                    tok = strtok(NULL, ", ");
                }
                if (nd == 1) { t->rows = (int)dims[0]; t->cols = 1; }
                else if (nd >= 2) { t->rows = (int)dims[0]; t->cols = (int)dims[1]; }
                uint64_t cnt = 1;
                for (int i = 0; i < nd; i++) cnt *= dims[i];
                t->count = cnt;
            }
        }
        const char *df = strstr(obj, "\"data_offsets\"");
        if (df && (long)(df - obj) < olen) {
            const char *lb = strchr(df, '[');
            if (lb) {
                uint64_t begin = strtoull(lb + 1, NULL, 10);
                const char *comma = strchr(lb, ',');
                uint64_t end = comma ? strtoull(comma + 1, NULL, 10) : 0;
                uint64_t esz = t->dtype == DT_F32 ? 4 : 2;
                /* Span validation (Archie P0-4): only begin was
                 * read before, unchecked. The span must be
                 * ordered, lie inside the shard file, and (when
                 * both shape and dtype were parsed) match the
                 * shape-derived element count exactly. */
                if (end < begin || end > fsize
                    || data_start + end > fsize
                    || (dt && (long)(dt - obj) < olen && t->count > 0
                        && end - begin != t->count * esz)) {
                    fprintf(stderr, "reject: tensor %s has "
                            "data_offsets [%llu, %llu) inconsistent "
                            "with the shard or its shape\n", key,
                            (unsigned long long)begin,
                            (unsigned long long)end);
                    free(hdr);
                    return -1;
                }
                t->byte_off = data_start + begin;
            }
        }
        g_ndir++;
    }
    free(hdr);
    return 0;
}

static int seek_to(FILE *f, uint64_t off)
{
#ifdef _WIN32
    return _fseeki64(f, (long long)off, SEEK_SET);
#else
    return fseeko(f, (off_t)off, SEEK_SET);
#endif
}

/* 64-bit position/size reads, the same discipline as seek_to.
 * A plain long ftell narrows a stream above 2 GiB on Windows
 * (Archie Round 3 P2-2, completing Codex M-P7, 2026-11-25). */
static off_t tell_of(FILE *f)
{
#ifdef _WIN32
    return (off_t)_ftelli64(f);
#else
    return ftello(f);
#endif
}

static int seek_end(FILE *f)
{
#ifdef _WIN32
    return _fseeki64(f, 0, SEEK_END);
#else
    return fseeko(f, 0, SEEK_END);
#endif
}

/* Convert raw source bytes of a tensor to fp32. This is the
 * one conversion every raw read path uses (direct reads and
 * the pipe's staged reads alike), so the two can never drift
 * apart arithmetically. */
static void raw_convert(const TEnt *t, const uint8_t *raw,
                        uint64_t nelems, float *out)
{
    if (t->dtype == DT_BF16) {
        const uint16_t *h = (const uint16_t *)raw;
        for (uint64_t i = 0; i < nelems; i++) out[i] = bf16_to_float(h[i]);
    } else if (t->dtype == DT_F16) {
        const uint16_t *h = (const uint16_t *)raw;
        for (uint64_t i = 0; i < nelems; i++) out[i] = f16_to_float(h[i]);
    } else {
        memcpy(out, raw, nelems * 4);
    }
}

/* Read elements [elem0, elem0+nelems) of a raw tensor as fp32. */
static int raw_read_range(const TEnt *t, uint64_t elem0, uint64_t nelems,
                          float *out)
{
    /* Tensor-local bound (Sol H5): a range read may never
     * leave the tensor it names. Without it, a short or
     * misbound tensor's read runs into its neighbour's bytes. */
    if (elem0 > t->count || nelems > t->count - elem0) return -1;
    int esz = t->dtype == DT_F32 ? 4 : 2;
    uint8_t *raw = (uint8_t *)malloc(nelems * (uint64_t)esz);
    if (!raw) return -1;
    FILE *f = g_shards[t->shard];
    if (seek_to(f, t->byte_off + elem0 * (uint64_t)esz) != 0
            || fread(raw, esz, nelems, f) != nelems) {
        free(raw);
        return -1;
    }
    raw_convert(t, raw, nelems, out);
    free(raw);
    return 0;
}

/* ---- packed .nq2 ---- */

static FILE *g_pk_stream = NULL;
static float *g_pk_book = NULL;         /* all books, fp32 */
static uint64_t g_pk_book_floats = 0;   /* floats in g_pk_book */
static uint32_t g_pk_block = 0, g_pk_entries = 0, g_pk_version = 0;
/* Block-level prefetch distance for the expansion loops
 * below (runtime tunable, default 2; configured with the
 * rest of the pipe geometry further down). */
static int g_prefetch_dist = 2; /* --prefetch-dist N (>= 0) */
static uint64_t g_pk_header = 0;        /* stream header bytes */
static uint64_t g_pk_nblocks = 0;

/* Format v3 identity and layouts (H3b, 2026-11-25): the
 * generation id both v3 headers carry, the v3 flags, and the
 * per-tensor tables the v3 layouts need. Legacy packs (v1/v2)
 * leave all of these at their defaults and read exactly as
 * before, with a notice at open. */
static uint64_t g_pk_gen = 0;
static uint32_t g_pk_flags = 0, g_pk_pq_d = 0, g_pk_idxw = 1;
static uint32_t *g_pk_tblock = NULL;    /* per-tensor block sizes */
static uint64_t *g_pk_book_off = NULL;  /* per-tensor book offset */
static uint16_t *g_pk_rawdata = NULL;   /* fp16-raw section */
static uint64_t *g_pk_raw_off = NULL;   /* per-tensor raw offset */

static uint32_t pk_block_of(const TEnt *t)
{
    return g_pk_tblock ? g_pk_tblock[t - g_dir] : g_pk_block;
}
static const float *pk_book_of(const TEnt *t)
{
    return g_pk_book + g_pk_book_off[t - g_dir];
}
static int pk_is_raw(const TEnt *t)
{
    return g_pk_raw_off && g_pk_raw_off[t - g_dir] != ~(uint64_t)0;
}
#define PK_FNV_SEED 14695981039346656037ull
static uint64_t pk_fnv(const void *data, size_t len, uint64_t h)
{
    const uint8_t *p = (const uint8_t *)data;
    for (size_t i = 0; i < len; i++) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ull;
    }
    return h;
}
static uint64_t pk_fnv_stream(uint64_t off, uint64_t len, uint64_t h)
{
    uint8_t buf[65536];
    if (seek_to(g_pk_stream, off) != 0) return 0;
    while (len) {
        size_t want = len > sizeof(buf) ? sizeof(buf) : (size_t)len;
        if (fread(buf, 1, want, g_pk_stream) != want) return 0;
        h = pk_fnv(buf, want, h);
        len -= want;
    }
    return h;
}

/* Verify the per-tensor digest manifest beside a v3 pack:
 * the generation line must match the headers, the tensor
 * lines must describe this directory in order, and every
 * segment digest (over the stored stream bytes) and book
 * digest (over the stored book floats) must recompute to
 * the recorded value. A missing manifest is a notice, not a
 * failure: the generation pairing still binds the pair. */
static int pk_verify_manifest(const char *stream_path)
{
    char mpath[1200];
    snprintf(mpath, sizeof(mpath), "%s", stream_path);
    size_t L = strlen(mpath);
    if (L > 4 && strcmp(mpath + L - 4, ".nq2") == 0)
        strcpy(mpath + L - 4, ".nq2manifest");
    else
        strncat(mpath, ".nq2manifest", sizeof(mpath) - L - 1);
    FILE *mf = fopen(mpath, "r");
    if (!mf) {
        fprintf(stderr, "notice: no manifest beside this pack; "
                "generation pairing checked, per-tensor digests "
                "not verified\n");
        return 0;
    }
    char line[1200];
    unsigned long long mgen = 0;
    int nt = -1, bad = 1;
    if (fgets(line, sizeof(line), mf)
            && strcmp(line, "NQ2MANIFEST 1\n") == 0
            && fgets(line, sizeof(line), mf)
            && sscanf(line, "generation %llx", &mgen) == 1
            && fgets(line, sizeof(line), mf)
            && sscanf(line, "tensors %d", &nt) == 1)
        bad = 0;
    if (bad || mgen != g_pk_gen || nt != g_ndir) {
        fprintf(stderr, "pack identity mismatch: the manifest "
                "beside this pack does not describe it "
                "(manifest generation %016llx, pack generation "
                "%016llx); refusing to run\n",
                mgen, (unsigned long long)g_pk_gen);
        fclose(mf);
        return -1;
    }
    for (int i = 0; i < g_ndir; i++) {
        char name[256], kind[16];
        unsigned long long b0, nb, segd, bookd;
        unsigned blk;
        if (!fgets(line, sizeof(line), mf)
                || sscanf(line, "tensor %255s %llu %llu %u %llx "
                          "%llx %15s", name, &b0, &nb, &blk,
                          &segd, &bookd, kind) != 7
                || strcmp(name, g_dir[i].name) != 0
                || b0 != g_dir[i].block0 || nb != g_dir[i].nblocks
                || blk != pk_block_of(&g_dir[i])) {
            fprintf(stderr, "pack identity mismatch: manifest "
                    "line %d does not describe tensor %s; "
                    "refusing to run\n", i, g_dir[i].name);
            fclose(mf);
            return -1;
        }
        uint64_t h = PK_FNV_SEED;
        if (pk_is_raw(&g_dir[i])) {
            h = pk_fnv(g_pk_rawdata + g_pk_raw_off[i],
                       g_dir[i].count * 2, h);
        } else {
            h = pk_fnv_stream(g_pk_header
                    + g_dir[i].block0 * g_pk_idxw,
                    g_dir[i].nblocks * g_pk_idxw, h);
            h = pk_fnv_stream(g_pk_header
                    + g_pk_nblocks * g_pk_idxw
                    + 2 * g_dir[i].block0,
                    2 * g_dir[i].nblocks, h);
            h = pk_fnv_stream(g_pk_header
                    + g_pk_nblocks * (g_pk_idxw + 2)
                    + 2 * g_dir[i].block0,
                    2 * g_dir[i].nblocks, h);
        }
        if (h != segd) {
            fprintf(stderr, "pack identity mismatch: tensor %s "
                    "fails its manifest segment digest; the "
                    "pack has been altered or mispaired\n", name);
            fclose(mf);
            return -1;
        }
        uint64_t bh = PK_FNV_SEED;
        if (!pk_is_raw(&g_dir[i]))
            bh = pk_fnv(pk_book_of(&g_dir[i]),
                        (size_t)g_pk_entries * pk_block_of(&g_dir[i])
                            * sizeof(float), bh);
        if (bh != bookd) {
            fprintf(stderr, "pack identity mismatch: tensor %s "
                    "fails its manifest book digest; the pack "
                    "has been altered or mispaired\n", name);
            fclose(mf);
            return -1;
        }
    }
    fclose(mf);
    printf("pack identity: generation %016llx; manifest "
           "verified for %d tensors\n",
           (unsigned long long)g_pk_gen, g_ndir);
    return 0;
}

static uint32_t pk_rd_u32(FILE *f)
{
    uint8_t b[4];
    if (fread(b, 1, 4, f) != 4) { fprintf(stderr, "short read\n"); exit(1); }
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8)
         | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

static uint64_t pk_rd_u64(FILE *f)
{
    uint64_t lo = pk_rd_u32(f), hi = pk_rd_u32(f);
    return lo | (hi << 32);
}

static float pk_half(const uint8_t *p)
{
    uint16_t h = (uint16_t)(p[0] | (p[1] << 8));
    return f16_to_float(h);
}

/* Expand packed elements [elem0, elem0+nelems) of one tensor. */
static int packed_read_range(const TEnt *t, uint64_t elem0, uint64_t nelems,
                             float *out)
{
    /* Audit nit: a zero-length range would underflow
     * (elem0 + nelems - 1) below. No caller passes one. */
    if (nelems == 0) return -1;
    uint32_t block = pk_block_of(t);
    uint64_t b0 = t->block0 + elem0 / block;
    uint64_t b1 = t->block0 + (elem0 + nelems - 1) / block;
    uint64_t nb = b1 - b0 + 1;
    uint8_t *idx = (uint8_t *)malloc(nb * g_pk_idxw);
    uint8_t *fr = (uint8_t *)malloc(nb * 4);
    float *tmp = (float *)malloc(block * sizeof(float));
    if (!idx || !fr || !tmp) { free(idx); free(fr); free(tmp); return -1; }
    if (seek_to(g_pk_stream, g_pk_header + b0 * g_pk_idxw) != 0
            || fread(idx, g_pk_idxw, nb, g_pk_stream) != nb) goto fail;
    if (seek_to(g_pk_stream, g_pk_header + g_pk_nblocks * g_pk_idxw
                + 2 * b0) != 0
            || fread(fr, 2, nb, g_pk_stream) != nb) goto fail;
    if (seek_to(g_pk_stream, g_pk_header
                + g_pk_nblocks * (g_pk_idxw + 2) + 2 * b0) != 0
            || fread(fr + nb * 2, 2, nb, g_pk_stream) != nb) goto fail;
    {
        const float *book = pk_book_of(t);
        for (uint64_t b = 0; b < nb; b++) {
            if (idx[b] >= g_pk_entries) goto fail;
            /* Book lookup optimization (streamtest variant 1,
             * measured +21 percent in the harness): pull the
             * prototype row g_prefetch_dist blocks ahead toward
             * cache while the current row streams out, so the
             * lookup never waits on memory. The distance is a
             * runtime tunable (prefetch adaptability, 2026-11-25;
             * default 2, the original hard-coded value). It is a
             * cache hint only and cannot change results. */
            if (b + (uint64_t)g_prefetch_dist < nb) {
#if defined(__GNUC__) || defined(__clang__)
                __builtin_prefetch(
                    book + (size_t)idx[b + (uint64_t)g_prefetch_dist]
                               * block,
                    0, 1);
#endif
            }
            float mn = pk_half(fr + b * 2);
            float sc = pk_half(fr + nb * 2 + b * 2);
            const float *proto = book + (size_t)idx[b] * block;
            uint64_t g0 = (b0 + b) * block - t->block0 * block;
            uint64_t lo = elem0 > g0 ? elem0 : g0;
            uint64_t hi = elem0 + nelems < g0 + block ? elem0 + nelems
                                                      : g0 + block;
            if (lo == g0 && hi == g0 + block) {
                /* Block-aligned and fully wanted (the streaming
                 * hot path: whole tensors for the pipe): expand
                 * straight into the destination, no bounce copy. */
                const float *restrict src = proto;
                float *restrict dst = out + (g0 - elem0);
                for (uint32_t i = 0; i < block; i++)
                    dst[i] = mn + sc * src[i];
            } else {
                for (uint32_t i = 0; i < block; i++)
                    tmp[i] = mn + sc * proto[i];
                for (uint64_t e = lo; e < hi; e++)
                    out[e - elem0] = tmp[e - g0];
            }
        }
    }
    free(idx); free(fr); free(tmp);
    return 0;
fail:
    free(idx); free(fr); free(tmp);
    return -1;
}

/* ---- Triforce split: fetch and expand halves of the pipe
 * read (JD's three-role pipeline, 2026-11-24). The fetch half
 * performs only disc reads, into caller-owned staging: the
 * three packed section reads, and (since the mode-2 staging
 * buffer landed, Archie P1-1) the raw source bytes of any
 * raw-served tensor or tensor tail. The expand half is
 * reconstruction only, with its arithmetic unchanged. The
 * fetch half takes its own FILE* handles (one for the packed
 * stream, one per shard) so pipe fetches never interleave
 * seek/read pairs with the main thread's head and embed reads
 * on the shared handles. ---- */
typedef struct {
    uint8_t *idx;      /* nb bytes */
    uint8_t *fr;       /* 4*nb bytes: minima, then scales */
    uint64_t b0, nb;
    /* Raw staging (Archie P1-1, completed 2026-11-24 night):
     * the source bytes of the raw-served part of the tensor,
     * read at fetch time on the fetcher's private shard handle.
     * raw_elem0 is the first element the staged bytes belong
     * to (nonzero only when a packed prefix is also staged in
     * idx/fr); raw_nelems is their element count. */
    uint8_t *raw;
    uint64_t raw_elem0, raw_nelems;
    /* 0 empty, 1 packed sections staged, 2 raw bytes staged
     * (packed prefix too when raw_elem0 > 0), 3 legacy fused:
     * expand reads through the shared handles, used only when
     * a private handle could not be opened. */
    int mode;
    double fetch_ms;
} Stage;

#ifndef _WIN32
/* The split fetch/expand halves serve only the POSIX triforce
 * workers; the Windows build keeps the fused single worker
 * (pipe_load) and never calls them, so they are guarded out
 * there instead of warning as unused (Archie Round 2 P2-B). */
static int packed_fetch_range(FILE *stream, const TEnt *t,
                              uint64_t elem0, uint64_t nelems, Stage *s)
{
    uint32_t block = pk_block_of(t);
    uint64_t b0 = t->block0 + elem0 / block;
    uint64_t b1 = t->block0 + (elem0 + nelems - 1) / block;
    uint64_t nb = b1 - b0 + 1;
    if (seek_to(stream, g_pk_header + b0 * g_pk_idxw) != 0
            || fread(s->idx, g_pk_idxw, nb, stream) != nb) return -1;
    if (seek_to(stream, g_pk_header + g_pk_nblocks * g_pk_idxw
                + 2 * b0) != 0
            || fread(s->fr, 2, nb, stream) != nb) return -1;
    if (seek_to(stream, g_pk_header
                + g_pk_nblocks * (g_pk_idxw + 2) + 2 * b0) != 0
            || fread(s->fr + nb * 2, 2, nb, stream) != nb) return -1;
    s->b0 = b0;
    s->nb = nb;
    s->mode = 1;
    return 0;
}

static int packed_expand_range(const TEnt *t, uint64_t elem0,
                               uint64_t nelems, const Stage *s,
                               float *out)
{
    uint32_t block = pk_block_of(t);
    uint64_t b0 = s->b0, nb = s->nb;
    const uint8_t *idx = s->idx, *fr = s->fr;
    float *tmp = (float *)malloc(block * sizeof(float));
    if (!tmp) return -1;
    {
        const float *book = pk_book_of(t);
        for (uint64_t b = 0; b < nb; b++) {
            if (idx[b] >= g_pk_entries) { free(tmp); return -1; }
            if (b + (uint64_t)g_prefetch_dist < nb) {
#if defined(__GNUC__) || defined(__clang__)
                __builtin_prefetch(
                    book + (size_t)idx[b + (uint64_t)g_prefetch_dist]
                               * block,
                    0, 1);
#endif
            }
            float mn = pk_half(fr + b * 2);
            float sc = pk_half(fr + nb * 2 + b * 2);
            const float *proto = book + (size_t)idx[b] * block;
            uint64_t g0 = (b0 + b) * block - t->block0 * block;
            uint64_t lo = elem0 > g0 ? elem0 : g0;
            uint64_t hi = elem0 + nelems < g0 + block ? elem0 + nelems
                                                      : g0 + block;
            if (lo == g0 && hi == g0 + block) {
                const float *restrict src = proto;
                float *restrict dst = out + (g0 - elem0);
                for (uint32_t i = 0; i < block; i++)
                    dst[i] = mn + sc * src[i];
            } else {
                for (uint32_t i = 0; i < block; i++)
                    tmp[i] = mn + sc * proto[i];
                for (uint64_t e = lo; e < hi; e++)
                    out[e - elem0] = tmp[e - g0];
            }
        }
    }
    free(tmp);
    return 0;
}
#endif /* !_WIN32 */

static const char *g_pk_stream_path = NULL;

static int g_use_packed = 0;
static int g_forensics = 0;   /* lab build: --forensics */
/* Runtime pipeline geometry (JD, 2026-11-24: step it up at
 * runtime, never hard-coded): how many tensors the pipe may
 * hold in flight, and how far the fetcher may run ahead of the
 * unpack demand. Defaults reproduce the original two-slot,
 * one-ahead pipe exactly.
 *
 * Prefetch adaptability (2026-11-25): every knob has three
 * configuration layers, strongest last... the built-in or
 * device-derived default, the environment (NQ2_PIPE_SLOTS,
 * NQ2_FETCH_AHEAD, NQ2_PREFETCH_DIST), and the command line
 * (--pipe-slots, --fetch-ahead, --prefetch-dist), which always
 * wins. The *_src strings record which layer each value came
 * from so the startup line can say so. The block-level
 * prefetch distance inside the expansion loops is the third
 * knob: how far ahead of the block being expanded the book
 * prototype row is pulled toward cache. Depth in all three
 * knobs is a performance property only; consumption order and
 * results are invariant under them (proven in the fix log's
 * regression matrix). */
static int g_pipe_slots = 2;    /* --pipe-slots N (>= 2) */
static int g_fetch_ahead = 1;   /* --fetch-ahead N (>= 1) */
/* (g_prefetch_dist is defined with the pack globals above:
 * the expansion loops sit earlier in the file than this
 * configuration block.) */
static const char *g_slots_src = "default";
static const char *g_fetch_src = "default";
static const char *g_dist_src = "default";
static int g_slots_explicit = 0; /* cli/env: no device derivation */
/* Expansion width (Tensor Handshake stage 2, JD 2026-11-25):
 * how many layers may be expanding at the same time. A knob
 * of its own, distinct from pipe slots: slots are staged
 * pipeline depth, lanes are simultaneous expansions. Same
 * three-tier family (device-derived default, NQ2_EXPAND_LANES,
 * --expand-lanes), strict parsing, range 1..64, and a loud
 * clamp when the requested width cannot fit the device. The
 * ladder {2, 4, 8, 16, 32, 64} is the residency dial (JD,
 * 2026-11-25, after the kimi-k3-in-c memory-for-speed ladder):
 * more memory buys more layers in flight, exactly as far as
 * the hardware ceiling allows, and the guardrails are what
 * make the dial safe to turn. */
static int g_expand_lanes = 2;  /* --expand-lanes N (1..64) */
static const char *g_lanes_src = "default";
static int g_lanes_explicit = 0; /* cli/env: no device derivation */
static const char *g_logits_dump = NULL;  /* lab: --logits-dump */
static int g_logits_dumped = 0;
/* Round 4 P1: --logits-dump-all appends one LOG1 block per
 * denoise evaluation instead of only the first, so the
 * comparator can watch reconstruction error amplify or settle
 * across the diffusion trajectory. */
static int g_logits_dump_all = 0;
static int g_logits_dump_count = 0;

/* ================= Precision Gate =================
 *
 * DTC-PRECISION-GATE-PLAN-2026-11-25: Dense Tensor Compute is
 * the engine's default compute path (JD's ruling, 2026-11-25);
 * expansion is the exception, spent where the table path's
 * float reassociation could change an outcome. Each pipe
 * ordinal carries a mark on the Tensor Handshake priority
 * list (MARK_DTC or MARK_EXPAND, decided by the consumer's
 * scorer ahead of the claim window); the gate arbitrates only
 * association noise between two computations of the same
 * quantized model, never the pack's quantization error.
 *
 * Postures: evidence (expanded always; the historical engine,
 * bit for bit), think-harder (expanded always, at the user's
 * request), fast (DTC wherever eligible), auto (DTC default
 * with predictive escalation: per-layer noise profile, drift
 * budget with shadow resync, margin gate at the head on final
 * steps). The gate machinery is POSIX-only, like the triforce
 * split it rides on; the Windows fused worker always expands
 * and says so in its startup note. */
enum { POSTURE_EVIDENCE = 0, POSTURE_AUTO = 1, POSTURE_FAST = 2,
       POSTURE_THINK = 3 };
static int g_posture = POSTURE_EVIDENCE;
static const char *g_posture_src = "default";
static int g_gate_lookahead = 2;        /* layers, 1..4 */
static const char *g_gate_lookahead_src = "default";
static int g_gate_resync = 32;          /* evaluations; 0 = off */
static const char *g_gate_resync_src = "default";
static double g_gate_eps_override = -1.0;    /* < 0: calibrated */
static double g_gate_budget_override = -1.0; /* < 0: calibrated */
static const char *g_gate_log_path = NULL;   /* --gate-log PATH */

/* Calibration for the open pack (plan section 5), loaded by
 * gate_calib_load after packed_open: a per-layer divergence
 * profile (delta: the layer's own increment; cum: the running
 * total), the tie epsilon and the drift budget. A pack without
 * calibration is never gated: Auto falls back to expanded. */
static int g_gate_calibrated = 0;
static double *g_gate_delta = NULL;     /* [M_L] */
static double *g_gate_cum = NULL;       /* [M_L] */
static double g_gate_delta_median = 0.0;
static double g_gate_eps = 0.0, g_gate_budget = 0.0;

/* Gate runtime state. Written only by the consumer thread, so
 * a run's decisions are a pure function of computed values and
 * the calibrated thresholds (plan section 7). */
static int g_gate_final_step = 0;   /* main: this denoise commits tokens */
static int g_gate_force_expand = 0; /* escalation in force, this call */
static double g_gate_drift2 = 0.0;  /* accumulated squared divergence */
static double g_gate_scale = 1.0;   /* profile scale, resync-anchored */
static int g_gate_finalized_upto = -1;  /* per pass, scorer horizon */
static long g_gate_evals = 0, g_gate_escalations = 0,
            g_gate_resyncs = 0;
static long g_gate_layers_dtc = 0, g_gate_layers_total = 0;
static int g_gate_since_resync = 0;

static double gate_eps_eff(void)
{
    return g_gate_eps_override >= 0.0 ? g_gate_eps_override
                                      : g_gate_eps;
}
static double gate_budget_eff(void)
{
    return g_gate_budget_override >= 0.0 ? g_gate_budget_override
                                         : g_gate_budget;
}
static const char *posture_name(int p)
{
    switch (p) {
    case POSTURE_AUTO: return "auto";
    case POSTURE_FAST: return "fast";
    case POSTURE_THINK: return "think-harder";
    default: return "evidence";
    }
}

/* The gate's debug channel (diagnostic doctrine: decisions are
 * logged per step in this build; a release build strips it). */
static void gate_logf(const char *fmt, ...)
{
    if (!g_gate_log_path) return;
    FILE *lf = fopen(g_gate_log_path, "a");
    if (!lf) return;
    va_list ap;
    va_start(ap, fmt);
    vfprintf(lf, fmt, ap);
    va_end(ap);
    fclose(lf);
}

static int src_read_range(const TEnt *t, uint64_t elem0, uint64_t nelems,
                          float *out)
{
    /* The shared range-read boundary enforces the tensor-local
     * bound for the packed path too (Sol H5); the raw path
     * re-checks it in raw_read_range. */
    if (elem0 > t->count || nelems > t->count - elem0) return -1;
    if (!g_use_packed) return raw_read_range(t, elem0, nelems, out);
    if (pk_is_raw(t)) {
        /* fp16-raw section (format v3): this tensor was
         * stored unquantized; serve it from the section the
         * pack carries, not from the shards. */
        const uint16_t *sec = g_pk_rawdata
            + g_pk_raw_off[t - g_dir] + elem0;
        for (uint64_t i = 0; i < nelems; i++)
            out[i] = f16_to_float(sec[i]);
        return 0;
    }
    /* The stream holds only whole blocks: elements past the last
     * packed block (tensors smaller than one block, and block
     * tails) were never packed. Serve the packed portion from
     * the stream and the remainder from the raw shards, which
     * are always open. Without this, a small tensor's read would
     * silently return the next tensor's first block. */
    uint64_t packed_elems = t->nblocks * pk_block_of(t);
    if (elem0 >= packed_elems)
        return raw_read_range(t, elem0, nelems, out);
    uint64_t in = nelems;
    if (elem0 + in > packed_elems) in = packed_elems - elem0;
    if (packed_read_range(t, elem0, in, out) != 0) return -1;
    if (in < nelems)
        return raw_read_range(t, elem0 + in, nelems - in, out + in);
    return 0;
}

/* Open the packed pair and assign stream segments to directory
 * tensors in directory order (the quantizer's packing order). */
static int packed_open(const char *stream_path, const char *book_path)
{
    g_pk_stream_path = stream_path;
    g_pk_stream = fopen(stream_path, "rb");
    if (!g_pk_stream) return -1;
    /* The packed reader makes three section reads per range;
     * a large stdio buffer keeps those streaming instead of
     * syscall-bound. */
    setvbuf(g_pk_stream, NULL, _IOFBF, 1 << 18);
    char magic[4];
    if (fread(magic, 1, 4, g_pk_stream) != 4
            || memcmp(magic, "NQ2\0", 4) != 0) return -1;
    g_pk_version = pk_rd_u32(g_pk_stream);
    g_pk_block = pk_rd_u32(g_pk_stream);
    g_pk_entries = pk_rd_u32(g_pk_stream);
    (void)pk_rd_u64(g_pk_stream);           /* total weights */
    g_pk_nblocks = pk_rd_u64(g_pk_stream);
    /* T2 hardening (CODE-REVIEW-2026-11-24): validate the pack
     * geometry before any division or allocation trusts it. */
    if (g_pk_block == 0 || g_pk_block > 4096 || g_pk_entries == 0
        || g_pk_entries > 256) {
        fprintf(stderr, "packed stream header has unsupported "
                "geometry (block %u, entries %u)\n",
                g_pk_block, g_pk_entries);
        return -1;
    }
    uint32_t n_books = 1;
    uint64_t *seg = NULL;
    uint32_t *raw_list = NULL, n_raw = 0;
    uint64_t raw_bytes = 0;
    if (g_pk_version == 2) {
        n_books = pk_rd_u32(g_pk_stream);
        seg = (uint64_t *)malloc(sizeof(uint64_t) * n_books);
        if (!seg) return -1;
        for (uint32_t i = 0; i < n_books; i++) seg[i] = pk_rd_u64(g_pk_stream);
    } else if (g_pk_version == 3) {
        uint32_t n_seg = pk_rd_u32(g_pk_stream);
        if ((int)n_seg != g_ndir) {
            fprintf(stderr, "packed model has %u segments but "
                    "the directory has %d tensors\n",
                    n_seg, g_ndir);
            return -1;
        }
        seg = (uint64_t *)malloc(sizeof(uint64_t) * n_seg);
        if (!seg) return -1;
        for (uint32_t i = 0; i < n_seg; i++)
            seg[i] = pk_rd_u64(g_pk_stream);
        g_pk_gen = pk_rd_u64(g_pk_stream);
        g_pk_flags = pk_rd_u32(g_pk_stream);
        if (g_pk_flags & 1u) {
            fprintf(stderr, "PQ packs (format v3, sub-books) are "
                    "expanded by nq2-infer and nq2-infer-lab in "
                    "this pass; the triforce pipe does not read "
                    "them yet\n");
            return -1;
        }
        if (g_pk_flags & 1u) {
            g_pk_pq_d = pk_rd_u32(g_pk_stream);
            if (g_pk_pq_d == 0 || g_pk_block % g_pk_pq_d != 0) {
                fprintf(stderr, "packed stream header has an "
                        "invalid PQ subvector dim %u for block "
                        "%u\n", g_pk_pq_d, g_pk_block);
                return -1;
            }
            g_pk_idxw = g_pk_block / g_pk_pq_d;
        }
        if (g_pk_flags & 2u) {
            g_pk_tblock = (uint32_t *)malloc(sizeof(uint32_t) * n_seg);
            if (!g_pk_tblock) return -1;
            for (uint32_t i = 0; i < n_seg; i++) {
                g_pk_tblock[i] = pk_rd_u32(g_pk_stream);
                if (g_pk_tblock[i] == 0 || g_pk_tblock[i] > 4096) {
                    fprintf(stderr, "packed stream header has "
                            "an unsupported per-tensor block %u\n",
                            g_pk_tblock[i]);
                    return -1;
                }
            }
        }
        if (g_pk_flags & 4u) {
            n_raw = pk_rd_u32(g_pk_stream);
            raw_list = (uint32_t *)malloc(sizeof(uint32_t)
                                          * (n_raw ? n_raw : 1));
            if (!raw_list) return -1;
            for (uint32_t i = 0; i < n_raw; i++) {
                raw_list[i] = pk_rd_u32(g_pk_stream);
                if (raw_list[i] >= n_seg) {
                    fprintf(stderr, "packed stream header names "
                            "raw segment %u of %u\n",
                            raw_list[i], n_seg);
                    return -1;
                }
                raw_bytes += g_dir[raw_list[i]].count * 2;
            }
        }
        n_books = n_seg;
    } else if (g_pk_version != 1) {
        return -1;
    }
    {
        off_t hb = tell_of(g_pk_stream);
        g_pk_header = hb > 0 ? (uint64_t)hb : 0;
    }
    /* Read-integrity (Round 4 P0 gap A, generalized): the
     * header's block count must account for the stream's
     * actual bytes exactly: header + index bytes + 2
     * frame-minimum bytes + 2 frame-scale bytes per block,
     * plus the fp16-raw section when present. A truncated or
     * overstated stream is refused here, not discovered
     * mid-read. */
    {
        if (g_pk_nblocks > (UINT64_MAX - g_pk_header) / (g_pk_idxw + 4))
            return -1;      /* header arithmetic would overflow */
        off_t save = tell_of(g_pk_stream);
        if (save < 0 || seek_end(g_pk_stream) != 0)
            return -1;
        off_t fsz = tell_of(g_pk_stream);
        if (fsz < 0 || seek_to(g_pk_stream, (uint64_t)save) != 0)
            return -1;
        uint64_t want = g_pk_header
            + (uint64_t)(g_pk_idxw + 4) * g_pk_nblocks + raw_bytes;
        if ((uint64_t)fsz != want) {
            fprintf(stderr, "packed stream size mismatch: file "
                    "has %lld bytes, header describes %llu\n",
                    (long long)fsz, (unsigned long long)want);
            return -1;
        }
    }
    FILE *bf = fopen(book_path, "rb");
    if (!bf) return -1;
    if (fread(magic, 1, 4, bf) != 4 || memcmp(magic, "NQ2B", 4) != 0)
        return -1;
    uint64_t total_book_floats;
    uint32_t stored_books = 1;
    if (g_pk_version == 3) {
        uint32_t bver = pk_rd_u32(bf);
        uint32_t b_block = pk_rd_u32(bf);
        uint32_t b_entries = pk_rd_u32(bf);
        uint32_t b_books = pk_rd_u32(bf);
        uint64_t b_gen = pk_rd_u64(bf);
        if (g_pk_flags & 1u) {
            uint32_t b_pq = pk_rd_u32(bf);
            if (b_pq != g_pk_pq_d) {
                fprintf(stderr, "stream and book disagree on "
                        "the PQ subvector dim\n");
                return -1;
            }
        }
        if (bver != 3 || b_block != g_pk_block
                || b_entries != g_pk_entries) {
            fprintf(stderr, "stream and book disagree\n");
            return -1;
        }
        if (b_gen != g_pk_gen) {
            fprintf(stderr, "pack identity mismatch: the "
                    "stream carries generation %016llx and the "
                    "book carries generation %016llx; they were "
                    "not packed together and will not be run "
                    "together\n",
                    (unsigned long long)g_pk_gen,
                    (unsigned long long)b_gen);
            return -1;
        }
        stored_books = b_books;
        if (stored_books == 1) {
            total_book_floats = (uint64_t)g_pk_entries * g_pk_block;
        } else {
            total_book_floats = 0;
            for (int i = 0; i < g_ndir; i++) {
                int is_raw = 0;
                for (uint32_t r = 0; r < n_raw; r++)
                    if (raw_list[r] == (uint32_t)i) is_raw = 1;
                if (!is_raw)
                    total_book_floats += (uint64_t)g_pk_entries
                        * (g_pk_tblock ? g_pk_tblock[i] : g_pk_block);
            }
        }
    } else if (g_pk_version == 2) {
        uint32_t bver = pk_rd_u32(bf);
        uint32_t b_block = pk_rd_u32(bf);
        uint32_t b_entries = pk_rd_u32(bf);
        uint32_t b_books = pk_rd_u32(bf);
        if (bver != 2 || b_block != g_pk_block
                || b_entries != g_pk_entries || b_books != n_books) {
            fprintf(stderr, "stream and book disagree\n");
            return -1;
        }
        stored_books = n_books;
        total_book_floats = (uint64_t)n_books * g_pk_entries * g_pk_block;
    } else {
        uint32_t b_block = pk_rd_u32(bf);
        uint32_t b_entries = pk_rd_u32(bf);
        if (b_block != g_pk_block || b_entries != g_pk_entries) {
            fprintf(stderr, "stream and book disagree\n");
            return -1;
        }
        total_book_floats = (uint64_t)g_pk_entries * g_pk_block;
    }
    g_pk_book = (float *)malloc(total_book_floats * sizeof(float));
    if (!g_pk_book) return -1;
    if (fread(g_pk_book, sizeof(float), total_book_floats, bf)
            != total_book_floats) return -1;
    fclose(bf);
    g_pk_book_floats = total_book_floats;

    /* Bind segments to directory tensors in directory order
     * (the quantizer's packing order), assign each tensor
     * its book offset, and account for the fp16-raw section. */
    g_pk_book_off = (uint64_t *)calloc((size_t)g_ndir, sizeof(uint64_t));
    g_pk_raw_off = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)g_ndir);
    if (!g_pk_book_off || !g_pk_raw_off) return -1;
    for (int i = 0; i < g_ndir; i++) g_pk_raw_off[i] = ~(uint64_t)0;
    {
        uint64_t raw_cursor = 0;
        for (uint32_t r = 0; r < n_raw; r++) {
            g_pk_raw_off[raw_list[r]] = raw_cursor;
            raw_cursor += g_dir[raw_list[r]].count;
        }
    }
    uint64_t cursor = 0, book_cursor = 0;
    uint32_t books_seen = 0;
    for (int i = 0; i < g_ndir; i++) {
        uint32_t bi = g_pk_tblock ? g_pk_tblock[i] : g_pk_block;
        if (g_pk_raw_off[i] != ~(uint64_t)0) {
            if (seg && seg[i] != 0) {
                fprintf(stderr, "packed segment %d is marked "
                        "raw but holds %llu blocks\n", i,
                        (unsigned long long)seg[i]);
                return -1;
            }
            g_dir[i].block0 = cursor;
            g_dir[i].nblocks = 0;
            continue;
        }
        uint64_t want_blocks;
        if (g_pk_version == 1) {
            want_blocks = g_dir[i].count / g_pk_block;
        } else {
            want_blocks = seg[i];
            /* The segment table must agree with the directory,
             * tensor by tensor: a pack serialized in a
             * different tensor order (or for a different
             * directory) fails loudly here instead of binding
             * stream segments to the wrong weights by
             * position (Codex Astra review H1, 2026-11-25). */
            if (want_blocks != g_dir[i].count / bi) {
                fprintf(stderr, "packed segment %d holds %llu "
                        "blocks but tensor %s has %llu (pack and "
                        "directory do not match)\n", i,
                        (unsigned long long)want_blocks,
                        g_dir[i].name,
                        (unsigned long long)(g_dir[i].count / bi));
                return -1;
            }
        }
        g_dir[i].block0 = cursor;
        g_dir[i].nblocks = want_blocks;
        cursor += want_blocks;
        if (stored_books == 1) {
            g_pk_book_off[i] = 0;
        } else {
            g_pk_book_off[i] = book_cursor;
            book_cursor += (uint64_t)g_pk_entries * bi;
            books_seen++;
        }
    }
    /* Read-integrity (Round 4 P0 gap B): the per-tensor
     * block walk must reconcile with the header's total. */
    if (cursor != g_pk_nblocks) {
        fprintf(stderr, "packed block walk mismatch: directory "
                "accounts for %llu blocks, header says %llu\n",
                (unsigned long long)cursor,
                (unsigned long long)g_pk_nblocks);
        free(seg);
        return -1;
    }
    if (stored_books != 1 && books_seen != stored_books) {
        fprintf(stderr, "packed book holds %u books but %u "
                "packed tensors need one\n",
                stored_books, books_seen);
        return -1;
    }
    if (book_cursor > total_book_floats
            || (stored_books != 1
                && book_cursor != total_book_floats)) {
        fprintf(stderr, "packed book size mismatch\n");
        return -1;
    }
    free(seg);
    free(raw_list);
    if (raw_bytes) {
        g_pk_rawdata = (uint16_t *)malloc(raw_bytes);
        if (!g_pk_rawdata) return -1;
        if (seek_to(g_pk_stream, g_pk_header
                    + (uint64_t)(g_pk_idxw + 4) * g_pk_nblocks) != 0
                || fread(g_pk_rawdata, 1, raw_bytes, g_pk_stream)
                       != raw_bytes)
            return -1;
    }
    if (g_pk_version == 3) {
        if (pk_verify_manifest(stream_path) != 0) return -1;
    } else {
        fprintf(stderr, "notice: legacy pack (format v%u): no "
                "generation identity; the stream and the book "
                "are bound by position, as before\n",
                g_pk_version);
    }
    return 0;
}

/* ================= prefetch pipeline =================
 *
 * JD's two-step prefetch, at tensor granularity: the forward
 * pass consumes weight tensors in a fixed ordinal order. One
 * slot computes while the prefetch thread materializes the
 * next ordinal into the other slot; the ordinal after that is
 * still on disc. Two tensors open at a time, ever.
 */

enum { SLOT_Q = 0, SLOT_K, SLOT_V, SLOT_O, SLOT_GATE, SLOT_UP,
       SLOT_DOWN, SLOTS_PER_LAYER };

/* Wall clock for pipeline telemetry only (never touches the math). */
static double pipe_now(void)
{
#ifdef _WIN32
    static LARGE_INTEGER freq;
    static int have_freq = 0;
    LARGE_INTEGER c;
    if (!have_freq) { QueryPerformanceFrequency(&freq); have_freq = 1; }
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
#endif
}

#define PIPE_MAX_LANES 64
/* Prefetch-list entry states (Tensor Handshake stage 2). */
enum { ORD_IDLE = 0, ORD_STAGED = 1, ORD_EXPANDING = 2,
       ORD_READY = 3 };

/* Precision Gate marks, carried on the same list: the scorer
 * finalizes each ordinal DTC or EXPAND ahead of the claim
 * window; an UNDECIDED ordinal is never claimable by a lane,
 * and the consumer's demand path resolves it as EXPAND. */
enum { MARK_UNDECIDED = 0, MARK_EXPAND = 1, MARK_DTC = 2 };

/* Telemetry roles: expansion lanes 0..PIPE_MAX_LANES-1, then the
 * consumer (the core loop), then the fetcher / fused worker. Each
 * role writes only its own accumulators; pipe_report reduces them
 * in this fixed order after the join. */
#define PIPE_ROLE_CONSUMER PIPE_MAX_LANES
#define PIPE_ROLE_FETCH (PIPE_MAX_LANES + 1)
#define PIPE_ROLES (PIPE_MAX_LANES + 2)

typedef struct {
    /* Tensor Handshake door (zero mutex, 2026-11-26): every
     * published change bumps it; every waiter opens it before
     * evaluating its predicate. Replaces the lock and condvar. */
    uint32_t door;
    uint32_t exited;        /* worker exit ticks (Windows join) */
#ifdef _WIN32
    HANDLE thread;
#else
    pthread_t thread;    /* fetch worker (triforce) */
    pthread_t lanes[PIPE_MAX_LANES]; /* unpack lanes (triforce) */
    int n_lanes;
#endif
    /* Pass-boundary handshake: the core loop asks for a boundary
     * (boundary_req), every worker parks at its decision point
     * with no claim held and acknowledges (ack_fetch, ack_lane),
     * the core loop alone rebuilds the list, then opens the new
     * pass (epoch_go). pass_consumed is the core loop's own
     * "is it time?": it counts consume ticks since the last
     * boundary, so the boundary runs exactly once per pass
     * however many entry points ask for it. */
    uint32_t boundary_req, epoch_go;
    uint32_t ack_fetch;
    uint32_t ack_lane[PIPE_MAX_LANES];
    int pass_consumed;
    float **buf;
    size_t cap_elems;
    int *have;
    int slots;              /* runtime: --pipe-slots */
    int want;
    int busy;
    int stop;
    int max_ord;
    /* Telemetry (JD's lookahead, measured): when each ordinal
     * became ready, how long the consumer stalled on misses, how
     * long ready tensors waited on hits (slack), and the load
     * path's own cost. Loads are serialized by the busy flag, so
     * the last-load fields are safe. */
    double *ready_at;
    double load_ms_last[PIPE_ROLES];     /* per role */
    uint64_t load_bytes_last[PIPE_ROLES];
    double stat_load_ms_r[PIPE_ROLES];   /* per role, reduced */
    uint64_t stat_bytes_r[PIPE_ROLES];
    double stat_stall_s, stat_slack_s, stat_load_ms;
    uint64_t stat_bytes;
    long stat_hits, stat_miss;
    /* Hits that carry a ready-slack stamp (expanded ordinals):
     * the slack mean divides by these, not by every hit, since a
     * DTC consume has no expansion-ready stamp (Archie Round 5
     * P2-PG2). */
    long stat_slack_n;
    /* The Tensor Handshake layer counter (JD, 2026-11-25): the
     * consumer's committed position in the ordinal sequence,
     * and the consume tick of the stage-2 protocol (below):
     * the prefetch priority list derives its horizon from it,
     * and the phase-boundary check compares it against the
     * fetch and expansion ticks before a pass may roll over.
     * Written only by the consumer inside pipe_get. It
     * coordinates preparation; consumption itself commits in
     * ordinal order whatever the list says, so it cannot
     * change results. */
    int layer_counter;
    /* Stall forensics (lab build only): per-ordinal accumulation
     * so every miss is attributable to a layer, a slot, and a
     * cause (waited on the worker, or the consumer gave up and
     * loaded the tensor itself). */
    long *fx_miss_count, *fx_self_count, *fx_load_count;
    double *fx_miss_wait_s, *fx_load_ms_sum;
    /* Triforce split state (POSIX path): the fetch thread owns
     * the disc, the unpack thread owns reconstruction, the main
     * loop consumes in fixed order. Three roles, three threads,
     * serialized consumption: the Timing Handshake. */
    Stage *st;
    int *st_owner;          /* ordinal staged in st[slot]; -1 none */
    int want_unpack;             /* unpack demand, set by the consumer */
    int fetched_upto;       /* highest ordinal staged this pass */
    int epoch;              /* bumped at each pass restart */
    int fetch_busy, unpack_busy;
    /* Demand generation (Archie P1-1): the consumer stamps each
     * posted demand; a worker stamps demand_seen when it next
     * evaluates its wait condition. The consumer may only fall
     * back to loading an ordinal itself once a worker has had
     * the chance to claim that demand... before this, the busy
     * flags were checked while the consumer still held the lock
     * it had broadcast under, so on a cold miss the workers
     * could never win the race and the consumer always
     * self-loaded. */
    long demand_posted, demand_seen;
    /* Tensor Handshake stage 2 (JD, 2026-11-25): the global
     * prefetch priority list and the phase-boundary ticks.
     * ord_state[ord] / ord_claim[ord] are the list: each
     * ordinal is IDLE, STAGED (sections staged, unclaimed),
     * EXPANDING (claimed by exactly one worker, or the
     * consumer) or READY. Priority is derived, never voted:
     * the lowest ordinal that is not READY is always the
     * highest-priority work, for the fetcher and for every
     * expansion lane alike, so work selection comes from the
     * list, not from private arithmetic. The row_* fields are
     * the per-worker rows: each worker publishes the ordinal
     * it is working on and the next one it intends to claim,
     * and writes only its own row. list_generation bumps on every
     * list mutation and broadcasts with it: one door for
     * every waiter, and the generation counter is the phase
     * boundary mechanism. The three ticks are fetched_upto
     * (fetch), n_ready (expansions completed, counted) and
     * layer_counter (consume); at a pass restart the consumer
     * verifies all three agree the prior pass drained before
     * the pass may roll over, and a disagreement is a printed
     * fault with a visible drain, never a silent wait. The
     * list governs scheduling only, never arithmetic or
     * commit order: consumption commits in ordinal order
     * regardless of expansion completion order, which is
     * what keeps every output bit-identical. */
    uint8_t *ord_state;     /* [max_ord+1] ORD_* */
    int *ord_claim;         /* [max_ord+1] lane id, 32 = consumer, -1 none */
    uint8_t *ord_expand_counted;   /* [max_ord+1] expansion tick counted this pass */
    int n_ready;            /* ordinals whose expansion completed this pass */
    long list_generation;          /* bumped on every list mutation */
    int expand_busy_n;      /* expansions in flight (lanes + consumer) */
    int row_fetch, row_fetch_next;
    int row_expand[PIPE_MAX_LANES + 1], row_expand_next[PIPE_MAX_LANES + 1];
    int row_consume;
    /* Highest ordinal the consumer has requested this pass.
     * The expansion window is bounded by buffer lifetime: an
     * ordinal x may be expanded into its slot only once the
     * slot's previous occupant (x - slots) is dead, which is
     * exactly when the consumer has requested x... so lanes
     * may claim at most req_max + slots - 1. Expanding past
     * that overwrites a tensor whose matmul has not run yet
     * (measured: nondeterministic tokens at slots 2 when the
     * window ran one ordinal too far). Fetch staging has no
     * such bound and runs as far ahead as configured; the
     * decoupling of fetch depth from expansion width lives
     * here. */
    int req_max;
    /* Precision Gate: the per-ordinal mark the scorer
     * finalized for this pass, each ordinal's DTC eligibility
     * (decided once at pipe_init from the pack, never
     * mid-pass), and the per-slot float mn/sc arrays the DTC
     * consume converts the staged halves into (POSIX only,
     * counted in g_pipe_staging_bytes). */
    uint8_t *ord_mark;      /* [max_ord+1] MARK_* */
    uint8_t *ord_dtc_ok;    /* [max_ord+1] (POSIX) */
    float **dtc_mn, **dtc_sc;   /* [slots][nb_max] (POSIX) */
    FILE *pk_priv;          /* fetch thread's own stream handle */
    /* Fetch thread's own handle per shard, for staging raw
     * tensor bytes at fetch time. pipe_fetch executions are
     * mutually exclusive across threads (the busy flags), so a
     * private handle per shard is enough; what it removes is
     * any interleaving with the main thread's head and embed
     * reads on the shared g_shards handles. */
    FILE *shard_priv[MAX_SHARDS];
} Pipe;

static Pipe g_pipe;
/* Precision Gate per-worker DTC tables (POSIX): T holds
 * nseg * entries floats and A holds nseg floats, one pair per
 * pool worker, allocated in pipe_init. Partitioning the DTC
 * consume by position cannot change any output bit: each
 * output element is produced by exactly one nqdc_apply row
 * loop against that position's own table. */
static float **g_dtc_T = NULL, **g_dtc_A = NULL;
/* Per-layer DTC eligibility: every one of the layer's seven
 * ordinals eligible (marks are per layer, calibration is per
 * layer). Filled in pipe_init. */
static uint8_t *g_gate_layer_ok = NULL;   /* [layers] */
/* Total bytes of per-slot staging buffers pipe_init actually
 * allocated (packed idx + fraction bytes + raw bytes), so the
 * startup budget accounts for them (M-I6). */
static size_t g_pipe_staging_bytes = 0;

/* Slot of an ordinal under the runtime slot count. */
static int pslot(int ord) { return ord % g_pipe.slots; }

static const TEnt *ord_ent(int ord)
{
    int layer = ord / SLOTS_PER_LAYER;
    int slot = ord % SLOTS_PER_LAYER;
    static const char *names[SLOTS_PER_LAYER] = {
        "self_attn.q_proj.weight", "self_attn.k_proj.weight",
        "self_attn.v_proj.weight", "self_attn.o_proj.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight",
        "mlp.down_proj.weight"
    };
    char full[256];
    snprintf(full, sizeof(full), "encoder.layers.%d.%s", layer, names[slot]);
    return dir_find(full);
}

/* The architecture globals are defined in the model
 * section below; tentative declarations so the shape
 * validator can sit with the pipe code it guards.
 * load_model fills them before the validator runs. */
static int M_H, M_L, M_HEADS, M_KVHEADS, M_HD, M_FFN,
    M_VOCAB;
static TEnt *g_head_ent;

/* Exact per-layer projection shapes, validated before any
 * worker starts (Sol H5). The pipeline stages each projection
 * into a shared slot buffer sized by the largest tensor, and
 * the matmuls consume architecture-derived dimensions; before
 * this check, a projection was verified only against the
 * buffer capacity, so a short one left stale bytes in the
 * reused buffer and a transposed same-count one passed every
 * test. The architecture is derived from layer 0 and the
 * embedding, so layer 0's row counts are the derivation
 * itself; every other tensor, and every column count, is
 * checked exactly here. The head is checked too: head_all
 * tiles it as M_VOCAB rows of M_H. */
static int validate_projection_shapes(void)
{
    int qdim = M_HEADS * M_HD, kv_dim = M_KVHEADS * M_HD;
    for (int ord = 0; ord < M_L * SLOTS_PER_LAYER; ord++) {
        const TEnt *t = ord_ent(ord);
        int rows, cols;
        if (!t) {
            fprintf(stderr, "geometry: no tensor for ordinal "
                    "%d\n", ord);
            return -1;
        }
        switch (ord % SLOTS_PER_LAYER) {
        case SLOT_Q: rows = qdim; cols = M_H; break;
        case SLOT_K: case SLOT_V: rows = kv_dim; cols = M_H; break;
        case SLOT_O: rows = M_H; cols = qdim; break;
        case SLOT_GATE: case SLOT_UP: rows = M_FFN; cols = M_H; break;
        default: rows = M_H; cols = M_FFN; break; /* SLOT_DOWN */
        }
        if (t->rows != rows || t->cols != cols) {
            fprintf(stderr, "geometry: %s is %dx%d, the "
                    "architecture requires %dx%d\n",
                    t->name, t->rows, t->cols, rows, cols);
            return -1;
        }
    }
    if (!g_head_ent || g_head_ent->rows != M_VOCAB
            || g_head_ent->cols != M_H) {
        fprintf(stderr, "geometry: diffusion_head.weight must "
                "be %dx%d\n", M_VOCAB, M_H);
        return -1;
    }
    return 0;
}

static void pipe_load(int slot, int ord, int role)
{
    const TEnt *t = ord_ent(ord);
    if (!t || t->count > g_pipe.cap_elems) {
        fprintf(stderr, "pipe: tensor for ordinal %d missing/oversize\n", ord);
        exit(1);
    }
    double t0 = pipe_now();
    if (src_read_range(t, 0, t->count, g_pipe.buf[slot]) != 0) {
        fprintf(stderr, "pipe: read failed for %s\n", t->name);
        exit(1);
    }
    g_pipe.load_ms_last[role] = (pipe_now() - t0) * 1000.0;
    g_pipe.load_bytes_last[role] = t->count * sizeof(float);
}

#ifndef _WIN32
/* The staging helper and the split pipe halves below are
 * POSIX-only, like the packed halves above (Archie Round 2
 * P2-B). */
/* Read the raw source bytes of elements [elem0, elem0+nelems)
 * of a tensor into the stage's raw buffer, on the fetcher's
 * private handle for the tensor's shard. The expand half then
 * converts from memory and does no disc I/O at all. */
static int raw_stage_range(const TEnt *t, uint64_t elem0,
                           uint64_t nelems, Stage *s)
{
    int esz = t->dtype == DT_F32 ? 4 : 2;
    FILE *f = g_pipe.shard_priv[t->shard];
    if (!f) return -1;
    if (seek_to(f, t->byte_off + elem0 * (uint64_t)esz) != 0
            || fread(s->raw, esz, nelems, f) != nelems)
        return -1;
    return 0;
}

/* Fetch half of a pipe load: disc only, into the slot's stage,
 * never into buf[slot] (the fetcher may run slots-1 ordinals
 * ahead of consumption, and buf[slot] is then still the
 * consumer's live buffer for an earlier ordinal: reading
 * straight into it overwrote tensors mid-compute, the
 * 2026-11-24 anchor failure). Three shapes:
 *   mode 1: fully packed tensor, sections staged through the
 *           private stream handle (the original split).
 *   mode 2: raw bytes staged through the private shard handle
 *           (Archie P1-1, completed 2026-11-24 night). When a
 *           packed prefix exists (a tensor the pack covers
 *           only in part), its sections are staged too and the
 *           raw bytes cover the tail from raw_elem0 on.
 *   mode 3: legacy fused fallback, only when a private handle
 *           could not be opened; expand reads through the
 *           shared handles, as mode 2 did before the staging
 *           buffer existed. */
static void pipe_fetch(int slot, int ord)
{
    const TEnt *t = ord_ent(ord);
    if (!t || t->count > g_pipe.cap_elems) {
        fprintf(stderr, "pipe: tensor for ordinal %d missing/oversize\n", ord);
        exit(1);
    }
    Stage *s = &g_pipe.st[slot];
    double t0 = pipe_now();
    s->mode = 0;
    s->raw_elem0 = 0;
    s->raw_nelems = 0;
    uint64_t packed_elems = (g_use_packed && g_pk_block)
        ? t->nblocks * pk_block_of(t) : 0;
    if (packed_elems >= t->count && g_pipe.pk_priv) {
        if (packed_fetch_range(g_pipe.pk_priv, t, 0, t->count, s) != 0) {
            fprintf(stderr, "pipe: fetch failed for %s\n", t->name);
            exit(1);
        }
        s->mode = 1;
    } else if ((!packed_elems || g_pipe.pk_priv)
               && g_pipe.shard_priv[t->shard]) {
        if (packed_elems > 0
                && packed_fetch_range(g_pipe.pk_priv, t, 0,
                                      packed_elems, s) != 0) {
            fprintf(stderr, "pipe: fetch failed for %s\n", t->name);
            exit(1);
        }
        s->raw_elem0 = packed_elems;
        s->raw_nelems = t->count - packed_elems;
        if (s->raw_nelems > 0
                && raw_stage_range(t, s->raw_elem0, s->raw_nelems,
                                   s) != 0) {
            fprintf(stderr, "pipe: fetch failed for %s\n", t->name);
            exit(1);
        }
        s->mode = 2;
    } else {
        s->mode = 3;
    }
    s->fetch_ms = (pipe_now() - t0) * 1000.0;
}

/* Unpack half: reconstruction only, from the stage into the
 * slot. Modes 1 and 2 do no disc I/O here at all: mode 1
 * expands the staged packed sections, mode 2 converts the
 * staged raw bytes (plus the staged packed prefix when
 * raw_elem0 > 0). Mode 3 is the legacy fused fallback and
 * still reads through the shared handles. */
static void pipe_expand(int slot, int ord, int role)
{
    const TEnt *t = ord_ent(ord);
    Stage *s = &g_pipe.st[slot];
    double t0 = pipe_now();
    if (s->mode == 1) {
        if (packed_expand_range(t, 0, t->count, s, g_pipe.buf[slot]) != 0) {
            fprintf(stderr, "pipe: expand failed for %s\n", t->name);
            exit(1);
        }
    } else if (s->mode == 2) {
        if (s->raw_elem0 > 0
                && packed_expand_range(t, 0, s->raw_elem0, s,
                                       g_pipe.buf[slot]) != 0) {
            fprintf(stderr, "pipe: expand failed for %s\n", t->name);
            exit(1);
        }
        if (s->raw_nelems > 0)
            raw_convert(t, s->raw, s->raw_nelems,
                        g_pipe.buf[slot] + s->raw_elem0);
    } else if (s->mode == 3) {
        if (src_read_range(t, 0, t->count, g_pipe.buf[slot]) != 0) {
            fprintf(stderr, "pipe: read failed for %s\n", t->name);
            exit(1);
        }
    }
    g_pipe.load_ms_last[role] = s->fetch_ms + (pipe_now() - t0) * 1000.0;
    g_pipe.load_bytes_last[role] = t->count * sizeof(float);
    s->mode = 0;
}
#endif /* !_WIN32 */

/* Tensor Handshake helpers (zero mutex, 2026-11-26). */
static void pipe_bump(void) { hs_door_bump(&g_pipe.door); }

#ifndef _WIN32
/* A worker's acknowledgement of the consumer's latest demand
 * (Archie P1-1): raised monotonically, and a change is a list
 * mutation that bumps the door (Sol H1); an unchanged value is
 * not, so the generation counter stays an honest instrument. */
static void pipe_ack_demand(void)
{
    if (hs_raise_long(&g_pipe.demand_seen, HS_LD(&g_pipe.demand_posted))) {
        HS_ADD(&g_pipe.list_generation, 1);
        pipe_bump();
    }
}

/* A worker's pass-boundary check, made only at its decision
 * point, where it holds no claim and no token: if the core loop
 * has asked for a boundary, acknowledge it and park until the
 * new pass opens (or the pipe stops). Returns 1 if it parked. */
static int pipe_park_at_boundary(uint32_t *ack)
{
    uint32_t br = HS_LD(&g_pipe.boundary_req);
    if (br == HS_LDX(ack)) return 0;
    HS_ST(ack, br);
    pipe_bump();
    for (;;) {
        uint32_t d = hs_door_open(&g_pipe.door);
        if (HS_LD(&g_pipe.stop) || HS_LD(&g_pipe.epoch_go) == br)
            return 1;
        hs_door_wait(&g_pipe.door, d);
    }
}
#endif

/* Called by the holder of the ordinal's claim or load token (a
 * lane, the fused worker or the consumer), or by the consumer
 * alone inside a pass boundary, once the ordinal's buffer is
 * complete. set_have publishes the slot's resident ordinal; the
 * DTC consume path, which never fills the buffer, passes 0. */
static void pipe_mark_ready(int ord, int role, int set_have)
{
    /* Tensor Handshake stage 2: the ordinal joins the READY
     * set, its claim is released, and the list generation
     * bumps so every waiter re-evaluates. The expansion
     * tick counts each ordinal once per pass (ord_expand_counted):
     * an ordinal that stayed resident from the last pass is
     * counted at the restart, and if its slot is later reused
     * and it is expanded again, the re-mark must not count it
     * twice. Zero mutex: everything this writes is owned by the
     * claim holder until the claim is released, and the release
     * (the store of -1) comes after every other write, so a
     * thread that sees the claim free sees the whole result,
     * and one that sees have[] name the ordinal sees its
     * ready_at stamp. */
    g_pipe.ready_at[ord] = pipe_now();
    g_pipe.stat_load_ms_r[role] += g_pipe.load_ms_last[role];
    g_pipe.stat_bytes_r[role] += g_pipe.load_bytes_last[role];
    g_pipe.fx_load_ms_sum[ord] += g_pipe.load_ms_last[role];
    g_pipe.fx_load_count[ord]++;
    if (set_have) HS_ST(&g_pipe.have[pslot(ord)], ord);
    if (g_pipe.ord_state) HS_ST(&g_pipe.ord_state[ord], (uint8_t)ORD_READY);
    if (g_pipe.ord_expand_counted
            && !HS_LDX(&g_pipe.ord_expand_counted[ord])) {
        HS_STX(&g_pipe.ord_expand_counted[ord], (uint8_t)1);
        HS_ADD(&g_pipe.n_ready, 1);
    }
    HS_ADD(&g_pipe.list_generation, 1);
    if (g_pipe.ord_claim) HS_ST(&g_pipe.ord_claim[ord], -1);
}

#ifdef _WIN32
/* Windows keeps the fused single worker in this variant; the
 * triforce split is implemented and measured on the POSIX
 * path. Zero mutex: the busy word is the load token, taken
 * with compare-and-swap by the worker or by the consumer, and
 * released by its holder after have[] is published. */
static HS_THREAD_FN(pipe_worker, unused)
{
    (void)unused;
    for (;;) {
        uint32_t d = hs_door_open(&g_pipe.door);
        if (HS_LD(&g_pipe.stop)) break;
        int ord = HS_LD(&g_pipe.want);
        if (ord >= 0 && ord <= g_pipe.max_ord
                && HS_LD(&g_pipe.have[pslot(ord)]) != ord
                && hs_cas_int(&g_pipe.busy, 0, 1)) {
            /* Re-check under the token: the consumer may have
             * loaded it between the read and the claim. want
             * cannot move past ord until ord is resident. */
            if (HS_LD(&g_pipe.have[pslot(ord)]) != ord) {
                pipe_load(pslot(ord), ord, PIPE_ROLE_FETCH);
                pipe_mark_ready(ord, PIPE_ROLE_FETCH, 1);
            }
            HS_ST(&g_pipe.busy, 0);
            pipe_bump();
            continue;
        }
        hs_door_wait(&g_pipe.door, d);
    }
    hs_exit_tick(&g_pipe.exited);
    HS_THREAD_RETURN;
}
#else
/* The fetcher's slot-free predicate: the slot's staging is free
 * when nothing is staged in it or its staged ordinal has been
 * expanded into the slot's buffer. */
static int pipe_stage_free(int c)
{
    int so = HS_LD(&g_pipe.st_owner[pslot(c)]);
    return so == -1 || HS_LD(&g_pipe.have[pslot(c)]) == so;
}

/* Triforce fetch worker: owns the disc. Stages the next
 * ordinal after the highest ever staged, at most
 * --fetch-ahead ordinals ahead of the unpack demand (runtime
 * configured), and only into a staging slot whose previous
 * occupant has been fully expanded. Zero mutex: fetch_busy is
 * the fetch token (compare-and-swap, cleared only by its
 * holder); fetched_upto and st_owner[] are written only by the
 * token holder, so the publication order below is the whole
 * protocol: stage bytes, then st_owner (release), then the
 * fetch tick. */
static void *fetch_worker(void *unused)
{
    (void)unused;
    for (;;) {
        uint32_t d = hs_door_open(&g_pipe.door);
        if (HS_LD(&g_pipe.stop)) return NULL;
        if (pipe_park_at_boundary(&g_pipe.ack_fetch)) continue;
        int c = HS_LD(&g_pipe.fetched_upto) + 1;
        int resident = c <= g_pipe.max_ord
            && HS_LD(&g_pipe.have[pslot(c)]) == c && pipe_stage_free(c);
        int fetchable = c <= g_pipe.max_ord
            && c <= HS_LD(&g_pipe.want_unpack) + g_fetch_ahead
            && pipe_stage_free(c);
        if ((resident || fetchable) && !HS_LD(&g_pipe.fetch_busy)
                && hs_cas_int(&g_pipe.fetch_busy, 0, 1)) {
            /* Re-evaluate under the token: only the holder moves
             * fetched_upto, so c is stable now. */
            c = HS_LD(&g_pipe.fetched_upto) + 1;
            if (c <= g_pipe.max_ord && HS_LD(&g_pipe.have[pslot(c)]) == c
                    && pipe_stage_free(c)) {
                /* Already resident from an earlier pass: mark it
                 * staged without re-reading it from disc (Archie
                 * P1-1; with slots > max_ord the fetcher used to
                 * re-read the whole model on every pass). The
                 * slot-free predicate is the same one the fetch
                 * branch below uses (Sol H1): staged data and
                 * resident data are separate fields and neither
                 * may overwrite the other's claim. */
                HS_ST(&g_pipe.st_owner[pslot(c)], c);
                HS_ST(&g_pipe.fetched_upto, c);
                HS_ADD(&g_pipe.list_generation, 1);
            } else if (c <= g_pipe.max_ord
                    && c <= HS_LD(&g_pipe.want_unpack) + g_fetch_ahead
                    && pipe_stage_free(c)) {
                HS_STX(&g_pipe.row_fetch, c);
                HS_STX(&g_pipe.row_fetch_next,
                       (c + 1 <= g_pipe.max_ord) ? c + 1 : -1);
                pipe_fetch(pslot(c), c);
                /* State before owner: once st_owner names c a lane
                 * may claim it and mark it READY, which this store
                 * must not overwrite. */
                HS_ST(&g_pipe.ord_state[c],
                      (uint8_t)((HS_LD(&g_pipe.have[pslot(c)]) == c)
                                ? ORD_READY : ORD_STAGED));
                HS_ST(&g_pipe.st_owner[pslot(c)], c);
                HS_ST(&g_pipe.fetched_upto, c);
                HS_ADD(&g_pipe.list_generation, 1);
            }
            HS_ST(&g_pipe.fetch_busy, 0);
            pipe_bump();
            continue;
        }
        pipe_ack_demand();
        HS_STX(&g_pipe.row_fetch, -1);
        hs_door_wait(&g_pipe.door, d);
    }
}

/* Triforce unpack lane: owns reconstruction. Each lane
 * claims work from the shared prefetch priority list: the
 * lowest staged, unexpanded, unclaimed ordinal inside the
 * expansion window, which is the demanded ordinal first by
 * construction. Claim exclusivity is a hard rule (JD,
 * 2026-11-25): an ordinal is claimed by exactly one lane, by
 * compare-and-swap on ord_claim (zero mutex, 2026-11-26), and
 * no second lane can take it while the claim stands; the
 * consumer's self-service path takes the same claim the same
 * way. Only the holder clears a claim, and always before it
 * returns to its decision point, so no claim can be held
 * across a pass boundary. A lane whose expansion finishes out
 * of ordinal order simply marks its ordinal ready;
 * consumption still commits in ordinal order. */
static void *unpack_worker(void *arg)
{
    int lane = (int)(intptr_t)arg;
    for (;;) {
        uint32_t d = hs_door_open(&g_pipe.door);
        if (HS_LD(&g_pipe.stop)) return NULL;
        if (pipe_park_at_boundary(&g_pipe.ack_lane[lane])) continue;
        int ord = -1;
        {
            /* Priority: the lowest staged, unexpanded,
             * unclaimed ordinal inside the expansion
             * window. The window is bounded by buffer
             * lifetime (req_max + slots - 1, see the
             * req_max field), NOT by the fetch depth:
             * staging may run far ahead, expansion may
             * not. The demanded ordinal is always inside
             * the window (slots >= 2), so the consumer's
             * own demand can always be claimed. */
                /* Precision Gate: a lane claims only an
                 * ordinal the scorer finalized EXPAND. A
                 * DTC ordinal's sections stay packed for
                 * the consumer's table path; an UNDECIDED
                 * one waits for the scorer, which always
                 * finalizes ahead of this window. */
            int hi = HS_LD(&g_pipe.req_max) + g_pipe.slots - 1;
            if (hi > g_pipe.max_ord) hi = g_pipe.max_ord;
            for (int o = 0; o <= hi; o++) {
                if (HS_LD(&g_pipe.st_owner[pslot(o)]) == o
                        && HS_LD(&g_pipe.have[pslot(o)]) != o
                        && HS_LD(&g_pipe.ord_claim[o]) < 0
                    && HS_LD(&g_pipe.ord_mark[o]) == MARK_EXPAND) {
                    ord = o;
                    break;
                }
            }
        }
        if (ord >= 0) {
            if (!hs_cas_int(&g_pipe.ord_claim[ord], -1, lane))
                continue;           /* another owner won it */
            /* Re-check under the claim: while it stands the
             * stage cannot be re-staged (the slot is not free)
             * and no one else may expand this ordinal. */
            if (HS_LD(&g_pipe.st_owner[pslot(ord)]) != ord
                    || HS_LD(&g_pipe.have[pslot(ord)]) == ord) {
                HS_ST(&g_pipe.ord_claim[ord], -1);
                pipe_bump();
                continue;
            }
            HS_ADD(&g_pipe.expand_busy_n, 1);
            HS_ST(&g_pipe.ord_state[ord], (uint8_t)ORD_EXPANDING);
            HS_STX(&g_pipe.row_expand[lane], ord);
            HS_STX(&g_pipe.row_expand_next[lane],
                   (ord + 1 <= g_pipe.max_ord) ? ord + 1 : -1);
            HS_ADD(&g_pipe.list_generation, 1);
            pipe_expand(pslot(ord), ord, lane);
            pipe_mark_ready(ord, lane, 1);
            HS_SUB(&g_pipe.expand_busy_n, 1);
            pipe_bump();
            continue;
        }
        /* Demand acknowledgement: bumps only when it changes,
         * like the fetcher's (Sol H1). */
        pipe_ack_demand();
        HS_STX(&g_pipe.row_expand[lane], -1);
        hs_door_wait(&g_pipe.door, d);
    }
}
#endif

static void pipe_init(int layers)
{
    size_t max_elems = 0;
    int max_ord = layers * SLOTS_PER_LAYER - 1;
    for (int o = 0; o <= max_ord; o++) {
        const TEnt *t = ord_ent(o);
        if (t && t->count > max_elems) max_elems = t->count;
    }
    if (!g_slots_explicit) {
        /* Device-derived slot default (prefetch adaptability,
         * 2026-11-25): as many slots as the device can feed,
         * no more. The fetcher, the expander, the consumer and
         * the compute pool share the machine, so cores/2 is the
         * useful ceiling; memory bounds it too, at a quarter
         * of physical RAM against the per-slot cost (the float
         * buffer plus its staging twin). Clamped to [2, 8]:
         * the pipe needs two slots to overlap at all, and past
         * eight the working set outruns any cache benefit. On
         * the bench (4 cores, 8 GB, 108 MB slot cost) this
         * yields exactly 2, the historical default, so an
         * unconfigured run is unchanged there. */
        long by_cores = cpu_count() / 2;
        uint64_t ram = device_ram_bytes();
        uint64_t slot_cost = (uint64_t)max_elems * 2 * sizeof(float);
        long by_ram = (ram && slot_cost)
            ? (long)(ram / 4 / slot_cost) : 2;
        long d = by_cores < by_ram ? by_cores : by_ram;
        if (d < 2) d = 2;
        if (d > 8) d = 8;
        g_pipe_slots = (int)d;
        g_slots_src = "device";
    }
    if (!g_lanes_explicit) {
        /* Device-derived expansion width, same doctrine as
         * the slot default above (Mega Tensor is hardware
         * aware: every default is derived from the machine
         * it runs on, JD 2026-11-25): cores/2 lanes, bounded
         * by a quarter of RAM against the per-slot cost,
         * clamped to [1, 8] as a default. On the bench this
         * yields exactly 2, the width the pipeline was built
         * and proven at. */
        long by_cores = cpu_count() / 2;
        uint64_t ram = device_ram_bytes();
        uint64_t slot_cost = (uint64_t)max_elems * 2 * sizeof(float);
        long by_ram = (ram && slot_cost)
            ? (long)(ram / 4 / slot_cost) : 2;
        long d = by_cores < by_ram ? by_cores : by_ram;
        if (d < 1) d = 1;
        if (d > 8) d = 8;
        g_expand_lanes = (int)d;
        g_lanes_src = "device";
    }
    if (g_pipe_slots < 2 || g_fetch_ahead < 1 || g_prefetch_dist < 0
            || g_expand_lanes < 1 || g_expand_lanes > PIPE_MAX_LANES) {
        fprintf(stderr, "pipe: --pipe-slots must be >= 2, "
                "--fetch-ahead >= 1, --prefetch-dist >= 0 and "
                "--expand-lanes in [1, %d]\n", PIPE_MAX_LANES);
        exit(1);
    }
    /* The effective width is clamped against the device with
     * the reason printed, never silently honoured and never
     * an allocation failure: lanes cannot exceed slots (each
     * expansion target lives in a slot), and the family's
     * RAM guardrail (a quarter of physical RAM against the
     * per-slot cost) bounds the width the same way it bounds
     * the slot default. */
    int lanes_ceiling;
    {
        int eff = g_expand_lanes;
        uint64_t ram = device_ram_bytes();
        uint64_t slot_cost = (uint64_t)max_elems * 2 * sizeof(float);
        /* The hardware ceiling of the residency dial: the
         * widest expansion the slot count and the RAM
         * guardrail (a quarter of physical RAM at the
         * per-slot cost) permit on THIS machine. It is
         * printed in the startup line whether or not it
         * binds, so every run confesses the ceiling it ran
         * under (JD, 2026-11-25). */
        lanes_ceiling = g_pipe_slots;
        if (ram && slot_cost) {
            long by_ram = (long)(ram / 4 / slot_cost);
            if (by_ram < lanes_ceiling)
                lanes_ceiling = (int)by_ram;
            if (lanes_ceiling < 1) lanes_ceiling = 1;
        }
        if (eff > g_pipe_slots) {
            printf("pipe: expand lanes %d clamped to %d: one "
                   "expansion per staged slot\n",
                   eff, g_pipe_slots);
            eff = g_pipe_slots;
        }
        if (eff > lanes_ceiling) {
            printf("pipe: expand lanes %d clamped to %d: "
                   "hardware ceiling, estimated cost exceeds "
                   "a quarter of device RAM at %.1f MB per "
                   "slot\n", eff, lanes_ceiling,
                   slot_cost / 1048576.0);
            eff = lanes_ceiling;
        }
#ifndef _WIN32
        g_pipe.n_lanes = eff;   /* the Windows struct has no lanes */
#endif
        g_expand_lanes = eff;
    }
    g_pipe.slots = g_pipe_slots;
    g_pipe.cap_elems = max_elems;
    g_pipe.buf = (float **)malloc(sizeof(float *) * g_pipe.slots);
    g_pipe.have = (int *)malloc(sizeof(int) * g_pipe.slots);
    g_pipe.st = (Stage *)calloc(g_pipe.slots, sizeof(Stage));
    g_pipe.st_owner = (int *)malloc(sizeof(int) * g_pipe.slots);
    if (!g_pipe.buf || !g_pipe.have || !g_pipe.st || !g_pipe.st_owner) {
        fprintf(stderr, "pipe: out of memory\n");
        exit(1);
    }
    for (int s = 0; s < g_pipe.slots; s++) {
        g_pipe.buf[s] = (float *)malloc(max_elems * sizeof(float));
        if (!g_pipe.buf[s]) {
            fprintf(stderr, "pipe: out of memory\n");
            exit(1);
        }
        g_pipe.have[s] = -1;
        g_pipe.st_owner[s] = -1;
    }
    printf("pipe: %d slots (%s), fetch-ahead %d (%s), "
           "prefetch distance %d (%s), expand lanes %d (%s), "
           "hardware ceiling %d lanes\n",
           g_pipe.slots, g_slots_src, g_fetch_ahead, g_fetch_src,
           g_prefetch_dist, g_dist_src,
           g_expand_lanes, g_lanes_src, lanes_ceiling);
#ifdef _WIN32
    /* Sol M3: the startup line must not report settings the
     * fused Windows worker ignores as if they were in force.
     * Slots size its buffers and are effective; the other
     * three knobs configure the POSIX triforce split. */
    printf("pipe: note: the Windows fused worker loads on "
           "demand; fetch-ahead, prefetch distance and expand "
           "lanes do not take effect on this path\n");
#endif
    g_pipe.layer_counter = -1;
    g_pipe.want = -1;
    g_pipe.want_unpack = -1;
    g_pipe.fetched_upto = -1;
    g_pipe.fetch_busy = g_pipe.unpack_busy = 0;
    g_pipe.max_ord = max_ord;
    g_pipe.ready_at = (double *)calloc((size_t)max_ord + 1, sizeof(double));
    g_pipe.fx_miss_count = (long *)calloc((size_t)max_ord + 1, sizeof(long));
    g_pipe.fx_self_count = (long *)calloc((size_t)max_ord + 1, sizeof(long));
    g_pipe.fx_load_count = (long *)calloc((size_t)max_ord + 1, sizeof(long));
    g_pipe.fx_miss_wait_s = (double *)calloc((size_t)max_ord + 1, sizeof(double));
    g_pipe.fx_load_ms_sum = (double *)calloc((size_t)max_ord + 1, sizeof(double));
    if (!g_pipe.ready_at || !g_pipe.fx_miss_count || !g_pipe.fx_self_count
            || !g_pipe.fx_load_count || !g_pipe.fx_miss_wait_s
            || !g_pipe.fx_load_ms_sum) {
        fprintf(stderr, "pipe: out of memory\n");
        exit(1);
    }
    g_pipe.stat_stall_s = g_pipe.stat_slack_s = g_pipe.stat_load_ms = 0.0;
    g_pipe.stat_bytes = 0;
    g_pipe.stat_hits = g_pipe.stat_miss = 0;
    g_pipe.stat_slack_n = 0;
    /* Tensor Handshake stage 2: the prefetch priority list
     * starts empty, every worker row idle. */
    g_pipe.ord_state = (uint8_t *)calloc((size_t)max_ord + 1, 1);
    g_pipe.ord_claim = (int *)malloc(sizeof(int) * ((size_t)max_ord + 1));
    g_pipe.ord_expand_counted = (uint8_t *)calloc((size_t)max_ord + 1, 1);
    g_pipe.ord_mark = (uint8_t *)calloc((size_t)max_ord + 1, 1);
    g_pipe.ord_dtc_ok = (uint8_t *)calloc((size_t)max_ord + 1, 1);
    if (!g_pipe.ord_state || !g_pipe.ord_claim || !g_pipe.ord_expand_counted
            || !g_pipe.ord_mark || !g_pipe.ord_dtc_ok) {
        fprintf(stderr, "pipe: out of memory\n");
        exit(1);
    }
    for (int o = 0; o <= max_ord; o++) {
        g_pipe.ord_claim[o] = -1;
#ifdef _WIN32
        /* The fused Windows worker always expands; its marks
         * are set once, here, and never move. */
        g_pipe.ord_mark[o] = MARK_EXPAND;
#endif
    }
    g_pipe.n_ready = 0;
    g_pipe.list_generation = 0;
    g_pipe.expand_busy_n = 0;
    g_pipe.row_fetch = g_pipe.row_fetch_next = -1;
    for (int i = 0; i <= PIPE_MAX_LANES; i++)
        g_pipe.row_expand[i] = g_pipe.row_expand_next[i] = -1;
    g_pipe.row_consume = -1;
    g_pipe.req_max = -1;
    /* Triforce staging: raw packed sections for one whole
     * tensor per slot, sized from the largest tensor, plus the
     * raw staging buffer (source bytes of one whole tensor;
     * 4 bytes per element covers the widest source dtype).
     * The raw buffer belongs to the split fetch/expand path,
     * which is POSIX-only: the Windows worker runs the fused
     * loader and never touches it, so it is not allocated
     * there (Codex Astra review M-I6, 2026-11-25; at Sigma's
     * largest projection that was 108 MiB per slot of dead
     * allocation). g_pipe_staging_bytes totals what was
     * actually allocated so the printed budget can say so. */
    {
        uint32_t min_block = g_pk_block;
        if (g_pk_tblock)
            for (int i = 0; i < g_ndir; i++)
                if (g_pk_tblock[i] < min_block)
                    min_block = g_pk_tblock[i];
        uint64_t nb_max = min_block
            ? (uint64_t)max_elems / min_block + 1 : 1;
        for (int s = 0; s < g_pipe.slots; s++) {
            g_pipe.st[s].idx = (uint8_t *)malloc(nb_max);
            g_pipe.st[s].fr = (uint8_t *)malloc(nb_max * 4);
#ifdef _WIN32
            g_pipe.st[s].raw = NULL;
#else
            g_pipe.st[s].raw = (uint8_t *)malloc(
                (uint64_t)max_elems * 4);
#endif
            g_pipe.st[s].mode = 0;
            g_pipe.st_owner[s] = -1;
            if (!g_pipe.st[s].idx || !g_pipe.st[s].fr
#ifndef _WIN32
                    || !g_pipe.st[s].raw
#endif
                    ) {
                fprintf(stderr, "pipe: out of memory\n");
                exit(1);
            }
            g_pipe_staging_bytes += (size_t)nb_max * 5;
#ifndef _WIN32
            g_pipe_staging_bytes += (size_t)max_elems * 4;
#endif
        }
    }
    if (g_pk_stream_path) {
        g_pipe.pk_priv = fopen(g_pk_stream_path, "rb");
        if (g_pipe.pk_priv)
            setvbuf(g_pipe.pk_priv, NULL, _IOFBF, 1 << 18);
    }
    /* The fetcher's private handle per shard, for raw staging.
     * A handle that fails to open stays NULL and pipe_fetch
     * falls back to mode 3 for that shard's tensors. */
    for (int i = 0; i < g_nshards && i < MAX_SHARDS; i++) {
        g_pipe.shard_priv[i] = NULL;
        if (g_shard_paths[i]) {
            g_pipe.shard_priv[i] = fopen(g_shard_paths[i], "rb");
            if (g_pipe.shard_priv[i])
                setvbuf(g_pipe.shard_priv[i], NULL, _IOFBF, 1 << 18);
        }
    }
#ifndef _WIN32
    /* Precision Gate: DTC eligibility is decided once, here,
     * from the pack, never mid-pass: packed mode in use, the
     * fetcher's private stream handle open (so fetches stage
     * sections, mode 1), single-byte indices (the kernel's
     * idx is one byte per block; PQ packs are refused at open
     * regardless), the tensor fully packed (no raw tail, not
     * fp16-raw), and the block dividing the row width.
     * Ineligible ordinals are permanently MARK_EXPAND. */
    g_gate_layer_ok = (uint8_t *)malloc((size_t)layers);
    if (!g_gate_layer_ok) {
        fprintf(stderr, "pipe: out of memory\n");
        exit(1);
    }
    for (int l = 0; l < layers; l++) g_gate_layer_ok[l] = 1;
    for (int o = 0; o <= max_ord; o++) {
        const TEnt *t = ord_ent(o);
        uint32_t blk = t ? pk_block_of(t) : 0;
        int ok = g_use_packed && g_pipe.pk_priv != NULL
            && g_pk_idxw == 1 && t && blk > 0
            && t->nblocks > 0 && !pk_is_raw(t)
            && (uint64_t)t->nblocks * blk == t->count
            && t->cols > 0 && (t->cols % (int)blk) == 0;
        g_pipe.ord_dtc_ok[o] = ok ? 1 : 0;
        if (!ok) g_gate_layer_ok[o / SLOTS_PER_LAYER] = 0;
    }
    /* DTC consume workspace: per-slot float mn/sc (one float
     * per block of the largest tensor) and per-worker tables
     * (nseg * entries floats, bounded by the widest tensor). */
    {
        uint64_t nb_max = 1, nseg_max = 1;
        for (int o = 0; o <= max_ord; o++) {
            const TEnt *t = ord_ent(o);
            if (!t) continue;
            if (t->nblocks > nb_max) nb_max = t->nblocks;
            uint32_t blk = pk_block_of(t);
            if (blk && t->cols > 0) {
                uint64_t ns = (uint64_t)t->cols / blk;
                if (ns > nseg_max) nseg_max = ns;
            }
        }
        uint32_t entries = g_pk_entries ? g_pk_entries : 1;
        int nw = g_pool.nworkers > 0 ? g_pool.nworkers : 1;
        g_pipe.dtc_mn = (float **)malloc(sizeof(float *) * g_pipe.slots);
        g_pipe.dtc_sc = (float **)malloc(sizeof(float *) * g_pipe.slots);
        g_dtc_T = (float **)malloc(sizeof(float *) * nw);
        g_dtc_A = (float **)malloc(sizeof(float *) * nw);
        if (!g_pipe.dtc_mn || !g_pipe.dtc_sc || !g_dtc_T || !g_dtc_A) {
            fprintf(stderr, "pipe: out of memory\n");
            exit(1);
        }
        for (int s = 0; s < g_pipe.slots; s++) {
            g_pipe.dtc_mn[s] = (float *)malloc(nb_max * sizeof(float));
            g_pipe.dtc_sc[s] = (float *)malloc(nb_max * sizeof(float));
            if (!g_pipe.dtc_mn[s] || !g_pipe.dtc_sc[s]) {
                fprintf(stderr, "pipe: out of memory\n");
                exit(1);
            }
            g_pipe_staging_bytes += (size_t)nb_max * 2 * sizeof(float);
        }
        for (int wdi = 0; wdi < nw; wdi++) {
            g_dtc_T[wdi] = (float *)malloc(
                (size_t)nseg_max * entries * sizeof(float));
            g_dtc_A[wdi] = (float *)malloc(
                (size_t)nseg_max * sizeof(float));
            if (!g_dtc_T[wdi] || !g_dtc_A[wdi]) {
                fprintf(stderr, "pipe: out of memory\n");
                exit(1);
            }
        }
    }
#endif
#ifdef _WIN32
    g_pipe.thread = CreateThread(NULL, 0, pipe_worker, NULL, 0, NULL);
    if (!g_pipe.thread) {
        fprintf(stderr, "pipe: cannot start the worker thread\n");
        exit(1);
    }
#else
    /* Sol M6: a worker that fails to start is a startup
     * failure, never a silent infinite wait downstream. */
    if (pthread_create(&g_pipe.thread, NULL, fetch_worker, NULL) != 0) {
        fprintf(stderr, "pipe: cannot start the fetch worker\n");
        exit(1);
    }
    for (int i = 0; i < g_pipe.n_lanes; i++) {
        if (pthread_create(&g_pipe.lanes[i], NULL, unpack_worker,
                           (void *)(intptr_t)i) != 0) {
            fprintf(stderr, "pipe: cannot start expand lane %d "
                    "of %d\n", i, g_pipe.n_lanes);
            exit(1);
        }
    }
#endif
}

#ifndef _WIN32
/* Parse the gate block from an open text file positioned at
 * its first gate line. On success the calibration globals are
 * committed and 0 returned. want_gen is the identity the
 * block must name: the pack generation for a v3 manifest, the
 * computed content key for a legacy sidecar. */
static int gate_parse_block(FILE *f, uint64_t want_gen)
{
    char line[512];
    unsigned long long gen = 0;
    double eps = 0.0, budget = 0.0;
    int ver = 0;
    if (!fgets(line, sizeof(line), f)
            || sscanf(line, "gate %d", &ver) != 1 || ver != 1
            || !fgets(line, sizeof(line), f)
            || sscanf(line, "gate-gen %llx", &gen) != 1
            || gen != want_gen
            || !fgets(line, sizeof(line), f)
            || sscanf(line, "gate-eps %lf", &eps) != 1
            || !fgets(line, sizeof(line), f)
            || sscanf(line, "gate-budget %lf", &budget) != 1
            || !(eps > 0.0) || !(budget > 0.0))
        return -1;
    double *delta = (double *)malloc(sizeof(double) * M_L);
    double *cum = (double *)malloc(sizeof(double) * M_L);
    if (!delta || !cum) { free(delta); free(cum); return -1; }
    for (int l = 0; l < M_L; l++) {
        int li = -1;
        if (!fgets(line, sizeof(line), f)
                || sscanf(line, "gate-layer %d %lf %lf",
                          &li, &delta[l], &cum[l]) != 3
                || li != l) {
            free(delta); free(cum);
            return -1;
        }
    }
    g_gate_delta = delta;
    g_gate_cum = cum;
    g_gate_eps = eps;
    g_gate_budget = budget;
    /* Median layer increment: the hot-layer line is drawn at
     * four times it (plan section 3, static sensitivity map). */
    {
        double *srt = (double *)malloc(sizeof(double) * M_L);
        if (!srt) { free(delta); free(cum); return -1; }
        memcpy(srt, delta, sizeof(double) * M_L);
        for (int i = 1; i < M_L; i++) {
            double v = srt[i];
            int j = i - 1;
            while (j >= 0 && srt[j] > v) { srt[j + 1] = srt[j]; j--; }
            srt[j + 1] = v;
        }
        g_gate_delta_median = (M_L % 2)
            ? srt[M_L / 2]
            : 0.5 * (srt[M_L / 2 - 1] + srt[M_L / 2]);
        free(srt);
    }
    g_gate_calibrated = 1;
    return 0;
}

/* Load the open pack's Precision Gate calibration (plan
 * section 5). v3 packs carry the gate lines appended to the
 * digest manifest, after the tensor lines the manifest
 * verifier already checked; they are read only here, and the
 * block must name the pack's generation. Legacy packs (no
 * manifest is ever read) carry the same lines in a sidecar
 * <stream>.nq2gate, keyed by a content hash over the pack
 * geometry and the stored book bytes, computed from the
 * values the engine already holds. A missing or mismatched
 * block is not an error: the pack simply has no calibration,
 * and Auto falls back to expanded for it. */
static void gate_calib_load(const char *stream_path)
{
    g_gate_calibrated = 0;
    if (!g_use_packed || !stream_path) return;
    if (g_pk_version == 3) {
        char mpath[1200];
        snprintf(mpath, sizeof(mpath), "%s", stream_path);
        size_t L = strlen(mpath);
        if (L > 4 && strcmp(mpath + L - 4, ".nq2") == 0)
            strcpy(mpath + L - 4, ".nq2manifest");
        else
            strncat(mpath, ".nq2manifest", sizeof(mpath) - L - 1);
        FILE *mf = fopen(mpath, "r");
        if (!mf) return;
        char line[1200];
        /* Skip the header (3 lines) and the tensor lines, then
         * the manifest's own "end" terminator: the gate block
         * is appended after all of them, at the end of the
         * file. (pk_verify_manifest never reads past the
         * tensor lines, so the block cannot disturb it.) */
        int skip = 3 + g_ndir, ok = 1;
        for (int i = 0; i < skip && ok; i++)
            if (!fgets(line, sizeof(line), mf)) ok = 0;
        if (ok && fgets(line, sizeof(line), mf)
                && (strcmp(line, "end\n") == 0
                    || strcmp(line, "end") == 0))
            gate_parse_block(mf, g_pk_gen);
        fclose(mf);
        return;
    }
    /* Legacy: content key over geometry words and book bytes,
     * the same derivation the calibration harness records. */
    uint64_t h = PK_FNV_SEED;
    {
        uint8_t w[12], v[16];
        uint32_t words[3] = { g_pk_version, g_pk_block,
                              g_pk_entries };
        uint64_t nums[2] = { g_pk_nblocks, g_pk_book_floats };
        for (int i = 0; i < 3; i++)
            for (int b = 0; b < 4; b++)
                w[i * 4 + b] = (uint8_t)(words[i] >> (8 * b));
        for (int i = 0; i < 2; i++)
            for (int b = 0; b < 8; b++)
                v[i * 8 + b] = (uint8_t)(nums[i] >> (8 * b));
        h = pk_fnv(w, sizeof(w), h);
        h = pk_fnv(v, sizeof(v), h);
    }
    h = pk_fnv(g_pk_book, (size_t)g_pk_book_floats * sizeof(float),
               h);
    char spath[1200];
    snprintf(spath, sizeof(spath), "%s", stream_path);
    size_t L = strlen(spath);
    if (L > 4 && strcmp(spath + L - 4, ".nq2") == 0)
        strcpy(spath + L - 4, ".nq2gate");
    else
        strncat(spath, ".nq2gate", sizeof(spath) - L - 1);
    FILE *sf = fopen(spath, "r");
    if (!sf) return;
    gate_parse_block(sf, h);
    fclose(sf);
}
#endif /* !_WIN32 */

#ifndef _WIN32
/* The Precision Gate scorer (consumer thread only): decides
 * one layer's mark from the posture, the calibration and the
 * running drift estimate, in the plan's order... forced sets
 * first (posture, an escalation in force, eligibility), then
 * the static sensitivity map (a layer whose calibrated
 * increment runs above four times the median goes expanded),
 * then the drift projection across the lookahead window:
 * if consuming this window at the table path would carry the
 * projected final divergence past the calibrated budget, the
 * crossing layer and the rest of the window go expanded. */
static int gate_decide_layer(int l)
{
    if (g_posture == POSTURE_THINK || g_posture == POSTURE_EVIDENCE)
        return MARK_EXPAND;
    if (!g_gate_layer_ok || !g_gate_layer_ok[l])
        return MARK_EXPAND;
    if (g_posture == POSTURE_FAST)
        return MARK_DTC;
    /* Auto: an unmeasured pack is never gated. */
    if (!g_gate_calibrated)
        return MARK_EXPAND;
    if (g_gate_force_expand)
        return MARK_EXPAND;
    if (g_gate_delta[l] > 4.0 * g_gate_delta_median)
        return MARK_EXPAND;
    {
        double proj2 = g_gate_drift2;
        int hi = l + g_gate_lookahead;
        if (hi > M_L) hi = M_L;
        for (int k = l; k < hi; k++) {
            double d = g_gate_delta[k] * g_gate_scale;
            proj2 += d * d;
        }
        if (sqrt(proj2) > gate_budget_eff())
            return MARK_EXPAND;
    }
    return MARK_DTC;
}

/* Finalize marks for layers up to `upto` (inclusive) that are
 * still undecided this pass. Runs on the consumer only (the
 * scorer's state is the consumer's); each mark is published
 * with a release store, and the generation bump and door bump
 * make newly finalized EXPAND ordinals claimable by the lanes.
 * Marks are immutable once finalized within a pass. */
static void gate_finalize_consumer(int upto)
{
    if (upto > M_L - 1) upto = M_L - 1;
    int changed = 0;
    for (int l = g_gate_finalized_upto + 1; l <= upto; l++) {
        int mark = gate_decide_layer(l);
        int base = l * SLOTS_PER_LAYER;
        for (int s = 0; s < SLOTS_PER_LAYER; s++)
            HS_ST(&g_pipe.ord_mark[base + s], (uint8_t)mark);
        g_gate_finalized_upto = l;
        changed = 1;
    }
    if (changed) {
        HS_ADD(&g_pipe.list_generation, 1);
        pipe_bump();
    }
}

/* The consumer's layer-top entry. */
static void gate_finalize_through(int upto)
{
    if (upto <= g_gate_finalized_upto) return;
    gate_finalize_consumer(upto);
}

/* Tensor Handshake phase boundary, factored out of pipe_get
 * for the Precision Gate: the DTC consume entry point needs
 * the same rollover before it may read ordinal 0's mark, so
 * matmul_pipe calls this before routing, and pipe_get and
 * dtc_consume call it again; the consume-tick count makes
 * every call after the first in a pass a no-op. Runs on the
 * consumer only. The gate adds its per-pass step at the end,
 * inside the exclusive phase: marks return to UNDECIDED and
 * the scorer finalizes the first lookahead window before the
 * pass opens, so no lane can claim an ordinal on a stale
 * mark. */
static void pipe_pass_boundary(void)
{
    /* Tensor Handshake phase boundary (zero mutex,
     * 2026-11-26). "Is it time?": only if this pass consumed
     * anything since the last boundary. The boundary used to
     * run on every ordinal-0 entry point that asked for it, and
     * in the Precision Gate both matmul_pipe and the consume
     * path ask, so the second call found the fresh pass already
     * in motion (a lane holding ordinal 0's claim, the consume
     * tick still at the old pass's end) and faulted: "ordinal
     * 0's slot is still claimed at the boundary". Counting the
     * consume ticks makes the boundary exactly once per pass. */
    if (g_pipe.pass_consumed == 0) return;
    /* "Can I?": ask every worker to park. A worker acknowledges
     * only at its decision point, after it has published its
     * last result and released its claim or token, so once all
     * acknowledgements are in, no claim is held anywhere and
     * the consumer owns the whole list until it opens the next
     * pass. */
    uint32_t br = HS_LDX(&g_pipe.epoch_go) + 1;
    HS_ST(&g_pipe.boundary_req, br);
    pipe_bump();
    for (;;) {
        uint32_t d = hs_door_open(&g_pipe.door);
        int all = HS_LD(&g_pipe.ack_fetch) == br;
        for (int i = 0; all && i < g_pipe.n_lanes; i++)
            all = HS_LD(&g_pipe.ack_lane[i]) == br;
        if (all) break;
        hs_door_wait(&g_pipe.door, d);
    }
    /* Tensor Handshake phase boundary (stage 2): the pass
     * may roll over only when all three ticks agree the
     * prior pass drained... the fetch tick (fetched_upto),
     * the expansion tick (n_ready) and the consume tick
     * (layer_counter) all at the last ordinal. A
     * disagreement is a printed fault and a visible drain
     * through consumer self-service below, never a silent
     * wait; a claimed-but-unready ordinal at the boundary
     * would mean a worker broke the park protocol, and stops
     * the run loudly rather than consuming a half-drained
     * pipeline. */
    if (g_pipe.layer_counter >= 0
            && (g_pipe.fetched_upto != g_pipe.max_ord
                || g_pipe.n_ready != g_pipe.max_ord + 1
                || g_pipe.layer_counter != g_pipe.max_ord)) {
        fprintf(stderr, "tensor handshake: phase boundary "
                "fault at pass restart: ticks fetch %d, "
                "expand %d of %d ready, consume %d; the "
                "prior pass did not drain\n",
                g_pipe.fetched_upto, g_pipe.n_ready,
                g_pipe.max_ord + 1, g_pipe.layer_counter);
        for (int o = 0; o <= g_pipe.max_ord; o++) {
            if (g_pipe.have[pslot(o)] == o) continue;
            int slot_claimed = 0;
            for (int o2 = 0; o2 <= g_pipe.max_ord; o2++)
                if (pslot(o2) == pslot(o)
                        && g_pipe.ord_claim[o2] >= 0)
                    slot_claimed = 1;
            if (g_pipe.ord_claim[o] >= 0 || slot_claimed) {
                fprintf(stderr, "tensor handshake: ordinal "
                        "%d's slot is still claimed at the "
                        "boundary; cannot reconcile\n", o);
                exit(1);
            }
            g_pipe.ord_claim[o] = PIPE_MAX_LANES;
            g_pipe.ord_state[o] = ORD_EXPANDING;
            pipe_load(pslot(o), o, PIPE_ROLE_CONSUMER);
            pipe_mark_ready(o, PIPE_ROLE_CONSUMER, 1);
        }
        fprintf(stderr, "tensor handshake: boundary drain "
                "complete; the pass restarts on a drained "
                "pipeline\n");
    }
    /* A new forward pass restarts the ordinal walk: reset
     * the staging bookkeeping so the fetcher works in this
     * pass too (before this it silently died after pass 1,
     * leaving later passes to consumer self-service).
     * have[] stays: a marked buffer still holds its ordinal.
     * No fetch is in flight here (the fetcher is parked), so
     * nothing from the old pass can publish into the new one.
     * The list is rebuilt from have[]: resident ordinals stay
     * READY, everything else returns to IDLE, all claims
     * clear. */
    g_pipe.epoch++;
    g_pipe.fetched_upto = -1;
    for (int s = 0; s < g_pipe.slots; s++)
        g_pipe.st_owner[s] = -1;
    g_pipe.n_ready = 0;
    for (int o = 0; o <= g_pipe.max_ord; o++) {
        /* Residency that will actually be used (Archie
         * Round 4 P2-1): have[] still holds the
         * last slots ordinals of the previous pass in the
         * low slots, but the new pass walks from ordinal 0
         * and overwrites slot s at request s, so a held
         * ordinal o >= slots is evicted before its turn
         * comes. An ordinal survives the restart iff its own
         * request finds it: o < slots (which also covers
         * slots > max_ord, where every ordinal keeps its own
         * slot all pass). */
        int resident = (g_pipe.have[pslot(o)] == o)
                       && o < g_pipe.slots;
        g_pipe.ord_state[o] = resident ? ORD_READY : ORD_IDLE;
        g_pipe.ord_claim[o] = -1;
        g_pipe.ord_expand_counted[o] = resident ? 1 : 0;
        if (resident) g_pipe.n_ready++;
    }
    g_pipe.req_max = -1;    /* the new pass has requested nothing */
    g_pipe.list_generation++;
    /* Precision Gate: per-pass mark reset, then the first
     * lookahead window finalized before the pass opens. */
    for (int o = 0; o <= g_pipe.max_ord; o++)
        HS_ST(&g_pipe.ord_mark[o], (uint8_t)MARK_UNDECIDED);
    g_gate_finalized_upto = -1;
    gate_finalize_consumer(g_gate_lookahead);
    /* The new pass opens: every write above happens before
     * the release of epoch_go, and every parked worker acquires
     * it before it looks at the list again. */
    g_pipe.pass_consumed = 0;
    HS_ST(&g_pipe.epoch_go, br);
    pipe_bump();
}
#endif /* !_WIN32 */


/* Get the materialized weights for ordinal ord, and set the
 * prefetch of ord+1 in motion. The returned buffer is valid
 * until the request for ord + pipe slots. */
static const float *pipe_get(int ord)
{
    /* If no worker has produced ord, the consumer takes the
     * ordinal's claim (and, for a fused load, the fetch or load
     * token) with compare-and-swap before it loads, so no worker
     * can start the same load into the same slot. Zero mutex,
     * 2026-11-26: every wait below is a door wait on published
     * state. */
    double t_entry = pipe_now();
    int was_ready = 0;
    int self_loaded = 0;
#ifdef _WIN32
    was_ready = (HS_LD(&g_pipe.have[pslot(ord)]) == ord);
    if (was_ready) {
        g_pipe.stat_hits++;
        g_pipe.stat_slack_s += pipe_now() - g_pipe.ready_at[ord];
        g_pipe.stat_slack_n++;
    }
    for (;;) {
        uint32_t d = hs_door_open(&g_pipe.door);
        if (HS_LD(&g_pipe.have[pslot(ord)]) == ord) break;
        if (hs_cas_int(&g_pipe.busy, 0, 1)) {
            if (HS_LD(&g_pipe.have[pslot(ord)]) != ord) {
                self_loaded = 1;
                pipe_load(pslot(ord), ord, PIPE_ROLE_CONSUMER);
                pipe_mark_ready(ord, PIPE_ROLE_CONSUMER, 1);
            }
            HS_ST(&g_pipe.busy, 0);
            pipe_bump();
            break;
        }
        hs_door_wait(&g_pipe.door, d);
    }
    if (ord + 1 <= g_pipe.max_ord) {
        HS_ST(&g_pipe.want, ord + 1);
        pipe_bump();
    }
    if (!was_ready) {
        double wait_s = pipe_now() - t_entry;
        g_pipe.stat_miss++;
        g_pipe.stat_stall_s += wait_s;
        g_pipe.fx_miss_count[ord]++;
        g_pipe.fx_miss_wait_s[ord] += wait_s;
        if (self_loaded) g_pipe.fx_self_count[ord]++;
    }
#else
    if (ord == 0) pipe_pass_boundary();
    if (ord > HS_LDX(&g_pipe.req_max)) HS_ST(&g_pipe.req_max, ord);
    was_ready = (HS_LD(&g_pipe.have[pslot(ord)]) == ord);
    if (was_ready) {
        g_pipe.stat_hits++;
        g_pipe.stat_slack_s += pipe_now() - g_pipe.ready_at[ord];
        g_pipe.stat_slack_n++;
    }
    if (!was_ready) {
        /* Demand ord from the triforce: the unpacker takes it,
         * the fetcher is already staging it or one past it. The
         * wait runs until a worker has evaluated THIS demand
         * (demand_seen) or is busy on the pipeline, not merely
         * until the flags happen to read idle (Archie P1-1). */
        HS_ST(&g_pipe.want_unpack, ord);
        long my_demand = HS_ADD(&g_pipe.demand_posted, 1);
        pipe_bump();
        for (;;) {
            uint32_t d = hs_door_open(&g_pipe.door);
            if (HS_LD(&g_pipe.have[pslot(ord)]) == ord) break;
            if (HS_LD(&g_pipe.fetch_busy)
                    || HS_LD(&g_pipe.expand_busy_n) > 0
                    || HS_LD(&g_pipe.ord_claim[ord]) >= 0
                    || HS_LD(&g_pipe.demand_seen) < my_demand) {
                hs_door_wait(&g_pipe.door, d);
                continue;
            }
            /* Both stages idle, ord unclaimed and still not
             * ready: the consumer does the work itself, as the
             * fused pipe did... expanding staged sections if
             * it has them, loading fused if it does not. The
             * consumer claims the ordinal in the list first
             * (claim id PIPE_MAX_LANES, its own row), so the
             * lanes honour its work exactly as they honour
             * each other's: claim exclusivity has no
             * exceptions. */
            if (!hs_cas_int(&g_pipe.ord_claim[ord], -1, PIPE_MAX_LANES))
                continue;
            if (HS_LD(&g_pipe.have[pslot(ord)]) == ord) {
                HS_ST(&g_pipe.ord_claim[ord], -1);
                pipe_bump();
                break;
            }
            int staged = HS_LD(&g_pipe.st_owner[pslot(ord)]) == ord;
            if (!staged && !hs_cas_int(&g_pipe.fetch_busy, 0, 1)) {
                /* The fetcher took the token first: let it finish. */
                HS_ST(&g_pipe.ord_claim[ord], -1);
                pipe_bump();
                continue;
            }
            self_loaded = 1;
            HS_ADD(&g_pipe.expand_busy_n, 1);
            HS_ST(&g_pipe.ord_state[ord], (uint8_t)ORD_EXPANDING);
            HS_STX(&g_pipe.row_expand[PIPE_MAX_LANES], ord);
            HS_ADD(&g_pipe.list_generation, 1);
            if (staged) {
                pipe_expand(pslot(ord), ord, PIPE_ROLE_CONSUMER);
                pipe_mark_ready(ord, PIPE_ROLE_CONSUMER, 1);
            } else {
                pipe_load(pslot(ord), ord, PIPE_ROLE_CONSUMER);
                if (HS_LD(&g_pipe.fetched_upto) < ord)
                    HS_ST(&g_pipe.fetched_upto, ord);
                pipe_mark_ready(ord, PIPE_ROLE_CONSUMER, 1);
                HS_ST(&g_pipe.fetch_busy, 0);
            }
            HS_SUB(&g_pipe.expand_busy_n, 1);
            HS_STX(&g_pipe.row_expand[PIPE_MAX_LANES], -1);
            pipe_bump();
            break;
        }
    }
    if (ord + 1 <= g_pipe.max_ord) {
        HS_ST(&g_pipe.want_unpack, ord + 1);
        pipe_bump();
    }
    if (!was_ready) {
        double wait_s = pipe_now() - t_entry;
        g_pipe.stat_miss++;
        g_pipe.stat_stall_s += wait_s;
        g_pipe.fx_miss_count[ord]++;
        g_pipe.fx_miss_wait_s[ord] += wait_s;
        if (self_loaded) g_pipe.fx_self_count[ord]++;
    }
#endif
    /* Tensor Handshake: publish the committed position (the
     * consume tick). The consumer is the only writer; the
     * phase-boundary check compares it against the fetch and
     * expansion ticks, and pass_consumed is what makes the
     * next boundary due. */
    HS_ST(&g_pipe.layer_counter, ord);
    HS_STX(&g_pipe.row_consume, ord);
    g_pipe.pass_consumed++;
    return g_pipe.buf[pslot(ord)];
}

#ifndef _WIN32
/* DTC consume work: one position's table build and apply.
 * Positions partition across the pool; no accumulation is
 * ever split, so the partition cannot change any bit. */
typedef struct {
    const float *x;         /* [n][cols] activations */
    float *y;               /* [n][rows] out */
    const uint8_t *idx;     /* the stage's index bytes */
    const float *mn, *sc;   /* per-slot converted halves */
    const float *book;      /* the tensor's own book */
    uint32_t entries, block;
    uint64_t nseg;
    int n, rows, cols;
} DtcCtx;

static void dtc_work(void *vctx, int worker, int nw)
{
    DtcCtx *c = (DtcCtx *)vctx;
    int i0 = (int)((int64_t)c->n * worker / nw);
    int i1 = (int)((int64_t)c->n * (worker + 1) / nw);
    float *T = g_dtc_T[worker], *A = g_dtc_A[worker];
    for (int i = i0; i < i1; i++) {
        nqdc_build(c->book, c->entries, c->block,
                   c->x + (size_t)i * c->cols, c->nseg, T, A);
        nqdc_apply(c->idx, c->sc, c->mn, T, A, c->entries,
                   c->nseg, (uint64_t)c->rows,
                   c->y + (size_t)i * c->rows);
    }
}

/* Consume ordinal ord through the DTC table path: the
 * fetcher staged its packed sections (mode 1) and no lane
 * will ever expand it (the claim predicate honours the
 * mark), so the consumer claims the ordinal itself, computes
 * from the stage, and counts the preparation through
 * pipe_mark_ready exactly as an expansion is counted... the
 * phase-boundary ticks therefore agree whichever path each
 * ordinal took. A mark decides only which arithmetic runs,
 * never correctness: if the expanded buffer turns out to be
 * authoritative after all (restart residency, a lane's claim
 * placed before the mark finalized, the boundary drain),
 * this falls back to pipe_get and the dense matmul. */
static void dtc_consume(int ord, const float *x, float *y,
                        int n, int rows, int cols)
{
    const TEnt *t = ord_ent(ord);
    int slot = pslot(ord);
    double t_entry = pipe_now();
    int self_fetched = 0;
    int demanded = 0;
    long my_demand = 0;
    double wait_s;                  /* the wait for the stage only */
    if (ord == 0) pipe_pass_boundary();
    if (ord > HS_LDX(&g_pipe.req_max)) HS_ST(&g_pipe.req_max, ord);
    /* Zero mutex (2026-11-26): the consumer takes the ordinal's
     * claim with compare-and-swap before it touches the stage,
     * exactly as a lane would; a lane that won the claim first,
     * or a buffer already expanded, sends the ordinal down the
     * dense route instead. */
    for (;;) {
        uint32_t d = hs_door_open(&g_pipe.door);
        if (HS_LD(&g_pipe.have[slot]) == ord
                || HS_LD(&g_pipe.ord_claim[ord]) >= 0) {
            const float *w = pipe_get(ord);
            matmul(x, w, y, n, rows, cols);
            return;
        }
        if (HS_LD(&g_pipe.st_owner[slot]) == ord) {
            if (!hs_cas_int(&g_pipe.ord_claim[ord], -1, PIPE_MAX_LANES))
                continue;
            if (HS_LD(&g_pipe.have[slot]) == ord
                    || HS_LD(&g_pipe.st_owner[slot]) != ord) {
                HS_ST(&g_pipe.ord_claim[ord], -1);
                pipe_bump();
                continue;
            }
            break;                  /* staged, and the claim is ours */
        }
        if (!demanded) {
            /* Demand the staging with pipe_get's discipline
             * (Archie P1-1): the consumer may self-serve only
             * once a worker has had the chance to evaluate this
             * demand. The lanes never claim a DTC ordinal, so the
             * expansion count here speaks of other ordinals' work. */
            HS_ST(&g_pipe.want_unpack, ord);
            my_demand = HS_ADD(&g_pipe.demand_posted, 1);
            demanded = 1;
            pipe_bump();
            continue;
        }
        if (HS_LD(&g_pipe.fetch_busy)
                || HS_LD(&g_pipe.expand_busy_n) > 0
                || HS_LD(&g_pipe.demand_seen) < my_demand) {
            hs_door_wait(&g_pipe.door, d);
            continue;
        }
        /* Both stages idle and the ordinal never staged: the
         * consumer fetches it itself, holding the ordinal's
         * claim and the fetch token, then consumes the stage it
         * filled. */
        if (!hs_cas_int(&g_pipe.ord_claim[ord], -1, PIPE_MAX_LANES))
            continue;
        if (HS_LD(&g_pipe.have[slot]) == ord
                || HS_LD(&g_pipe.st_owner[slot]) == ord
                || !hs_cas_int(&g_pipe.fetch_busy, 0, 1)) {
            HS_ST(&g_pipe.ord_claim[ord], -1);
            pipe_bump();
            continue;
        }
        self_fetched = 1;
        HS_ST(&g_pipe.ord_state[ord], (uint8_t)ORD_EXPANDING);
        HS_STX(&g_pipe.row_expand[PIPE_MAX_LANES], ord);
        HS_ADD(&g_pipe.expand_busy_n, 1);
        HS_ADD(&g_pipe.list_generation, 1);
        pipe_fetch(slot, ord);
        HS_ST(&g_pipe.ord_state[ord], (uint8_t)ORD_STAGED);
        HS_ST(&g_pipe.st_owner[slot], ord);
        if (HS_LD(&g_pipe.fetched_upto) < ord)
            HS_ST(&g_pipe.fetched_upto, ord);
        HS_ST(&g_pipe.fetch_busy, 0);
        pipe_bump();
        break;
    }
    /* The wait for the stage ends here, before the table
     * compute below (Archie Round 5 P2-PG2): stall telemetry
     * charges only time spent waiting on the pipeline, never the
     * consume's own arithmetic. A stage found waiting on the
     * first look (no demand posted, no self-fetch) is a prefetch
     * hit, as pipe_get counts a buffer found ready. */
    wait_s = pipe_now() - t_entry;
    if (!self_fetched) {
        HS_ST(&g_pipe.ord_state[ord], (uint8_t)ORD_EXPANDING);
        HS_STX(&g_pipe.row_expand[PIPE_MAX_LANES], ord);
        HS_ADD(&g_pipe.expand_busy_n, 1);
        HS_ADD(&g_pipe.list_generation, 1);
    }
    Stage *s = &g_pipe.st[slot];
    double fetch_ms = s->fetch_ms;
    uint64_t nb = s->nb;
    /* The stage is stable while st_owner names this ordinal
     * and its buffer was never expanded (the fetcher's
     * slot-free predicate), and the consumer holds the
     * ordinal's claim, so the sections may be consumed in
     * place. The shape is the eligibility invariant from
     * pipe_init; anything else is a pipe fault, loudly. */
    if (s->mode != 1 || nb != t->nblocks || s->b0 != t->block0) {
        fprintf(stderr, "precision gate: ordinal %d staged in "
                "an unusable shape (mode %d, %llu of %llu "
                "blocks); refusing to consume it as DTC\n",
                ord, s->mode, (unsigned long long)nb,
                (unsigned long long)t->nblocks);
        exit(1);
    }
    {
        float *mn = g_pipe.dtc_mn[slot], *sc = g_pipe.dtc_sc[slot];
        for (uint64_t b = 0; b < nb; b++) {
            mn[b] = pk_half(s->fr + b * 2);
            sc[b] = pk_half(s->fr + nb * 2 + b * 2);
        }
        DtcCtx dc;
        dc.x = x; dc.y = y; dc.idx = s->idx; dc.mn = mn; dc.sc = sc;
        dc.book = pk_book_of(t);
        dc.entries = g_pk_entries;
        dc.block = pk_block_of(t);
        dc.nseg = (uint64_t)(cols / (int)dc.block);
        dc.n = n; dc.rows = rows; dc.cols = cols;
        pool_run(dtc_work, &dc);
    }
    HS_ST(&g_pipe.st_owner[slot], -1);   /* stage consumed; slot free */
    g_pipe.load_ms_last[PIPE_ROLE_CONSUMER] = fetch_ms;
    g_pipe.load_bytes_last[PIPE_ROLE_CONSUMER] =
        nb * (uint64_t)(g_pk_idxw + 4);
    pipe_mark_ready(ord, PIPE_ROLE_CONSUMER, 0);
    HS_SUB(&g_pipe.expand_busy_n, 1);
    HS_STX(&g_pipe.row_expand[PIPE_MAX_LANES], -1);
    {
        /* Hit/miss accounting, mirroring pipe_get: a stage
         * already waiting when the consumer asked is a hit; one
         * the consumer had to demand or fetch itself is a miss,
         * charged the staging wait only. */
        if (!demanded && !self_fetched) {
            g_pipe.stat_hits++;
        } else {
            g_pipe.stat_miss++;
            g_pipe.stat_stall_s += wait_s;
            g_pipe.fx_miss_count[ord]++;
            g_pipe.fx_miss_wait_s[ord] += wait_s;
            if (self_fetched) g_pipe.fx_self_count[ord]++;
        }
    }
    if (ord + 1 <= g_pipe.max_ord) HS_ST(&g_pipe.want_unpack, ord + 1);
    pipe_bump();
    /* The consume tick, as in pipe_get. */
    HS_ST(&g_pipe.layer_counter, ord);
    HS_STX(&g_pipe.row_consume, ord);
    g_pipe.pass_consumed++;
}
#endif /* !_WIN32 */

/* Route one pipe matmul by the ordinal's mark: EXPAND is
 * today's path, bit for bit (pipe_get, then the dense
 * matmul); DTC consumes the staged sections through the
 * kernel. On Windows the marks are all EXPAND, set at
 * pipe_init, and this is exactly the historical call. */
static void matmul_pipe(const float *x, int ord, float *y,
                        int n, int rows, int cols)
{
#ifndef _WIN32
    if (ord == 0) {
        /* Ordinal 0's mark is decided by the boundary's
         * finalize, so the rollover runs before the read. The
         * consume path's own ordinal-0 call is then a no-op. */
        pipe_pass_boundary();
    }
    if (HS_LDX(&g_pipe.ord_mark[ord]) == MARK_DTC) {
        dtc_consume(ord, x, y, n, rows, cols);
        return;
    }
#endif
    { const float *w = pipe_get(ord); matmul(x, w, y, n, rows, cols); }
}

static void pipe_report(void)
{
    /* Deterministic reduce after the join: per-role load
     * telemetry summed in fixed role order. */
    g_pipe.stat_load_ms = 0.0;
    g_pipe.stat_bytes = 0;
    for (int r = 0; r < PIPE_ROLES; r++) {
        g_pipe.stat_load_ms += g_pipe.stat_load_ms_r[r];
        g_pipe.stat_bytes += g_pipe.stat_bytes_r[r];
    }
    long total = g_pipe.stat_hits + g_pipe.stat_miss;
    if (total == 0) return;
    printf("prefetch telemetry: %ld requests, %ld hits (%.1f%%), "
           "%ld misses\n",
           total, g_pipe.stat_hits,
           100.0 * (double)g_pipe.stat_hits / (double)total,
           g_pipe.stat_miss);
    printf("  tensor handshake: layer counter %d of %d "
           "(consumer's committed ordinal)\n",
           g_pipe.layer_counter, g_pipe.max_ord);
    printf("  tensor handshake ticks: fetch %d, expand %d of "
           "%d ready, consume %d; prefetch list generation "
           "%ld\n",
           g_pipe.fetched_upto, g_pipe.n_ready,
           g_pipe.max_ord + 1, g_pipe.layer_counter,
           g_pipe.list_generation);
    printf("  consumer stall on misses: %.1f ms total; "
           "mean ready slack on hits: %.1f ms\n",
           g_pipe.stat_stall_s * 1000.0,
           g_pipe.stat_slack_n
               ? g_pipe.stat_slack_s * 1000.0 / g_pipe.stat_slack_n
               : 0.0);
    if (g_pipe.stat_load_ms > 0)
        printf("  load path (disc + reconstruct): %.1f ms for "
               "%.2f GB expanded, %.0f MB/sec\n",
               g_pipe.stat_load_ms,
               (double)g_pipe.stat_bytes / 1e9,
               (double)g_pipe.stat_bytes
                   / (g_pipe.stat_load_ms / 1000.0) / 1e6);
    if (g_forensics) {
        printf("  stall forensics by ordinal (misses only; self = "
               "consumer loaded it itself):\n");
        for (int o = 0; o <= g_pipe.max_ord; o++) {
            if (!g_pipe.fx_miss_count[o]) continue;
            const TEnt *t = ord_ent(o);
            printf("    ord %3d %-44s misses %4ld self %4ld "
                   "wait %9.1f ms (mean %7.1f) load mean %7.1f ms\n",
                   o, t ? t->name : "?",
                   g_pipe.fx_miss_count[o], g_pipe.fx_self_count[o],
                   g_pipe.fx_miss_wait_s[o] * 1000.0,
                   g_pipe.fx_miss_wait_s[o] * 1000.0
                       / (double)g_pipe.fx_miss_count[o],
                   g_pipe.fx_load_count[o]
                       ? g_pipe.fx_load_ms_sum[o]
                             / (double)g_pipe.fx_load_count[o]
                       : 0.0);
        }
    }
}

static void pipe_shutdown(void)
{
    /* Stop is published, the door bumps, and each worker sees it
     * at its next decision point (a parked worker too). */
    HS_ST(&g_pipe.stop, 1);
    pipe_bump();
#ifdef _WIN32
    hs_wait_count(&g_pipe.exited, 1u);
    CloseHandle(g_pipe.thread);
#else
    pthread_join(g_pipe.thread, NULL);
    for (int i = 0; i < g_pipe.n_lanes; i++)
        pthread_join(g_pipe.lanes[i], NULL);
#endif
}

/* ================= model ================= */

static int M_H = 3072;          /* hidden size */
static int M_L = 26;            /* layers */
static int M_HEADS = 32;
static int M_KVHEADS = 8;
static int M_HD = 128;          /* head dim */
static int M_FFN = 9216;
static int M_VOCAB = 131072;
static int M_ED = 16;           /* continuous embedding dim */
static float M_EPS = 1e-5f;

static float *g_norm_in = NULL;     /* [L][H] */
static float *g_norm_post = NULL;   /* [L][H] */
static float *g_norm_final = NULL;  /* [H] */
static float *g_in_linear = NULL;   /* [H][ED] */
static float *g_sc_linear = NULL;   /* [H][ED] */
static float *g_gamma_linear = NULL;/* [H][64] */
static float *g_E = NULL;           /* [VOCAB][ED], rows normalized */
static double g_gamma0 = -3.0, g_gamma1 = 6.0;

static float *load_small(const char *name)
{
    TEnt *t = dir_find(name);
    if (!t) { fprintf(stderr, "missing tensor %s\n", name); exit(1); }
    float *buf = (float *)malloc(t->count * sizeof(float));
    if (!buf || src_read_range(t, 0, t->count, buf) != 0) {
        fprintf(stderr, "cannot load %s\n", name);
        exit(1);
    }
    return buf;
}

/* load_small with an architecture conformance check (Codex
 * Astra review H4, 2026-11-25): the tensor's own span check
 * cannot catch a tensor whose declared count is smaller than
 * the extent its consumer reads (the layer-norm copy reads
 * M_H floats whatever the norm tensor declares). A checkpoint
 * whose shapes disagree with the derived architecture fails
 * loudly here, before any workspace is allocated or worker
 * started, instead of over-reading the allocation. */
static float *load_small_req(const char *name, uint64_t need)
{
    TEnt *t = dir_find(name);
    if (!t) { fprintf(stderr, "missing tensor %s\n", name); exit(1); }
    if (t->count < need) {
        fprintf(stderr, "tensor %s holds %llu elements, the "
                "architecture requires %llu\n", name,
                (unsigned long long)t->count,
                (unsigned long long)need);
        exit(1);
    }
    return load_small(name);
}

/* YaRN rotary tables, following the reference exactly:
 * factor = max_pos / original_max_pos (0.25 for this model),
 * correction ramp between beta_fast and beta_slow dims. */
static float *g_rope_cos = NULL, *g_rope_sin = NULL;  /* [maxpos][HD] */
static int g_rope_max = 0;

static void rope_init(int max_pos)
{
    const double theta = 1000000.0;
    const double orig_max = 16384.0;
    const double factor = 4096.0 / orig_max;
    const double beta_fast = 32.0, beta_slow = 1.0;
    int half = M_HD / 2;
    double inv_freq[64];
    double corr = 2.0 * log(theta);
    double low_d = M_HD * log(orig_max / (beta_fast * 2.0 * 3.141592653589793))
                   / corr;
    double high_d = M_HD * log(orig_max / (beta_slow * 2.0 * 3.141592653589793))
                    / corr;
    int low = (int)floor(low_d); if (low < 0) low = 0;
    int high = (int)ceil(high_d); if (high > M_HD - 1) high = M_HD - 1;
    for (int i = 0; i < half; i++) {
        double pos_freq = pow(theta, (double)(2 * i) / M_HD);
        double inv_extrap = 1.0 / pos_freq;
        double inv_interp = 1.0 / (factor * pos_freq);
        double ramp = ((double)i - low) / (high - low > 0.001 ? high - low : 0.001);
        if (ramp < 0.0) ramp = 0.0;
        if (ramp > 1.0) ramp = 1.0;
        double ef = 1.0 - ramp;
        inv_freq[i] = inv_interp * (1.0 - ef) + inv_extrap * ef;
    }
    g_rope_max = max_pos;
    g_rope_cos = (float *)malloc((size_t)max_pos * M_HD * sizeof(float));
    g_rope_sin = (float *)malloc((size_t)max_pos * M_HD * sizeof(float));
    if (!g_rope_cos || !g_rope_sin) { fprintf(stderr, "oom rope\n"); exit(1); }
    for (int p = 0; p < max_pos; p++) {
        for (int i = 0; i < half; i++) {
            double ang = inv_freq[i] * p;
            float c = (float)cos(ang), s = (float)sin(ang);
            g_rope_cos[(size_t)p * M_HD + i] = c;
            g_rope_cos[(size_t)p * M_HD + half + i] = c;
            g_rope_sin[(size_t)p * M_HD + i] = s;
            g_rope_sin[(size_t)p * M_HD + half + i] = s;
        }
    }
}

/* KV cache: bf16 storage, fp32 math. K is stored post-rotary,
 * exactly as the reference caches it. */
typedef struct {
    uint16_t **k, **v;      /* [layer][cap * kv_dim] */
    int cap;
    int len;                /* committed positions */
} KVCache;

static KVCache g_cache;

static void cache_init(int cap)
{
    int kv_dim = M_KVHEADS * M_HD;
    g_cache.k = (uint16_t **)malloc(sizeof(uint16_t *) * M_L);
    g_cache.v = (uint16_t **)malloc(sizeof(uint16_t *) * M_L);
    g_cache.cap = cap;
    g_cache.len = 0;
    for (int l = 0; l < M_L; l++) {
        g_cache.k[l] = (uint16_t *)malloc((size_t)cap * kv_dim * 2);
        g_cache.v[l] = (uint16_t *)malloc((size_t)cap * kv_dim * 2);
        if (!g_cache.k[l] || !g_cache.v[l]) {
            fprintf(stderr, "oom cache\n");
            exit(1);
        }
    }
}

static uint16_t f32_to_bf16(float f)
{
    /* round to nearest even, matching the reference's cast;
     * plain truncation biased every cached key and value down */
    uint32_t bits;
    memcpy(&bits, &f, 4);
    if ((bits & 0x7fffffffu) > 0x7f800000u) return 0x7fc0u; /* NaN -> bf16 quiet NaN; without this the rounding add carries into the sign bit and erases the NaN (Archie P0-5, guard from littlenemo/src/attn.c) */
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

/* ================= encoder ================= */

typedef struct {
    float *xn;        /* [maxn][H] normed input */
    float *q;         /* [maxn][HEADS*HD] */
    float *kcur;      /* [maxn][KVHEADS*HD] */
    float *vcur;      /* [maxn][KVHEADS*HD] */
    float *att;       /* [maxn][HEADS*HD] */
    float *gate;      /* [maxn][FFN] */
    float *up;        /* [maxn][FFN] */
    float *tmp;       /* [maxn][H] */
    int maxn;
} WS;

static WS ws_alloc(int maxn)
{
    WS w;
    w.maxn = maxn;
    w.xn = (float *)malloc((size_t)maxn * M_H * sizeof(float));
    w.q = (float *)malloc((size_t)maxn * M_HEADS * M_HD * sizeof(float));
    w.kcur = (float *)malloc((size_t)maxn * M_KVHEADS * M_HD * sizeof(float));
    w.vcur = (float *)malloc((size_t)maxn * M_KVHEADS * M_HD * sizeof(float));
    w.att = (float *)malloc((size_t)maxn * M_HEADS * M_HD * sizeof(float));
    w.gate = (float *)malloc((size_t)maxn * M_FFN * sizeof(float));
    w.up = (float *)malloc((size_t)maxn * M_FFN * sizeof(float));
    w.tmp = (float *)malloc((size_t)maxn * M_H * sizeof(float));
    if (!w.xn || !w.q || !w.kcur || !w.vcur || !w.att || !w.gate
            || !w.up || !w.tmp) {
        fprintf(stderr, "oom workspace\n");
        exit(1);
    }
    return w;
}

enum { ATT_WRITE = 0, ATT_READ = 1, ATT_BLOCK_ONLY = 2 };

typedef struct {
    const float *q;         /* [n][HEADS*HD], post-rotary */
    const float *kcur;      /* [n][KV] current block keys */
    const float *vcur;
    const int *pos;         /* [n] query positions */
    float *att;             /* [n][HEADS*HD] out */
    int n, layer, mode, clen;
} AttCtx;

static void att_work(void *vctx, int worker, int nw)
{
    AttCtx *c = (AttCtx *)vctx;
    int kv_dim = M_KVHEADS * M_HD;
    int groups = M_HEADS / M_KVHEADS;
    int total_qh = c->n * M_HEADS;
    int a0 = (int)((int64_t)total_qh * worker / nw);
    int a1 = (int)((int64_t)total_qh * (worker + 1) / nw);
    int total_keys = (c->mode == ATT_BLOCK_ONLY) ? c->n : c->clen + c->n;
    const uint16_t *ck = c->layer >= 0 ? g_cache.k[c->layer] : NULL;
    const uint16_t *cv = c->layer >= 0 ? g_cache.v[c->layer] : NULL;
    float *scores = (float *)malloc(sizeof(float) * total_keys);
    if (!scores) exit(1);
    for (int a = a0; a < a1; a++) {
        int i = a / M_HEADS;
        int h = a % M_HEADS;
        int kh = h / groups;
        const float *qv = c->q + (size_t)i * M_HEADS * M_HD + h * M_HD;
        double mx = -1e30;
        for (int j = 0; j < total_keys; j++) {
            int allowed;
            if (c->mode == ATT_WRITE) {
                int kpos = j < c->clen ? j : c->pos[j - c->clen];
                allowed = kpos <= c->pos[i];
            } else {
                allowed = 1;
            }
            if (!allowed) { scores[j] = -1e30f; continue; }
            double dot = 0.0;
            if (j < c->clen && c->mode != ATT_BLOCK_ONLY) {
                const uint16_t *kv = ck + (size_t)j * kv_dim + kh * M_HD;
                for (int d = 0; d < M_HD; d++)
                    dot += (double)qv[d] * bf16_to_float(kv[d]);
            } else {
                const float *kv = c->kcur
                    + (size_t)(j - (c->mode == ATT_BLOCK_ONLY ? 0 : c->clen))
                          * kv_dim + kh * M_HD;
                for (int d = 0; d < M_HD; d++) dot += (double)qv[d] * kv[d];
            }
            double s = dot * (1.0 / sqrt((double)M_HD));
            scores[j] = (float)s;
            if (s > mx) mx = s;
        }
        double sum = 0.0;
        for (int j = 0; j < total_keys; j++) {
            if (scores[j] <= -1e29f) { scores[j] = 0.0f; continue; }
            scores[j] = (float)exp((double)scores[j] - mx);
            sum += scores[j];
        }
        float *ov = c->att + (size_t)i * M_HEADS * M_HD + h * M_HD;
        for (int d = 0; d < M_HD; d++) ov[d] = 0.0f;
        if (sum <= 0.0) continue;
        double inv = 1.0 / sum;
        for (int j = 0; j < total_keys; j++) {
            if (scores[j] == 0.0f) continue;
            double p = scores[j] * inv;
            if (j < c->clen && c->mode != ATT_BLOCK_ONLY) {
                const uint16_t *vv = cv + (size_t)j * kv_dim + kh * M_HD;
                for (int d = 0; d < M_HD; d++)
                    ov[d] += (float)(p * bf16_to_float(vv[d]));
            } else {
                const float *vv = c->vcur
                    + (size_t)(j - (c->mode == ATT_BLOCK_ONLY ? 0 : c->clen))
                          * kv_dim + kh * M_HD;
                for (int d = 0; d < M_HD; d++) ov[d] += (float)(p * vv[d]);
            }
        }
    }
    free(scores);
}

/* hidden: [n][H] in/out (residual stream). pos: [n] positions. */
static void encoder_layer(float *hidden, const int *pos, int n, int layer,
                          int mode, WS *w)
{
    int qdim = M_HEADS * M_HD, kv_dim = M_KVHEADS * M_HD;
    int base = layer * SLOTS_PER_LAYER;
#ifndef _WIN32
    /* Precision Gate: the scorer finalizes this layer's mark
     * and the lookahead window's before any request is made,
     * so the claim window never reaches an undecided ordinal
     * the lanes would have to guess at. */
    gate_finalize_through(layer + g_gate_lookahead);
#endif
    rmsnorm(w->xn, hidden, g_norm_in + (size_t)layer * M_H, n, M_H, M_EPS);
    matmul_pipe(w->xn, base + SLOT_Q, w->q, n, qdim, M_H);
    matmul_pipe(w->xn, base + SLOT_K, w->kcur, n, kv_dim, M_H);
    matmul_pipe(w->xn, base + SLOT_V, w->vcur, n, kv_dim, M_H);
    /* rotary, in place on q and kcur */
    for (int i = 0; i < n; i++) {
        const float *cs = g_rope_cos + (size_t)pos[i] * M_HD;
        const float *sn = g_rope_sin + (size_t)pos[i] * M_HD;
        float *vecs[2] = { w->q + (size_t)i * qdim, NULL };
        vecs[1] = w->kcur + (size_t)i * kv_dim;
        int dims[2] = { qdim, kv_dim };
        for (int v = 0; v < 2; v++) {
            float *x = vecs[v];
            for (int off = 0; off < dims[v]; off += M_HD) {
                float *xh = x + off;
                float hold[64];
                for (int d = 0; d < M_HD / 2; d++) hold[d] = xh[d];
                for (int d = 0; d < M_HD / 2; d++) {
                    float x1 = hold[d], x2 = xh[M_HD / 2 + d];
                    xh[d] = x1 * cs[d] - x2 * sn[d];
                    xh[M_HD / 2 + d] = x2 * cs[M_HD / 2 + d]
                                       + x1 * sn[M_HD / 2 + d];
                }
            }
        }
        /* Llama-4 style query scaling (1.0 below 16384) */
        double sc = 1.0 + 0.1 * log(1.0 + floor(pos[i] / 16384.0));
        if (sc != 1.0) {
            float *x = w->q + (size_t)i * qdim;
            for (int d = 0; d < qdim; d++) x[d] = (float)(x[d] * sc);
        }
    }
    if (mode == ATT_WRITE) {
        int kv_dim2 = kv_dim;
        for (int i = 0; i < n; i++) {
            uint16_t *kd = g_cache.k[layer] + (size_t)(g_cache.len + i) * kv_dim2;
            uint16_t *vd = g_cache.v[layer] + (size_t)(g_cache.len + i) * kv_dim2;
            for (int d = 0; d < kv_dim2; d++) {
                kd[d] = f32_to_bf16(w->kcur[(size_t)i * kv_dim2 + d]);
                vd[d] = f32_to_bf16(w->vcur[(size_t)i * kv_dim2 + d]);
            }
        }
    }
    AttCtx ac;
    ac.q = w->q; ac.kcur = w->kcur; ac.vcur = w->vcur; ac.pos = pos;
    ac.att = w->att; ac.n = n; ac.layer = layer; ac.mode = mode;
    ac.clen = g_cache.len;
    pool_run(att_work, &ac);
    matmul_pipe(w->att, base + SLOT_O, w->tmp, n, M_H, qdim);
    for (int i = 0; i < n * M_H; i++) hidden[i] += w->tmp[i];

    rmsnorm(w->xn, hidden, g_norm_post + (size_t)layer * M_H, n, M_H, M_EPS);
    matmul_pipe(w->xn, base + SLOT_GATE, w->gate, n, M_FFN, M_H);
    matmul_pipe(w->xn, base + SLOT_UP, w->up, n, M_FFN, M_H);
    for (int i = 0; i < n * M_FFN; i++)
        w->gate[i] = silu_f(w->gate[i]) * w->up[i];
    matmul_pipe(w->gate, base + SLOT_DOWN, w->tmp, n, M_H, M_FFN);
    for (int i = 0; i < n * M_H; i++) hidden[i] += w->tmp[i];
#ifndef _WIN32
    /* Precision Gate drift accounting: the layer's mark is
     * the path it was scored for. A layer consumed expanded
     * leaves the state the exact quantized state, so the
     * estimate restarts from zero there; a layer consumed at
     * the table adds its calibrated increment, scaled by the
     * resync-anchored factor. Counting follows the marks,
     * not the fault fallbacks, so the estimate stays a pure
     * function of the run's decisions. */
    if (g_use_packed && (g_posture == POSTURE_AUTO
                         || g_posture == POSTURE_FAST)) {
        g_gate_layers_total++;
        if (g_pipe.ord_mark[base] == MARK_DTC) {
            g_gate_layers_dtc++;
            if (g_posture == POSTURE_AUTO && g_gate_calibrated) {
                double d = g_gate_delta[layer] * g_gate_scale;
                g_gate_drift2 += d * d;
            }
        } else if (g_posture == POSTURE_AUTO) {
            g_gate_drift2 = 0.0;
        }
    }
#endif
}

/* Full backbone pass. ATT_WRITE commits KV and advances the
 * cache length after the pass. Returns hidden states after the
 * final norm in `hidden`. */
static void encoder_forward(float *hidden, const int *pos, int n, int mode,
                            WS *w)
{
    for (int l = 0; l < M_L; l++)
        encoder_layer(hidden, pos, n, l, mode, w);
    rmsnorm(w->tmp, hidden, g_norm_final, n, M_H, M_EPS);
    memcpy(hidden, w->tmp, (size_t)n * M_H * sizeof(float));
    if (mode == ATT_WRITE) g_cache.len += n;
}

/* ================= diffusion ================= */

static TEnt *g_head_ent = NULL;
static TEnt *g_embed_ent = NULL;

/* Diffusion-head logits for all B hidden rows at once, streaming
 * the head matrix in row tiles: each tile is read from disc once
 * and multiplied against every position. The full matrix is
 * never materialized, and the head crosses the disc-to-CPU path
 * once per pass instead of once per position. */
static void head_all(const float *h, int B, float *logits)
{
    const int TILE = 2048;
    float *tile = (float *)malloc((size_t)TILE * M_H * sizeof(float));
    float *tmp = (float *)malloc((size_t)B * TILE * sizeof(float));
    if (!tile || !tmp) { fprintf(stderr, "oom head tile\n"); exit(1); }
    for (int r0 = 0; r0 < M_VOCAB; r0 += TILE) {
        int rows = M_VOCAB - r0 < TILE ? M_VOCAB - r0 : TILE;
        if (src_read_range(g_head_ent, (uint64_t)r0 * M_H,
                           (uint64_t)rows * M_H, tile) != 0) {
            fprintf(stderr, "head read failed\n");
            exit(1);
        }
        matmul(h, tile, tmp, B, rows, M_H);
        for (int i = 0; i < B; i++)
            for (int j = 0; j < rows; j++)
                logits[(size_t)i * M_VOCAB + r0 + j]
                    = tmp[(size_t)i * rows + j];
    }
    free(tile);
    free(tmp);
}

/* softmax(logits) @ E for one position: the model's clean
 * embedding estimate. */
static void mean_from_logits(const float *logits, float *mean_out)
{
    double mx = -1e30;
    for (int v = 0; v < M_VOCAB; v++)
        if (logits[v] > mx) mx = logits[v];
    double sum = 0.0;
    /* Accumulate in double: summing 131,072 probability-weighted
     * rows into a float loses real digits off the mean. */
    double acc[64];
    for (int d = 0; d < M_ED; d++) acc[d] = 0.0;
    for (int v = 0; v < M_VOCAB; v++) {
        double p = exp((double)logits[v] - mx);
        sum += p;
        const float *erow = g_E + (size_t)v * M_ED;
        for (int d = 0; d < M_ED; d++) acc[d] += p * erow[d];
    }
    for (int d = 0; d < M_ED; d++) mean_out[d] = (float)(acc[d] / sum);
}

/* One denoiser evaluation, mirroring the reference
 * _denoise_block. z is [B][ED] in double. Fills the conditional
 * logits [B][VOCAB], the guided estimate x_guided [B][ED], and
 * the raw conditional mean cond_mean [B][ED]. */
static void denoise(const double *z, double gamma, const float *selfcond,
                    double guidance, const int *bpos, int B,
                    float *logits, float *x_guided, float *cond_mean,
                    WS *w)
{
#ifndef _WIN32
    /* Precision Gate, per-evaluation state: Auto on a
     * calibrated pack counts the evaluation, decides whether
     * this one is a shadow-resync evaluation, and arms the
     * input save the resync and the margin gate both need.
     * The force flag starts down for every evaluation. */
    int gate_auto = (g_posture == POSTURE_AUTO && g_gate_calibrated);
    int gate_final = g_gate_final_step;
    int gate_resync_now = 0;
    g_gate_force_expand = 0;
    if (gate_auto) {
        g_gate_evals++;
        if (g_gate_resync > 0
                && ++g_gate_since_resync >= g_gate_resync)
            gate_resync_now = 1;
    }
    float *h_saved = NULL, *h_fast = NULL;
    if (gate_auto && (gate_final || gate_resync_now)) {
        h_saved = (float *)malloc((size_t)B * M_H * sizeof(float));
        h_fast = (float *)malloc((size_t)B * M_H * sizeof(float));
        if (!h_saved || !h_fast) {
            fprintf(stderr, "oom denoise\n");
            exit(1);
        }
    }
#endif
    double alpha_sq = 1.0 / (1.0 + exp(gamma));
    double sigma_sq = 1.0 / (1.0 + exp(-gamma));
    double alpha = sqrt(alpha_sq);
    double zvar = alpha_sq / M_ED + sigma_sq;
    double zscale = 1.0 / sqrt(zvar);

    float *xin = (float *)malloc((size_t)B * M_ED * sizeof(float));
    float *h = (float *)malloc((size_t)B * M_H * sizeof(float));
    float *hu = (float *)malloc((size_t)B * M_H * sizeof(float));
    float *tmp = (float *)malloc((size_t)B * M_H * sizeof(float));
    if (!xin || !h || !hu || !tmp) { fprintf(stderr, "oom denoise\n"); exit(1); }
    for (int i = 0; i < B * M_ED; i++) xin[i] = (float)(z[i] * zscale);
    matmul(xin, g_in_linear, h, B, M_H, M_ED);
    memcpy(hu, h, (size_t)B * M_H * sizeof(float));
    if (selfcond) {
        float *scin = (float *)malloc((size_t)B * M_ED * sizeof(float));
        if (!scin) { fprintf(stderr, "oom denoise\n"); exit(1); }
        double s = sqrt((double)M_ED);
        for (int i = 0; i < B * M_ED; i++) scin[i] = (float)(selfcond[i] * s);
        matmul(scin, g_sc_linear, tmp, B, M_H, M_ED);
        for (int i = 0; i < B * M_H; i++) h[i] += tmp[i];
        free(scin);
    }
    /* gamma (log-SNR) embedding: sin/cos of exp-spaced freqs */
    {
        float ge[64];
        float *grow = (float *)malloc((size_t)M_H * sizeof(float));
        if (!grow) { fprintf(stderr, "oom denoise\n"); exit(1); }
        for (int j = 0; j < 32; j++) {
            double fr = exp(-5.0 + (10.0 * j) / 31.0);
            ge[j] = (float)sin(fr * gamma);
            ge[32 + j] = (float)cos(fr * gamma);
        }
        matmul(ge, g_gamma_linear, grow, 1, M_H, 64);
        for (int i = 0; i < B; i++)
            for (int d = 0; d < M_H; d++) {
                h[(size_t)i * M_H + d] += grow[d];
                hu[(size_t)i * M_H + d] += grow[d];
            }
        free(grow);
    }
    /* The output prior (alpha / sigma_sq) * (z @ E^T) does not
     * depend on the hidden states, so it is computed once and
     * added to both the conditional and unconditional logits. */
    double prior = alpha / sigma_sq;
    float *prior_dot = (float *)malloc((size_t)B * M_VOCAB * sizeof(float));
    float *zf = (float *)malloc((size_t)B * M_ED * sizeof(float));
    if (!prior_dot || !zf) { fprintf(stderr, "oom denoise\n"); exit(1); }
    for (int i = 0; i < B * M_ED; i++) zf[i] = (float)z[i];
    matmul(zf, g_E, prior_dot, B, M_VOCAB, M_ED);
    for (uint64_t i = 0; i < (uint64_t)B * M_VOCAB; i++)
        prior_dot[i] = (float)(prior * prior_dot[i]);
    free(zf);

    /* conditional pass: block attends the clean prefix cache */
#ifndef _WIN32
    if (h_saved) memcpy(h_saved, h, (size_t)B * M_H * sizeof(float));
#endif
    encoder_forward(h, bpos, B, ATT_READ, w);
#ifndef _WIN32
    if (gate_resync_now) {
        /* Shadow resync (plan 2.2): the same input run again
         * expanded. The measured divergence between the two
         * outputs re-anchors the drift scale (clamped, so one
         * odd evaluation cannot whip the gate), the run
         * proceeds on the expanded output, and the drift
         * estimate restarts from that known state. */
        memcpy(h_fast, h, (size_t)B * M_H * sizeof(float));
        memcpy(h, h_saved, (size_t)B * M_H * sizeof(float));
        g_gate_force_expand = 1;
        encoder_forward(h, bpos, B, ATT_READ, w);
        {
            double num = 0.0, den = 0.0;
            for (int i = 0; i < B * M_H; i++) {
                double d = (double)h_fast[i] - h[i];
                num += d * d;
                den += (double)h[i] * h[i];
            }
            double measured = den > 0.0 ? sqrt(num / den) : 0.0;
            double predicted = sqrt(g_gate_drift2);
            if (predicted > 0.0) {
                double ns = measured / predicted;
                if (ns < 0.25) ns = 0.25;
                if (ns > 4.0) ns = 4.0;
                g_gate_scale = ns;
            }
            g_gate_drift2 = 0.0;
            g_gate_since_resync = 0;
            g_gate_resyncs++;
            gate_logf("resync eval %ld: measured %.6e "
                      "predicted %.6e scale %.4f\n",
                      g_gate_evals, measured, predicted,
                      g_gate_scale);
        }
    }
#endif
    head_all(h, B, logits);
    for (int i = 0; i < B; i++) {
        float *lg = logits + (size_t)i * M_VOCAB;
        const float *pd = prior_dot + (size_t)i * M_VOCAB;
        for (int v = 0; v < M_VOCAB; v++) lg[v] += pd[v];
        mean_from_logits(lg, cond_mean + (size_t)i * M_ED);
    }
#ifndef _WIN32
    if (gate_final && gate_auto && !g_gate_force_expand) {
        /* Margin gate (plan 2.1): on the step that commits
         * tokens, a position whose top two logits sit inside
         * the calibrated tie epsilon can flip under the table
         * path's reassociation. Escalate the evaluation to
         * expanded and recompute before anything consumes
         * these logits; the dump below then records the
         * escalated logits, which is what the sampler eats.
         * (When a resync already forced this evaluation, the
         * logits are the exact ones and there is nothing
         * further to escalate to.) */
        double min_margin = 1e30;
        for (int i = 0; i < B; i++) {
            const float *lg = logits + (size_t)i * M_VOCAB;
            float t1 = -1e30f, t2 = -1e30f;
            for (int v = 0; v < M_VOCAB; v++) {
                if (lg[v] > t1) { t2 = t1; t1 = lg[v]; }
                else if (lg[v] > t2) { t2 = lg[v]; }
            }
            if ((double)t1 - (double)t2 < min_margin)
                min_margin = (double)t1 - (double)t2;
        }
        if (min_margin < gate_eps_eff()) {
            g_gate_force_expand = 1;
            g_gate_escalations++;
            gate_logf("escalate eval %ld: min margin %.6e "
                      "< eps %.6e\n",
                      g_gate_evals, min_margin, gate_eps_eff());
            memcpy(h, h_saved, (size_t)B * M_H * sizeof(float));
            encoder_forward(h, bpos, B, ATT_READ, w);
            head_all(h, B, logits);
            for (int i = 0; i < B; i++) {
                float *lg = logits + (size_t)i * M_VOCAB;
                const float *pd = prior_dot + (size_t)i * M_VOCAB;
                for (int v = 0; v < M_VOCAB; v++) lg[v] += pd[v];
                mean_from_logits(lg, cond_mean + (size_t)i * M_ED);
            }
        }
    }
#endif
    /* Lab build, behavioral gate (Neon, 2026-11-24): dump the
     * first evaluation's conditional logits, so a raw run and a
     * packed run over the same fixed token IDs can be compared
     * (top-1 / top-5 agreement, cosine) with no tokenizer in the
     * loop. The dump is the conditional logits exactly as the
     * sampler consumed them, prior included. */
    if (g_logits_dump && (g_logits_dump_all || !g_logits_dumped)) {
        FILE *lf = fopen(g_logits_dump,
                         g_logits_dump_count ? "ab" : "wb");
        if (!lf) { fprintf(stderr, "logits dump: cannot open\n"); exit(1); }
        uint32_t hdr[3] = { 0x31474F4Cu, (uint32_t)B, (uint32_t)M_VOCAB };
        if (fwrite(hdr, sizeof(uint32_t), 3, lf) != 3
                || fwrite(logits, sizeof(float), (size_t)B * M_VOCAB, lf)
                       != (size_t)B * M_VOCAB) {
            fprintf(stderr, "logits dump: short write\n"); exit(1); }
        fclose(lf);
        g_logits_dumped = 1;
        g_logits_dump_count++;
    }
    if (guidance != 1.0) {
        float *logits_u = (float *)malloc((size_t)B * M_VOCAB * sizeof(float));
        float *x_u = (float *)malloc((size_t)B * M_ED * sizeof(float));
        if (!logits_u || !x_u) { fprintf(stderr, "oom denoise\n"); exit(1); }
        encoder_forward(hu, bpos, B, ATT_BLOCK_ONLY, w);
        head_all(hu, B, logits_u);
        for (int i = 0; i < B; i++) {
            float *lg = logits_u + (size_t)i * M_VOCAB;
            const float *pd = prior_dot + (size_t)i * M_VOCAB;
            for (int v = 0; v < M_VOCAB; v++) lg[v] += pd[v];
            mean_from_logits(lg, x_u + (size_t)i * M_ED);
        }
        for (int i = 0; i < B * M_ED; i++)
            x_guided[i] = (float)(x_u[i] + guidance * (cond_mean[i] - x_u[i]));
        free(logits_u);
        free(x_u);
    } else {
        memcpy(x_guided, cond_mean, (size_t)B * M_ED * sizeof(float));
    }
    free(prior_dot);
    free(xin); free(h); free(hu); free(tmp);
#ifndef _WIN32
    /* The force flag's scope is this call: the unconditional
     * pass above ran under it; nothing after this call may. */
    g_gate_force_expand = 0;
    free(h_saved); free(h_fast);
#endif
}

static void encode_clean(const int *ids, int n, const int *pos, WS *w)
{
    float *hidden = (float *)malloc((size_t)n * M_H * sizeof(float));
    if (!hidden) { fprintf(stderr, "oom encode\n"); exit(1); }
    for (int i = 0; i < n; i++) {
        if (src_read_range(g_embed_ent, (uint64_t)ids[i] * M_H, M_H,
                           hidden + (size_t)i * M_H) != 0) {
            fprintf(stderr, "embed read failed\n");
            exit(1);
        }
    }
    encoder_forward(hidden, pos, n, ATT_WRITE, w);
    free(hidden);
}

/* ================= main ================= */

static int cmp_str(const void *a, const void *b)
{
    return strcmp((const char *)a, (const char *)b);
}

/* One environment layer of the pipe configuration: read an
 * integer variable strictly (the whole value must be the
 * number), or fail loudly. Returns 1 when the variable was
 * set. The command line is parsed after this and wins. */
static int pipe_env_int(const char *name, int *out)
{
    const char *v = getenv(name);
    if (!v || !*v) return 0;
    char *end = NULL;
    long n = strtol(v, &end, 10);
    if (end == v || *end != '\0') {
        fprintf(stderr, "pipe: %s '%s' is not a number\n", name, v);
        exit(1);
    }
    *out = (int)n;
    return 1;
}

/* Strict CLI integer, the same standard as pipe_env_int
 * (Sol M3): the whole argument must be the number, so a
 * mistyped value fails loudly instead of atoi's silent 0. */
static void pipe_cli_int(const char *flag, const char *v, int *out)
{
    char *end = NULL;
    long n = strtol(v, &end, 10);
    if (end == v || *end != '\0') {
        fprintf(stderr, "pipe: %s '%s' is not a number\n", flag, v);
        exit(1);
    }
    *out = (int)n;
}

int main(int argc, char **argv)
{
    const char *dir = NULL;
    const char *pk_stream = NULL, *pk_book = NULL;
    const char *ids_arg = NULL;
    int max_new = 64, steps = 32, block = 32;
    int sampler_ddim = 1, schedule_lognsnr = 1;
    double guidance = 3.0, score_temp = 0.3;
    uint64_t seed = 1234;
    int threads = 0;

    /* Environment layer (below the CLI, which overrides it). */
    if (pipe_env_int("NQ2_PIPE_SLOTS", &g_pipe_slots)) {
        g_slots_src = "env";
        g_slots_explicit = 1;
    }
    if (pipe_env_int("NQ2_FETCH_AHEAD", &g_fetch_ahead))
        g_fetch_src = "env";
    if (pipe_env_int("NQ2_PREFETCH_DIST", &g_prefetch_dist))
        g_dist_src = "env";
    if (pipe_env_int("NQ2_EXPAND_LANES", &g_expand_lanes)) {
        g_lanes_src = "env";
        g_lanes_explicit = 1;
    }
    {
        const char *pv = getenv("NQ2_POSTURE");
        if (pv && *pv) {
            if (strcmp(pv, "evidence") == 0) g_posture = POSTURE_EVIDENCE;
            else if (strcmp(pv, "auto") == 0) g_posture = POSTURE_AUTO;
            else if (strcmp(pv, "fast") == 0) g_posture = POSTURE_FAST;
            else if (strcmp(pv, "think-harder") == 0) g_posture = POSTURE_THINK;
            else {
                fprintf(stderr, "precision gate: NQ2_POSTURE "
                        "'%s' is not a posture (evidence, auto, "
                        "fast, think-harder)\n", pv);
                exit(1);
            }
            g_posture_src = "env";
        }
    }
    if (pipe_env_int("NQ2_GATE_LOOKAHEAD", &g_gate_lookahead))
        g_gate_lookahead_src = "env";
    if (pipe_env_int("NQ2_GATE_RESYNC", &g_gate_resync))
        g_gate_resync_src = "env";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--dir") == 0 && i + 1 < argc) dir = argv[++i];
        else if (strcmp(argv[i], "--packed") == 0 && i + 2 < argc) {
            pk_stream = argv[i + 1];
            pk_book = argv[i + 2];
            i += 2;
        }
        else if (strcmp(argv[i], "--ids") == 0 && i + 1 < argc) ids_arg = argv[++i];
        else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) max_new = atoi(argv[++i]);
        else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--block") == 0 && i + 1 < argc) block = atoi(argv[++i]);
        else if (strcmp(argv[i], "--sampler") == 0 && i + 1 < argc) sampler_ddim = strcmp(argv[++i], "ddpm") != 0;
        else if (strcmp(argv[i], "--schedule") == 0 && i + 1 < argc) schedule_lognsnr = strcmp(argv[++i], "edm_sigma") != 0;
        else if (strcmp(argv[i], "--guidance") == 0 && i + 1 < argc) guidance = atof(argv[++i]);
        else if (strcmp(argv[i], "--score-temp") == 0 && i + 1 < argc) score_temp = atof(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--forensics") == 0) g_forensics = 1;
        else if (strcmp(argv[i], "--pipe-slots") == 0 && i + 1 < argc) { pipe_cli_int("--pipe-slots", argv[++i], &g_pipe_slots); g_slots_src = "cli"; g_slots_explicit = 1; }
        else if (strcmp(argv[i], "--fetch-ahead") == 0 && i + 1 < argc) { pipe_cli_int("--fetch-ahead", argv[++i], &g_fetch_ahead); g_fetch_src = "cli"; }
        else if (strcmp(argv[i], "--prefetch-dist") == 0 && i + 1 < argc) { pipe_cli_int("--prefetch-dist", argv[++i], &g_prefetch_dist); g_dist_src = "cli"; }
        else if (strcmp(argv[i], "--expand-lanes") == 0 && i + 1 < argc) { pipe_cli_int("--expand-lanes", argv[++i], &g_expand_lanes); g_lanes_src = "cli"; g_lanes_explicit = 1; }
        else if (strcmp(argv[i], "--posture") == 0 && i + 1 < argc) {
            const char *pv = argv[++i];
            if (strcmp(pv, "evidence") == 0) g_posture = POSTURE_EVIDENCE;
            else if (strcmp(pv, "auto") == 0) g_posture = POSTURE_AUTO;
            else if (strcmp(pv, "fast") == 0) g_posture = POSTURE_FAST;
            else if (strcmp(pv, "think-harder") == 0) g_posture = POSTURE_THINK;
            else {
                fprintf(stderr, "precision gate: --posture "
                        "'%s' is not a posture (evidence, auto, "
                        "fast, think-harder)\n", pv);
                return 1;
            }
            g_posture_src = "cli";
        }
        else if (strcmp(argv[i], "--gate-lookahead") == 0 && i + 1 < argc) { pipe_cli_int("--gate-lookahead", argv[++i], &g_gate_lookahead); g_gate_lookahead_src = "cli"; }
        else if (strcmp(argv[i], "--gate-resync") == 0 && i + 1 < argc) { pipe_cli_int("--gate-resync", argv[++i], &g_gate_resync); g_gate_resync_src = "cli"; }
        else if (strcmp(argv[i], "--gate-eps") == 0 && i + 1 < argc) {
            const char *sv = argv[++i];
            char *end = NULL;
            double v = strtod(sv, &end);
            if (end == sv || *end != '\0' || !(v > 0.0)) {
                fprintf(stderr, "precision gate: --gate-eps "
                        "'%s' is not a positive number\n", sv);
                return 1;
            }
            g_gate_eps_override = v;
        }
        else if (strcmp(argv[i], "--gate-budget") == 0 && i + 1 < argc) {
            const char *sv = argv[++i];
            char *end = NULL;
            double v = strtod(sv, &end);
            if (end == sv || *end != '\0' || !(v > 0.0)) {
                fprintf(stderr, "precision gate: --gate-budget "
                        "'%s' is not a positive number\n", sv);
                return 1;
            }
            g_gate_budget_override = v;
        }
        else if (strcmp(argv[i], "--gate-log") == 0 && i + 1 < argc) g_gate_log_path = argv[++i];
        else if (strcmp(argv[i], "--logits-dump") == 0 && i + 1 < argc) g_logits_dump = argv[++i];
        else if (strcmp(argv[i], "--logits-dump-all") == 0 && i + 1 < argc) { g_logits_dump = argv[++i]; g_logits_dump_all = 1; }
        else {
            fprintf(stderr, "unknown or incomplete option: %s\n", argv[i]);
            return 1;
        }
    }
    if (steps < 1) steps = 1;
    if (g_gate_lookahead < 1 || g_gate_lookahead > 4) {
        fprintf(stderr, "precision gate: --gate-lookahead must "
                "be in [1, 4]\n");
        return 1;
    }
    if (g_gate_resync < 0) {
        fprintf(stderr, "precision gate: --gate-resync must "
                "be >= 0\n");
        return 1;
    }
    if (block < 1) block = 1;
    if (block > 4096) {
        fprintf(stderr, "warning: block clamped to 4096\n");
        block = 4096;
    }
    if (!dir || !ids_arg) {
        fprintf(stderr,
            "usage: nq2-infer --dir MODEL_DIR [--packed STREAM BOOK]\n"
            "       --ids 1,25,882 [--tokens 64] [--steps 32]\n"
            "       [--block 32] [--sampler ddim|ddpm]\n"
            "       [--schedule log_snr|edm_sigma] [--guidance 3.0]\n"
            "       [--score-temp 0.3] [--seed 1234] [--threads N]\n"
            "       [--posture evidence|auto|fast|think-harder]\n"
            "       [--gate-lookahead 1..4] [--gate-resync N]\n"
            "       [--gate-eps X] [--gate-budget X] [--gate-log FILE]\n");
        return 1;
    }

    /* directory: every .safetensors in the model dir, sorted so
     * shard order matches the quantizer's packing order */
    {
        char names[16][512];
        int nn = 0;
        /* opendir/readdir, not popen("ls ...") (Archie P0-3): no
         * shell in the path, so --dir cannot inject commands, and
         * the same code runs on Windows (MinGW carries dirent.h),
         * which also retires the old _WIN32 early exit that refused
         * to run there. loader.c scans its directory the same way. */
        DIR *dd = opendir(dir);
        if (!dd) { fprintf(stderr, "cannot list %s\n", dir); return 1; }
        struct dirent *de;
        while ((de = readdir(dd)) != NULL && nn < MAX_SHARDS) {
            size_t L = strlen(de->d_name);
            if (L > 12 && strcmp(de->d_name + L - 12, ".safetensors") == 0) {
                snprintf(names[nn], 512, "%s", de->d_name);
                nn++;
            }
        }
        closedir(dd);
        qsort(names, nn, 512, cmp_str);
        if (nn == 0) { fprintf(stderr, "no safetensors in %s\n", dir); return 1; }
        for (int s = 0; s < nn; s++) {
            char path[1024];
            snprintf(path, sizeof(path), "%s/%s", dir, names[s]);
            if (shard_open_raw(path, s) != 0) {
                fprintf(stderr, "cannot open %s\n", path);
                return 1;
            }
        }
        g_nshards = nn;
    }

    /* derive the model's shape from its own tensors */
    g_embed_ent = dir_find("encoder.embed_tokens.weight");
    g_head_ent = dir_find("diffusion_head.weight");
    if (!g_embed_ent || !g_head_ent) {
        fprintf(stderr, "not a Sigma-family model directory\n");
        return 1;
    }
    M_VOCAB = g_embed_ent->rows;
    M_H = g_embed_ent->cols;
    {
        TEnt *e = dir_find("replaid_embedding_matrix.matrix");
        TEnt *g0 = dir_find("encoder.layers.0.mlp.gate_proj.weight");
        TEnt *q0 = dir_find("encoder.layers.0.self_attn.q_proj.weight");
        TEnt *k0 = dir_find("encoder.layers.0.self_attn.k_proj.weight");
        if (!e || !g0 || !q0 || !k0) {
            fprintf(stderr, "model directory is incomplete\n");
            return 1;
        }
        M_ED = e->cols;
        if (M_ED > 64) {
            fprintf(stderr, "embed_dim %d above the supported 64\n",
                    M_ED);
            return 1;
        }
        M_FFN = g0->rows;
        if (q0->rows % M_HD != 0 || k0->rows % M_HD != 0 || M_HD > 128) {
            fprintf(stderr, "unsupported head geometry\n");
            return 1;
        }
        M_HEADS = q0->rows / M_HD;
        M_KVHEADS = k0->rows / M_HD;
        /* The attention path groups query heads by
         * M_HEADS / M_KVHEADS: both counts must be real and
         * the division exact (H4). */
        if (M_HEADS < 1 || M_KVHEADS < 1
                || M_HEADS % M_KVHEADS != 0) {
            fprintf(stderr, "unsupported head grouping "
                    "(%d query heads over %d kv heads)\n",
                    M_HEADS, M_KVHEADS);
            return 1;
        }
        int maxl = -1;
        for (int i = 0; i < g_ndir; i++) {
            int li;
            if (sscanf(g_dir[i].name, "encoder.layers.%d.", &li) == 1
                    && li > maxl)
                maxl = li;
        }
        M_L = maxl + 1;
    }

    if (pk_stream) {
        g_use_packed = 1;
        if (packed_open(pk_stream, pk_book) != 0) {
            fprintf(stderr, "cannot open packed model\n");
            return 1;
        }
#ifndef _WIN32
        gate_calib_load(pk_stream);
#endif
    }

    /* resident small tensors */
    g_norm_in = (float *)malloc((size_t)M_L * M_H * sizeof(float));
    g_norm_post = (float *)malloc((size_t)M_L * M_H * sizeof(float));
    if (!g_norm_in || !g_norm_post) { fprintf(stderr, "oom\n"); return 1; }
    for (int l = 0; l < M_L; l++) {
        char nm[256];
        snprintf(nm, sizeof(nm), "encoder.layers.%d.input_layernorm.weight", l);
        float *b = load_small_req(nm, (uint64_t)M_H);
        memcpy(g_norm_in + (size_t)l * M_H, b, M_H * sizeof(float));
        free(b);
        snprintf(nm, sizeof(nm), "encoder.layers.%d.post_attention_layernorm.weight", l);
        b = load_small_req(nm, (uint64_t)M_H);
        memcpy(g_norm_post + (size_t)l * M_H, b, M_H * sizeof(float));
        free(b);
    }
    g_norm_final = load_small_req("encoder.norm.weight",
                                  (uint64_t)M_H);
    g_in_linear = load_small_req("replaid_input_linear.weight",
                                 (uint64_t)M_H * M_ED);
    g_sc_linear = load_small_req("replaid_selfcond_linear.weight",
                                 (uint64_t)M_H * M_ED);
    g_gamma_linear = load_small_req("replaid_gamma_linear.weight",
                                    (uint64_t)M_H * 64);
    {
        float *raw_e = load_small_req("replaid_embedding_matrix.matrix",
                                      (uint64_t)M_VOCAB * M_ED);
        g_E = (float *)malloc((size_t)M_VOCAB * M_ED * sizeof(float));
        if (!g_E) { fprintf(stderr, "oom\n"); return 1; }
        for (int v = 0; v < M_VOCAB; v++) {
            const float *r = raw_e + (size_t)v * M_ED;
            double nrm = 0.0;
            for (int d = 0; d < M_ED; d++) nrm += (double)r[d] * r[d];
            double inv = 1.0 / (sqrt(nrm) + 1e-8);
            for (int d = 0; d < M_ED; d++)
                g_E[(size_t)v * M_ED + d] = (float)(r[d] * inv);
        }
        free(raw_e);
        float *gb = load_small_req("gamma_bounds.endpoint_values",
                                   2);
        g_gamma0 = gb[0];
        g_gamma1 = gb[1];
        free(gb);
    }

    /* prompt ids */
    int prompt[8192];
    int plen = 0;
    {
        char buf[65536];
        snprintf(buf, sizeof(buf), "%s", ids_arg);
        char *tok = strtok(buf, ", ");
        while (tok && plen < 8192) {
            prompt[plen++] = atoi(tok);
            tok = strtok(NULL, ", ");
        }
    }
    if (plen == 0) { fprintf(stderr, "empty prompt\n"); return 1; }
    for (int i = 0; i < plen; i++)
        if (prompt[i] < 0 || prompt[i] >= M_VOCAB) {
            fprintf(stderr, "token id %d out of range\n", prompt[i]);
            return 1;
        }

    int cap = plen + max_new + block + 8;
    if (validate_projection_shapes() != 0) return 1;
    if (threads <= 0) threads = cpu_count();
    pool_init(threads);
    pipe_init(M_L);
    rope_init(cap);
    cache_init(cap);
    WS w = ws_alloc(plen > block ? plen : block);

    /* The pipe budget counts the float slots AND the staging
     * buffers pipe_init allocated (M-I6): at Sigma the staging
     * is larger than the slots it stages for. */
    size_t pipe_bytes = (size_t)g_pipe.slots * g_pipe.cap_elems * sizeof(float)
        + g_pipe_staging_bytes;
    size_t cache_bytes = (size_t)M_L * 2 * cap * (M_KVHEADS * M_HD) * 2;
    printf("NanoQuatTwo inference engine\n");
    printf("model: %d layers, hidden %d, vocab %d, embed_dim %d, "
           "source: %s\n", M_L, M_H, M_VOCAB, M_ED,
           g_use_packed ? "packed .nq2" : "raw safetensors");
    printf("threads: %d workers + fetch/unpack (triforce); derived budget: "
           "pipe %.1f MB (incl. staging %.1f MB), KV cache %.1f MB, "
           "embedding matrix %.1f MB\n",
           threads, pipe_bytes / 1048576.0,
           g_pipe_staging_bytes / 1048576.0, cache_bytes / 1048576.0,
           (size_t)M_VOCAB * M_ED * 4 / 1048576.0);
#ifndef _WIN32
    if (g_use_packed) {
        printf("precision gate: posture %s (%s), lookahead %d "
               "(%s), resync %d (%s)",
               posture_name(g_posture), g_posture_src,
               g_gate_lookahead, g_gate_lookahead_src,
               g_gate_resync, g_gate_resync_src);
        if (g_gate_calibrated)
            printf(", calibration: eps %.4e%s, budget %.4e%s",
                   gate_eps_eff(),
                   g_gate_eps_override >= 0.0 ? " OVERRIDE" : "",
                   gate_budget_eff(),
                   g_gate_budget_override >= 0.0 ? " OVERRIDE" : "");
        else
            printf(", calibration: none");
        printf("\n");
        if (g_posture == POSTURE_AUTO && !g_gate_calibrated)
            printf("precision gate: Auto without calibration "
                   "falls back to expanded for this pack\n");
    } else {
        printf("precision gate: inert (raw run: every ordinal "
               "expands)\n");
    }
#else
    printf("precision gate: note: the Windows fused worker "
           "always expands; postures do not take effect on "
           "this path\n");
#endif

    /* gamma grid, fp64 */
    double *grid = (double *)malloc(sizeof(double) * (steps + 1));
    if (!grid) { fprintf(stderr, "oom\n"); return 1; }
    if (schedule_lognsnr) {
        for (int i = 0; i <= steps; i++)
            grid[i] = g_gamma1 + (g_gamma0 - g_gamma1) * ((double)i / steps);
    } else {
        double rho = 7.0;
        double smax = exp(g_gamma1 / 2.0), smin = exp(g_gamma0 / 2.0);
        double a = pow(smax, 1.0 / rho), b = pow(smin, 1.0 / rho);
        for (int i = 0; i <= steps; i++) {
            double s = pow(a + (b - a) * ((double)i / steps), rho);
            if (s < 1e-30) s = 1e-30;
            grid[i] = 2.0 * log(s);
        }
    }

    rng_seed(seed);
    {
        int *pos = (int *)malloc(sizeof(int) * (plen > block ? plen : block));
        for (int i = 0; i < plen; i++) pos[i] = i;
        encode_clean(prompt, plen, pos, &w);

        int *out_ids = (int *)malloc(sizeof(int) * max_new);
        double *z = (double *)malloc((size_t)block * M_ED * sizeof(double));
        float *selfcond = (float *)malloc((size_t)block * M_ED * sizeof(float));
        float *logits = (float *)malloc((size_t)block * M_VOCAB * sizeof(float));
        float *x_guided = (float *)malloc((size_t)block * M_ED * sizeof(float));
        float *cond_mean = (float *)malloc((size_t)block * M_ED * sizeof(float));
        if (!pos || !out_ids || !z || !selfcond || !logits || !x_guided
                || !cond_mean) {
            fprintf(stderr, "oom\n");
            return 1;
        }
        long nfe = 0;
        int generated = 0;
        int nblocks = (max_new + block - 1) / block;
        for (int bidx = 0; bidx < nblocks; bidx++) {
            for (int i = 0; i < block * M_ED; i++) z[i] = randn();
            int have_sc = 0;
            for (int i = 0; i < block; i++)
                pos[i] = plen + generated + i;
            for (int s = 0; s <= steps; s++) {
                double gt = grid[s];
                g_gate_final_step = (s == steps);
                denoise(z, gt, have_sc ? selfcond : NULL, guidance,
                        pos, block, logits, x_guided, cond_mean, &w);
                g_gate_final_step = 0;
                nfe += guidance != 1.0 ? 2 : 1;
                if (s == steps) {
                    for (int i = 0; i < block; i++) {
                        const float *lg = logits + (size_t)i * M_VOCAB;
                        int best = 0;
                        for (int v = 1; v < M_VOCAB; v++)
                            if (lg[v] > lg[best]) best = v;
                        if (generated + i < max_new)
                            out_ids[generated + i] = best;
                    }
                    break;
                }
                memcpy(selfcond, cond_mean,
                       (size_t)block * M_ED * sizeof(float));
                have_sc = 1;
                double a_sq_t = 1.0 / (1.0 + exp(gt));
                double a_t = sqrt(a_sq_t);
                double s_t = sqrt(1.0 / (1.0 + exp(-gt)));
                double gs = grid[s + 1];
                double a_sq_s = 1.0 / (1.0 + exp(gs));
                double a_s = sqrt(a_sq_s);
                double s_s = sqrt(1.0 / (1.0 + exp(-gs)));
                for (int i = 0; i < block * M_ED; i++) {
                    double x = x_guided[i];
                    double eps = (z[i] - a_t * x) / (s_t < 1e-8 ? 1e-8 : s_t);
                    if (score_temp != 1.0) {
                        eps /= score_temp;
                        x = (z[i] - s_t * eps) / (a_t < 1e-8 ? 1e-8 : a_t);
                    }
                    if (sampler_ddim) {
                        z[i] = a_s * x + s_s * eps;
                    } else {
                        double c = -expm1(gs - gt);
                        double noise = randn();
                        double var = c * (1.0 - a_sq_s);
                        if (var < 0.0) var = 0.0;
                        z[i] = (1.0 - c) * sqrt(a_sq_s / a_sq_t) * z[i]
                             + c * a_s * x + sqrt(var) * noise;
                    }
                }
            }
            generated += block;
            if (bidx + 1 < nblocks) {
                int blk_ids[4096];
                for (int i = 0; i < block; i++)
                    blk_ids[i] = out_ids[bidx * block + i];
                encode_clean(blk_ids, block, pos, &w);
            }
        }
        printf("generated token ids:");
        for (int i = 0; i < max_new; i++) printf(" %d", out_ids[i]);
        printf("\n");
        printf("nfe: %ld (denoiser evaluations), seed %llu\n",
               nfe, (unsigned long long)seed);
    }

#ifndef _WIN32
    if (g_use_packed && g_posture == POSTURE_AUTO)
        printf("precision gate: %ld evaluations, %ld "
               "escalations, %ld resyncs; layer consumptions "
               "via DTC %ld of %ld\n",
               g_gate_evals, g_gate_escalations, g_gate_resyncs,
               g_gate_layers_dtc, g_gate_layers_total);
    else if (g_use_packed && g_posture == POSTURE_FAST)
        printf("precision gate: layer consumptions via DTC "
               "%ld of %ld (fast posture)\n",
               g_gate_layers_dtc, g_gate_layers_total);
#endif
    pipe_shutdown();     /* join first, then reduce and report */
    pool_shutdown();
    pipe_report();
    return 0;
}
