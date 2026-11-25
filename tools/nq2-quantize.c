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

static uint64_t rng_state = 0x2545F4914F6CDD1Dull;
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
                else {
                    fprintf(stderr, "warning: tensor %s has unrecognized "
                            "dtype '%s'; assuming F32\n", key, dts);
                    t->dtype = DT_F32;
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
static int tensor_read(Shard *sh, TensorInfo *t, uint64_t start,
                       uint64_t count, float *dst)
{
    uint64_t esz = t->dtype == DT_F32 ? 4 : 2;
    uint64_t byte_off = sh->data_start + t->begin + start * esz;
    uint8_t *raw = (uint8_t *)malloc(count * esz);
    if (!raw) return -1;
    if (seek_to(sh->file, byte_off) != 0) { free(raw); return -1; }
    if (fread(raw, esz, count, sh->file) != count) { free(raw); return -1; }
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

/* Normalize one block into out; also returns the block's frame
 * (minimum and scale) through mn_out / sc_out. */
static void normalize_block(const float *raw, uint32_t block,
                            float *out, float *mn_out, float *sc_out)
{
    if (g_norm_rms) {
        double s = 0.0;
        for (uint32_t i = 0; i < block; i++) s += (double)raw[i] * raw[i];
        float sc = (float)dsqrt(s / block);
        if (sc == 0.0f) sc = 1.0f;
        for (uint32_t i = 0; i < block; i++) out[i] = raw[i] / sc;
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
    if (sc == 0.0f) sc = 1.0f;
    for (uint32_t i = 0; i < block; i++) out[i] = (raw[i] - mn) / sc;
    *mn_out = mn;
    *sc_out = sc;
}

static int nearest_centroid(const float *cent, uint32_t entries,
                            uint32_t block, const float *x)
{
    int best = 0;
    double best_err = 1e30;
    for (uint32_t k = 0; k < entries; k++) {
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
    for (uint32_t k = 0; k < entries; k++) {
        uint64_t src = rng_next() % sample_n;
        memcpy(cent + (size_t)k * block, sample + src * block,
               block * sizeof(float));
    }
    double *sums = (double *)calloc((size_t)entries * block, sizeof(double));
    uint64_t *counts = (uint64_t *)calloc(entries, sizeof(uint64_t));
    if (!sums || !counts) { fprintf(stderr, "out of memory\n"); exit(1); }
    for (int it = 0; it < iters; it++) {
        memset(sums, 0, (size_t)entries * block * sizeof(double));
        memset(counts, 0, entries * sizeof(uint64_t));
        for (uint64_t b = 0; b < sample_n; b++) {
            const float *x = sample + b * block;
            int k = nearest_centroid(cent, entries, block, x);
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
        printf("  kmeans iter %d: max centroid shift %.5f\n", it, shift);
    }
    free(sums);
    free(counts);
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

/* ---------------- main ---------------- */

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
    const char *shard_paths[16];
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
        else if (argv[i][0] != '-' && n_shard_paths < 16) shard_paths[n_shard_paths++] = argv[i];
        else { fprintf(stderr, "bad argument: %s\n", argv[i]); return 1; }
    }
    if (!out_prefix || n_shard_paths == 0) {
        fprintf(stderr,
            "usage: nq2-quantize --out PREFIX shard.safetensors [more shards]\n"
            "         [--block 256] [--entries 256] [--budget N]\n"
            "         [--filter a,b] [--sample-blocks N] [--iters N]\n");
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

    /* Open shards and build the scope: whole tensors, in order,
     * while the running weight total is under the budget. */
    Shard *shards = (Shard *)calloc(n_shard_paths, sizeof(Shard));
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
        /* Per-tensor books (format version 2). Each tensor is
         * sampled, learned, and packed with its own codebook, so
         * every book speaks only its own tensor's language. The
         * stream gains a tensor table (blocks per tensor) so the
         * runner knows which book each block belongs to. */
        float *books = (float *)calloc((size_t)g_n_scope * entries * block,
                                       sizeof(float));
        if (!books) { fprintf(stderr, "out of memory\n"); return 1; }
        uint64_t *tblocks = (uint64_t *)malloc(sizeof(uint64_t) * g_n_scope);
        double *terr = (double *)calloc(g_n_scope, sizeof(double));
        double *tref = (double *)calloc(g_n_scope, sizeof(double));
        double res_err2 = 0.0;
        if (!tblocks || !terr || !tref) { fprintf(stderr, "out of memory\n"); return 1; }
        PackCtx pctx;
        memset(&pctx, 0, sizeof(pctx));
        pctx.block = block;
        pctx.entries = entries;
        pctx.idx = (uint8_t *)malloc(g_total_blocks);
        pctx.mn_h = (uint16_t *)malloc(g_total_blocks * 2);
        pctx.sc_h = (uint16_t *)malloc(g_total_blocks * 2);
        pctx.tmp = (float *)malloc(block * sizeof(float));
        if (!pctx.idx || !pctx.mn_h || !pctx.sc_h || !pctx.tmp) {
            fprintf(stderr, "out of memory packing\n");
            return 1;
        }
        for (int s = 0; s < g_n_scope; s++) {
            TensorInfo *ti = g_scope[s].tensor;
            tblocks[s] = ti->count / block;
            float *cent = books + (size_t)s * entries * block;
            printf("tensor %d/%d %s: %llu blocks\n", s + 1, g_n_scope,
                   ti->name, (unsigned long long)tblocks[s]);
            if (tblocks[s] == 0) continue;
            uint64_t cap = tensor_sample_cap;
            if (cap > tblocks[s]) cap = tblocks[s];
            SampleCtx sctx;
            memset(&sctx, 0, sizeof(sctx));
            sctx.block = block;
            sctx.sample_cap = cap;
            sctx.sample = (float *)malloc(cap * block * sizeof(float));
            sctx.tmp = (float *)malloc(block * sizeof(float));
            if (!sctx.sample || !sctx.tmp) { fprintf(stderr, "out of memory sampling\n"); return 1; }
            walk_one_tensor(s, block, sample_fn, &sctx);
            learn_book(sctx.sample, sctx.sample_n, block, entries, iters, cent);
            free(sctx.sample); free(sctx.tmp);
            pctx.cent = cent;
            double e0 = pctx.err2, r0 = pctx.ref2;
            walk_one_tensor(s, block, pack_fn, &pctx);
            terr[s] = pctx.err2 - e0;
            tref[s] = pctx.ref2 - r0;
            printf("  tensor relative L2: %.4f\n",
                   tref[s] > 0 ? dsqrt(terr[s] / tref[s]) : 0.0);
            if (g_residual) {
                ResSampleCtx rctx;
                memset(&rctx, 0, sizeof(rctx));
                rctx.block = block;
                rctx.entries = entries;
                rctx.cent1 = cent;
                rctx.inner.block = block;
                rctx.inner.sample_cap = cap;
                rctx.inner.sample = (float *)malloc(cap * block * sizeof(float));
                rctx.inner.tmp = (float *)malloc(block * sizeof(float));
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
                float *cent2 = (float *)malloc((size_t)entries * block * sizeof(float));
                if (!cent2) { fprintf(stderr, "out of memory (residual)\n"); return 1; }
                learn_book(rctx.inner.sample, rctx.inner.sample_n,
                           block, entries, iters, cent2);
                free(rctx.inner.sample); free(rctx.inner.tmp);
                ResFidCtx fctx;
                memset(&fctx, 0, sizeof(fctx));
                fctx.block = block;
                fctx.entries = entries;
                fctx.cent1 = cent;
                fctx.cent2 = cent2;
                fctx.tmp = rctx.tmp;
                fctx.recon = rctx.recon;
                fctx.resid = rctx.resid;
                fctx.ntmp = rctx.ntmp;
                walk_one_tensor(s, block, res_fid_fn, &fctx);
                res_err2 += fctx.err2;
                printf("  residual stage: two-stage relative L2 %.4f\n",
                       fctx.ref2 > 0 ? dsqrt(fctx.err2 / fctx.ref2) : 0.0);
                free(cent2);
                free(rctx.tmp); free(rctx.recon);
                free(rctx.resid); free(rctx.ntmp);
            }
        }
        double rel_l2 = pctx.ref2 > 0 ? dsqrt(pctx.err2 / pctx.ref2) : 0.0;
        printf("packed %llu blocks in %d per-tensor books; "
               "relative L2 error %.4f\n",
               (unsigned long long)pctx.cursor, g_n_scope, rel_l2);

        char pathbuf[1024];
        uint32_t version = 2;
        uint32_t n_books = (uint32_t)g_n_scope;
        snprintf(pathbuf, sizeof(pathbuf), "%s.nq2book", out_prefix);
        FILE *bf = fopen(pathbuf, "wb");
        if (!bf) { fprintf(stderr, "cannot write %s\n", pathbuf); return 1; }
        fwrite("NQ2B", 1, 4, bf);
        fwrite(&version, 4, 1, bf);
        fwrite(&block, 4, 1, bf);
        fwrite(&entries, 4, 1, bf);
        fwrite(&n_books, 4, 1, bf);
        fwrite(books, sizeof(float),
               (size_t)g_n_scope * entries * block, bf);
        fclose(bf);

        snprintf(pathbuf, sizeof(pathbuf), "%s.nq2", out_prefix);
        FILE *sf = fopen(pathbuf, "wb");
        if (!sf) { fprintf(stderr, "cannot write %s\n", pathbuf); return 1; }
        fwrite("NQ2\0", 1, 4, sf);
        fwrite(&version, 4, 1, sf);
        fwrite(&block, 4, 1, sf);
        fwrite(&entries, 4, 1, sf);
        fwrite(&g_total_weights, 8, 1, sf);
        fwrite(&g_total_blocks, 8, 1, sf);
        fwrite(&n_books, 4, 1, sf);
        fwrite(tblocks, 8, g_n_scope, sf);
        fwrite(pctx.idx, 1, g_total_blocks, sf);
        fwrite(pctx.mn_h, 2, g_total_blocks, sf);
        fwrite(pctx.sc_h, 2, g_total_blocks, sf);
        fclose(sf);

        uint64_t book_bytes = 20 + (uint64_t)g_n_scope * entries * block * 4;
        uint64_t stream_bytes = 36 + (uint64_t)g_n_scope * 8 +
                                g_total_blocks * 5;
        double rate = (double)stream_bytes * 8.0 / (double)g_total_weights;
        double crate = (double)(stream_bytes + book_bytes) * 8.0
                       / (double)g_total_weights;
        snprintf(pathbuf, sizeof(pathbuf), "%s.report.txt", out_prefix);
        FILE *rf = fopen(pathbuf, "w");
        if (rf) {
            fprintf(rf, "NanoQuatTwo quantization report (per-tensor books, format v2)\n");
            fprintf(rf, "tensors packed: %d (one book each)\n", g_n_scope);
            fprintf(rf, "weights packed: %llu\n", (unsigned long long)g_total_weights);
            fprintf(rf, "block size: %u, entries per book: %u\n", block, entries);
            fprintf(rf, "normalization: %s\n", g_norm_rms ? "rms (no offset)" : "minmax");
            fprintf(rf, "index stream (.nq2): %llu bytes\n", (unsigned long long)stream_bytes);
            fprintf(rf, "memory file (.nq2book): %llu bytes\n", (unsigned long long)book_bytes);
            fprintf(rf, "stream rate: %.4f bits per weight (stream only)\n", rate);
            fprintf(rf, "complete rate: %.4f bits per weight (stream + book, two-part accounting)\n", crate);
            fprintf(rf, "relative L2 error on real weights: %.4f\n", rel_l2);
            if (g_residual) {
                double two_l2 = pctx.ref2 > 0
                    ? dsqrt(res_err2 / pctx.ref2) : 0.0;
                uint64_t stream2 = stream_bytes + g_total_blocks * 5;
                uint64_t book2 = book_bytes
                    + (uint64_t)g_n_scope * entries * block * 4;
                double crate2 = (double)(stream2 + book2) * 8.0
                    / (double)g_total_weights;
                fprintf(rf, "residual stage (experimental, measured only; "
                        "the written stream is single-stage):\n");
                fprintf(rf, "  two-stage relative L2: %.4f\n", two_l2);
                fprintf(rf, "  two-stage complete rate would be: "
                        "%.4f bits per weight\n", crate2);
            }
            fprintf(rf, "per-tensor relative L2:\n");
            for (int s = 0; s < g_n_scope; s++)
                fprintf(rf, "  %s: %.4f\n", g_scope[s].tensor->name,
                        tref[s] > 0 ? dsqrt(terr[s] / tref[s]) : 0.0);
            fclose(rf);
        }
        printf("index stream: %llu bytes; memory file: %llu bytes\n",
               (unsigned long long)stream_bytes,
               (unsigned long long)book_bytes);
        printf("rate: %.4f bits per weight (stream only); "
               "complete %.4f (stream + book)\n", rate, crate);
        return 0;
    }

    /* Pass 1: sample normalized blocks from across the scope. */
    if (sample_cap == 0 || tensor_sample_cap == 0) {
        fprintf(stderr, "--sample-blocks and --tensor-sample-blocks "
                "must be at least 1\n");
        return 1;
    }
    if (sample_cap > g_total_blocks) sample_cap = g_total_blocks;
    SampleCtx sctx;
    memset(&sctx, 0, sizeof(sctx));
    sctx.block = block;
    sctx.sample_cap = sample_cap;
    sctx.sample = (float *)malloc(sample_cap * block * sizeof(float));
    sctx.tmp = (float *)malloc(block * sizeof(float));
    if (!sctx.sample || !sctx.tmp) { fprintf(stderr, "out of memory sampling\n"); return 1; }
    walk_blocks(block, sample_fn, &sctx);
    printf("sampled %llu blocks for k-means\n",
           (unsigned long long)sctx.sample_n);

    /* Learn the book: k-means (Lloyd) on the sample. */
    float *cent = (float *)malloc((size_t)entries * block * sizeof(float));
    float *tmp = (float *)malloc(block * sizeof(float));
    if (!cent || !tmp) { fprintf(stderr, "out of memory learning\n"); return 1; }
    learn_book(sctx.sample, sctx.sample_n, block, entries, iters, cent);
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
    walk_blocks(block, pack_fn, &pctx);
    double rel_l2 = pctx.ref2 > 0 ? dsqrt(pctx.err2 / pctx.ref2) : 0.0;
    printf("packed %llu blocks; relative L2 error %.4f\n",
           (unsigned long long)pctx.cursor, rel_l2);

    /* Write the memory file: magic, block, entries, prototypes. */
    char pathbuf[1024];
    snprintf(pathbuf, sizeof(pathbuf), "%s.nq2book", out_prefix);
    FILE *bf = fopen(pathbuf, "wb");
    if (!bf) { fprintf(stderr, "cannot write %s\n", pathbuf); return 1; }
    fwrite("NQ2B", 1, 4, bf);
    fwrite(&block, 4, 1, bf);
    fwrite(&entries, 4, 1, bf);
    fwrite(cent, sizeof(float), (size_t)entries * block, bf);
    fclose(bf);

    /* Write the index stream: header, indexes, mins, scales. */
    snprintf(pathbuf, sizeof(pathbuf), "%s.nq2", out_prefix);
    FILE *sf = fopen(pathbuf, "wb");
    if (!sf) { fprintf(stderr, "cannot write %s\n", pathbuf); return 1; }
    uint32_t version = 1;
    fwrite("NQ2\0", 1, 4, sf);
    fwrite(&version, 4, 1, sf);
    fwrite(&block, 4, 1, sf);
    fwrite(&entries, 4, 1, sf);
    fwrite(&g_total_weights, 8, 1, sf);
    fwrite(&g_total_blocks, 8, 1, sf);
    fwrite(pctx.idx, 1, g_total_blocks, sf);
    fwrite(pctx.mn_h, 2, g_total_blocks, sf);
    fwrite(pctx.sc_h, 2, g_total_blocks, sf);
    fclose(sf);

    uint64_t book_bytes = 12 + (uint64_t)entries * block * 4;
    uint64_t stream_bytes = 32 + g_total_blocks * 5;
    double rate = (double)stream_bytes * 8.0 / (double)g_total_weights;
    double crate = (double)(stream_bytes + book_bytes) * 8.0
                   / (double)g_total_weights;
    snprintf(pathbuf, sizeof(pathbuf), "%s.report.txt", out_prefix);
    FILE *rf = fopen(pathbuf, "w");
    if (rf) {
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
        fclose(rf);
    }
    printf("index stream: %llu bytes; memory file: %llu bytes\n",
           (unsigned long long)stream_bytes, (unsigned long long)book_bytes);
    printf("rate: %.4f bits per weight (stream only); "
           "complete %.4f (stream + book)\n", rate, crate);
    return 0;
}
