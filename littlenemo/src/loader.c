/* loader.c - see loader.h. */
#define _POSIX_C_SOURCE 200809L /* strdup, fseeko under strict C99 */
#include "loader.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mamba.h" /* nemo_bf16_to_f32 */

static uint64_t rd_u64le(const unsigned char *p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)p[i] << (8 * i);
    return v;
}

static int dtype_info(const char *dt, int *esz, int *is_bf16, int *is_f32) {
    *is_bf16 = 0;
    *is_f32 = 0;
    if (!strcmp(dt, "BF16")) { *esz = 2; *is_bf16 = 1; return 0; }
    if (!strcmp(dt, "F32")) { *esz = 4; *is_f32 = 1; return 0; }
    if (!strcmp(dt, "F16")) { *esz = 2; return 0; }
    if (!strcmp(dt, "F64")) { *esz = 8; return 0; }
    if (!strcmp(dt, "I64")) { *esz = 8; return 0; }
    if (!strcmp(dt, "I32")) { *esz = 4; return 0; }
    return -1;
}

/* Extract "key":"str" / "key":[ints] from the JSON object spanning
 * [obj, obj_end). The search is scoped to the object: a key missing
 * here must never be borrowed from the next tensor's object. */
static const char *find_key(const char *obj, const char *obj_end,
                            const char *key) {
    size_t klen = strlen(key);
    for (const char *p = obj; p < obj_end; p++) {
        if (*p != '"') continue;
        if ((size_t)(obj_end - (p + 1)) < klen + 1) return NULL;
        if (strncmp(p + 1, key, klen) == 0 && p[1 + klen] == '"') {
            const char *colon = p + 1 + klen;
            while (colon < obj_end && *colon != ':') colon++;
            if (colon >= obj_end) return NULL;
            const char *v = colon + 1;
            while (v < obj_end && (*v == ' ' || *v == '\t')) v++;
            return v < obj_end ? v : NULL;
        }
    }
    return NULL;
}

static int obj_string(const char *obj, const char *obj_end, const char *key,
                      char *out, size_t outsz) {
    const char *v = find_key(obj, obj_end, key);
    if (!v || *v != '"') return -1;
    v++;
    size_t i = 0;
    while (v < obj_end && *v != '"' && i + 1 < outsz) out[i++] = *v++;
    out[i] = 0;
    return (v < obj_end && *v == '"') ? 0 : -1;
}

static int obj_u64s(const char *obj, const char *obj_end, const char *key,
                    uint64_t *out, int maxn) {
    const char *v = find_key(obj, obj_end, key);
    if (!v || *v != '[') return -1;
    v++;
    int n = 0;
    while (v < obj_end && *v != ']' && n < maxn) {
        while (v < obj_end && (*v == ' ' || *v == ',' || *v == '\t')) v++;
        if (v >= obj_end || *v == ']') break;
        /* strtoull accepts a leading '-': a negative shape or offset
         * would wrap to a huge unsigned value. Refuse signs outright. */
        if (*v == '-' || *v == '+') return -1;
        char *end = NULL;
        out[n] = strtoull(v, &end, 10);
        if (end == v) return -1;
        n++;
        v = end;
    }
    return n;
}

static int cp_push(nemo_checkpoint *cp, const nemo_tensor_info *t) {
    if (cp->n == cp->cap) {
        int ncap = cp->cap ? cp->cap * 2 : 256;
        nemo_tensor_info *ni = realloc(cp->items, (size_t)ncap * sizeof *ni);
        if (!ni) return -1;
        cp->items = ni;
        cp->cap = ncap;
    }
    cp->items[cp->n++] = *t;
    return 0;
}

static int index_shard(nemo_checkpoint *cp, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    unsigned char lenbuf[8];
    if (fread(lenbuf, 1, 8, f) != 8) { fclose(f); return -1; }
    uint64_t hlen = rd_u64le(lenbuf);
    if (hlen == 0 || hlen > (1ull << 28)) { fclose(f); return -1; }
    char *hdr = malloc(hlen + 1);
    if (!hdr) { fclose(f); return -1; }
    if (fread(hdr, 1, hlen, f) != hlen) { free(hdr); fclose(f); return -1; }
    hdr[hlen] = 0;
    fclose(f);

    const char *p = hdr;
    if (*p == '{') p++;
    while (*p) {
        while (*p == ' ' || *p == ',' || *p == '\n' || *p == '\t') p++;
        if (*p == '}' || *p == 0) break;
        if (*p != '"') { free(hdr); return -1; }
        const char *ns = ++p;
        while (*p && *p != '"') p++;
        /* A truncated header leaves the scan sitting on the buffer's
         * terminating NUL, not on a quote: stepping past it would walk
         * off the allocation. Refuse instead. */
        if (*p != '"') { free(hdr); return -1; }
        size_t nlen = (size_t)(p - ns);
        char name[512];
        if (nlen >= sizeof name) { free(hdr); return -1; }
        memcpy(name, ns, nlen);
        name[nlen] = 0;
        p++;
        while (*p == ' ' || *p == ':') p++;
        int depth = 0;
        const char *obj = p;
        do { if (*p == '{') depth++; else if (*p == '}') depth--; p++; } while (*p && depth > 0);
        if (depth != 0) { free(hdr); return -1; }
        const char *obj_end = p;
        if (!strcmp(name, "__metadata__")) continue;
        char dtype[16];
        uint64_t shape[8], offs[2];
        if (obj_string(obj, obj_end, "dtype", dtype, sizeof dtype) != 0) { free(hdr); return -1; }
        int nd = obj_u64s(obj, obj_end, "shape", shape, 8);
        if (nd < 0) { free(hdr); return -1; }
        if (obj_u64s(obj, obj_end, "data_offsets", offs, 2) != 2) { free(hdr); return -1; }
        int esz, is_bf16, is_f32;
        if (dtype_info(dtype, &esz, &is_bf16, &is_f32) != 0) { free(hdr); return -1; }
        uint64_t count = 1;
        for (int i = 0; i < nd; i++) {
            if (shape[i] && count > UINT64_MAX / shape[i]) { free(hdr); return -1; }
            count *= shape[i];
        }
        /* The span product itself can wrap (count near 2^64 / esz):
         * guard it before comparing, and refuse reversed offsets. */
        if (count > UINT64_MAX / (uint64_t)esz) { free(hdr); return -1; }
        if (offs[1] < offs[0]) { free(hdr); return -1; }
        if (offs[1] - offs[0] != count * (uint64_t)esz) { free(hdr); return -1; }
        nemo_tensor_info t;
        memset(&t, 0, sizeof t);
        t.name = strdup(name);
        t.shard_path = strdup(path);
        if (!t.name || !t.shard_path) { free(t.name); free(t.shard_path); free(hdr); return -1; }
        t.data_off = 8 + hlen + offs[0];
        t.count = count;
        t.ndim = nd;
        memcpy(t.shape, shape, sizeof shape);
        t.elem_size = esz;
        t.is_bf16 = is_bf16;
        t.is_f32 = is_f32;
        if (cp_push(cp, &t) != 0) { free(t.name); free(t.shard_path); free(hdr); return -1; }
    }
    free(hdr);
    return 0;
}

static int ends_with(const char *s, const char *suffix) {
    size_t a = strlen(s), b = strlen(suffix);
    return a >= b && !strcmp(s + a - b, suffix);
}

int nemo_checkpoint_open(nemo_checkpoint *cp, const char *dir) {
    memset(cp, 0, sizeof *cp);
    DIR *d = opendir(dir);
    if (!d) return -1;
    struct dirent *e;
    int shards = 0;
    while ((e = readdir(d)) != NULL) {
        if (!ends_with(e->d_name, ".safetensors")) continue;
        char path[1024];
        snprintf(path, sizeof path, "%s/%s", dir, e->d_name);
        if (index_shard(cp, path) != 0) { closedir(d); nemo_checkpoint_close(cp); return -1; }
        shards++;
    }
    closedir(d);
    return shards ? 0 : -1;
}

void nemo_checkpoint_close(nemo_checkpoint *cp) {
    for (int i = 0; i < cp->n; i++) {
        free(cp->items[i].name);
        free(cp->items[i].shard_path);
    }
    free(cp->items);
    memset(cp, 0, sizeof *cp);
}

const nemo_tensor_info *nemo_find(const nemo_checkpoint *cp, const char *name) {
    for (int i = 0; i < cp->n; i++)
        if (!strcmp(cp->items[i].name, name)) return &cp->items[i];
    return NULL;
}

float *nemo_load_f32(const nemo_checkpoint *cp, const nemo_tensor_info *t) {
    (void)cp;
    FILE *f = fopen(t->shard_path, "rb");
    if (!f) return NULL;
    /* fseeko with off_t, never fseek with (long): long is 32-bit on
     * Windows and real shards exceed 4 GB. */
    if (fseeko(f, (off_t)t->data_off, SEEK_SET) != 0) { fclose(f); return NULL; }
    float *out = malloc((size_t)t->count * sizeof(float));
    if (!out) { fclose(f); return NULL; }
    if (t->is_bf16) {
        uint16_t *raw = malloc((size_t)t->count * 2);
        if (!raw) { free(out); fclose(f); return NULL; }
        if (fread(raw, 2, t->count, f) != t->count) { free(raw); free(out); fclose(f); return NULL; }
        for (uint64_t i = 0; i < t->count; i++) out[i] = nemo_bf16_to_f32(raw[i]);
        free(raw);
    } else if (t->is_f32) {
        /* Dispatch on dtype, not element size: an I32 tensor must be
         * refused, never reinterpreted as float bits. */
        if (fread(out, 4, t->count, f) != t->count) { free(out); fclose(f); return NULL; }
    } else {
        free(out);
        fclose(f);
        return NULL;
    }
    fclose(f);
    return out;
}

uint64_t nemo_tensor_fnv(const nemo_checkpoint *cp, const nemo_tensor_info *t) {
    (void)cp;
    FILE *f = fopen(t->shard_path, "rb");
    if (!f) return 0;
    if (fseeko(f, (off_t)t->data_off, SEEK_SET) != 0) { fclose(f); return 0; }
    uint64_t h = 1469598103934665603ull;
    unsigned char buf[65536];
    uint64_t left = t->count * (uint64_t)t->elem_size;
    while (left) {
        size_t want = left > sizeof buf ? sizeof buf : (size_t)left;
        size_t got = fread(buf, 1, want, f);
        if (got == 0) { fclose(f); return 0; }
        for (size_t i = 0; i < got; i++) { h ^= buf[i]; h *= 1099511628211ull; }
        left -= got;
    }
    fclose(f);
    return h;
}

uint16_t *nemo_load_bf16(const nemo_checkpoint *cp, const nemo_tensor_info *t) {
    (void)cp;
    if (!t || !t->is_bf16) return NULL;
    FILE *f = fopen(t->shard_path, "rb");
    if (!f) return NULL;
    uint16_t *buf = (uint16_t *)malloc(t->count * sizeof(uint16_t));
    if (!buf) { fclose(f); return NULL; }
    if (fseeko(f, (off_t)t->data_off, SEEK_SET) != 0 ||
        fread(buf, sizeof(uint16_t), t->count, f) != t->count) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    return buf;
}
