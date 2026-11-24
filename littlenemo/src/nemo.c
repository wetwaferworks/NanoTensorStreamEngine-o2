/* nemo.c - Little Nemo engine (Nemotron-H family), pure C.
 *
 * Milestone P0: checkpoint inspector.
 *   nemo inspect <shard.safetensors>
 * prints every tensor in the shard (name, dtype, shape, element count, byte
 * span) straight from the safetensors header, plus totals. The header is
 * untrusted input: length-guarded and span-checked in the hardened pattern
 * established by the 2026-11-24 NanoQuant hardening pass.
 *
 * Runtime code is C only (JD's standing rule). No external libraries.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static uint64_t rd_u64le(const unsigned char *p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)p[i] << (8 * i);
    return v;
}

/* Find the value for key "name" inside the object spanning
 * [obj, obj_end). Scoped: a missing key is never borrowed from the
 * next tensor's object. */
static const char *find_entry(const char *obj, const char *obj_end,
                              const char *name) {
    size_t nlen = strlen(name);
    for (const char *p = obj; p < obj_end; p++) {
        if (*p != '"') continue;
        if ((size_t)(obj_end - (p + 1)) < nlen + 1) return NULL;
        if (strncmp(p + 1, name, nlen) == 0 && p[1 + nlen] == '"') {
            const char *colon = p + 1 + nlen;
            while (colon < obj_end && *colon != ':') colon++;
            if (colon >= obj_end) return NULL;
            const char *v = colon + 1;
            while (v < obj_end && (*v == ' ' || *v == '\t')) v++;
            return v < obj_end ? v : NULL;
        }
    }
    return NULL;
}

/* Extract a quoted string value for "key" inside [obj, obj_end). */
static int obj_string(const char *obj, const char *obj_end, const char *key,
                      char *out, size_t outsz) {
    const char *v = find_entry(obj, obj_end, key);
    if (!v || *v != '"') return -1;
    v++;
    size_t i = 0;
    while (v < obj_end && *v != '"' && i + 1 < outsz) out[i++] = *v++;
    out[i] = 0;
    return (v < obj_end && *v == '"') ? 0 : -1;
}

/* Extract up to maxn integers of array "key" inside [obj, obj_end). */
static int obj_u64s(const char *obj, const char *obj_end, const char *key,
                    uint64_t *out, int maxn) {
    const char *v = find_entry(obj, obj_end, key);
    if (!v || *v != '[') return -1;
    v++;
    int n = 0;
    while (v < obj_end && *v != ']' && n < maxn) {
        while (v < obj_end && (*v == ' ' || *v == ',' || *v == '\t')) v++;
        /* strtoull accepts a leading '-': refuse signs outright. */
        if (v < obj_end && (*v == '-' || *v == '+')) return -1;
        if (*v == ']') break;
        char *end = NULL;
        out[n++] = strtoull(v, &end, 10);
        if (end == v) return -1;
        v = end;
    }
    return n;
}

static int dtype_size(const char *dt) {
    if (!strcmp(dt, "F64")) return 8;
    if (!strcmp(dt, "F32") || !strcmp(dt, "I32") || !strcmp(dt, "U32")) return 4;
    if (!strcmp(dt, "F16") || !strcmp(dt, "BF16") || !strcmp(dt, "I16") || !strcmp(dt, "U16")) return 2;
    if (!strcmp(dt, "I8") || !strcmp(dt, "U8") || !strcmp(dt, "BOOL")) return 1;
    return 0;
}

static int inspect(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "nemo: cannot open %s\n", path); return 1; }
    unsigned char lenbuf[8];
    if (fread(lenbuf, 1, 8, f) != 8) { fprintf(stderr, "nemo: short read on header length\n"); fclose(f); return 1; }
    uint64_t hlen = rd_u64le(lenbuf);
    if (hlen == 0 || hlen > (1ull << 28)) {
        fprintf(stderr, "nemo: refusing implausible header length %llu\n", (unsigned long long)hlen);
        fclose(f); return 1;
    }
    char *hdr = malloc(hlen + 1);
    if (!hdr) { fprintf(stderr, "nemo: out of memory\n"); fclose(f); return 1; }
    if (fread(hdr, 1, hlen, f) != hlen) { fprintf(stderr, "nemo: short read on header\n"); free(hdr); fclose(f); return 1; }
    hdr[hlen] = 0;
    fclose(f);

    /* Walk top-level keys: each is "name": { ... } with balanced braces. */
    uint64_t total_elems = 0, total_span = 0;
    int ntensors = 0;
    const char *p = hdr;
    if (*p == '{') p++;
    while (*p) {
        while (*p == ' ' || *p == ',' || *p == '\n' || *p == '\t') p++;
        if (*p == '}' || *p == 0) break;
        if (*p != '"') { fprintf(stderr, "nemo: header parse error\n"); free(hdr); return 1; }
        const char *name_start = ++p;
        while (*p && *p != '"') p++;
        /* Truncated header: the scan stops on the terminating NUL,
         * not a quote; stepping past it would leave the allocation. */
        if (*p != '"') { fprintf(stderr, "nemo: truncated header\n"); free(hdr); return 1; }
        size_t nlen = (size_t)(p - name_start);
        char name[512];
        if (nlen >= sizeof name) { fprintf(stderr, "nemo: tensor name too long\n"); free(hdr); return 1; }
        memcpy(name, name_start, nlen); name[nlen] = 0;
        p++; /* closing quote */
        while (*p == ' ' || *p == ':') p++;
        if (!strcmp(name, "__metadata__")) {
            /* skip its object */
            int depth = 0;
            do { if (*p == '{') depth++; else if (*p == '}') depth--; p++; } while (*p && depth > 0);
            if (depth != 0) { fprintf(stderr, "nemo: truncated header\n"); free(hdr); return 1; }
            continue;
        }
        if (*p != '{') { fprintf(stderr, "nemo: header parse error at %s\n", name); free(hdr); return 1; }
        const char *obj = p;
        /* Find this object's end before extracting anything from it. */
        const char *obj_end = obj;
        {
            int depth = 0;
            do { if (*obj_end == '{') depth++; else if (*obj_end == '}') depth--; obj_end++; } while (*obj_end && depth > 0);
            if (depth != 0) { fprintf(stderr, "nemo: truncated header\n"); free(hdr); return 1; }
        }
        char dtype[16];
        uint64_t shape[8], offs[2];
        if (obj_string(obj, obj_end, "dtype", dtype, sizeof dtype) != 0) { fprintf(stderr, "nemo: no dtype for %s\n", name); free(hdr); return 1; }
        int nd = obj_u64s(obj, obj_end, "shape", shape, 8);
        if (nd < 0) { fprintf(stderr, "nemo: no shape for %s\n", name); free(hdr); return 1; }
        if (obj_u64s(obj, obj_end, "data_offsets", offs, 2) != 2) { fprintf(stderr, "nemo: no offsets for %s\n", name); free(hdr); return 1; }
        int esz = dtype_size(dtype);
        if (esz == 0) { fprintf(stderr, "nemo: unknown dtype %s for %s\n", dtype, name); free(hdr); return 1; }
        uint64_t count = 1;
        for (int i = 0; i < nd; i++) {
            if (shape[i] && count > UINT64_MAX / shape[i]) { fprintf(stderr, "nemo: shape overflow for %s\n", name); free(hdr); return 1; }
            count *= shape[i];
        }
        /* The span product can itself wrap; guard before comparing. */
        if (count > UINT64_MAX / (uint64_t)esz) { fprintf(stderr, "nemo: span overflow for %s\n", name); free(hdr); return 1; }
        if (offs[1] < offs[0]) { fprintf(stderr, "nemo: reversed offsets for %s\n", name); free(hdr); return 1; }
        uint64_t span = offs[1] - offs[0];
        if (span != count * (uint64_t)esz) {
            fprintf(stderr, "nemo: span mismatch for %s (span %llu, count %llu x %d)\n",
                    name, (unsigned long long)span, (unsigned long long)count, esz);
            free(hdr); return 1;
        }
        printf("%-72s %-5s [", name, dtype);
        for (int i = 0; i < nd; i++) printf("%s%llu", i ? " x " : "", (unsigned long long)shape[i]);
        printf("]  %llu elems\n", (unsigned long long)count);
        total_elems += count;
        total_span += span;
        ntensors++;
        /* skip this object (its end was validated above) */
        p = obj_end;
    }
    printf("------------------------------------------------------------------\n");
    printf("%d tensors, %llu elements, %llu data bytes\n", ntensors,
           (unsigned long long)total_elems, (unsigned long long)total_span);
    free(hdr);
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 3 && !strcmp(argv[1], "inspect")) return inspect(argv[2]);
    fprintf(stderr, "usage: nemo inspect <shard.safetensors>\n");
    return 2;
}
