/* ln_resid.c - write the refinement residuals of chosen projections,
 * pure C.
 *
 * For each named projection, R = W_original - W_decoded_pack (both bf16,
 * widened exactly), written as an F32 safetensors tensor named
 * "<name>.refine" with the projection's shape. nq2-quantize then packs
 * the residuals with the same recipe as the base pack, and
 * nq2-merge-refine appends them to the base pack as a new, expanded pack.
 *
 * Usage: ln_resid --ref DIR --pack-ckpt DIR --out FILE.safetensors NAME...
 * Output is byte-identical for the same inputs (no threads, fixed order).
 */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "loader.h"

int main(int argc, char **argv) {
    const char *ref = NULL, *pck = NULL, *out = NULL;
    int first = argc;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--ref") && i + 1 < argc) ref = argv[++i];
        else if (!strcmp(argv[i], "--pack-ckpt") && i + 1 < argc) pck = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else { first = i; break; }
    }
    if (!ref || !pck || !out || first >= argc) {
        fprintf(stderr, "usage: ln_resid --ref DIR --pack-ckpt DIR --out FILE.safetensors NAME...\n");
        return 2;
    }
    int n = argc - first;
    nemo_checkpoint A, B;
    if (nemo_checkpoint_open(&A, ref) || nemo_checkpoint_open(&B, pck)) { fprintf(stderr, "checkpoint open failed\n"); return 1; }
    const nemo_tensor_info **ta = malloc(sizeof(*ta) * (size_t)n), **tb = malloc(sizeof(*tb) * (size_t)n);
    if (!ta || !tb) return 1;
    /* header */
    size_t cap = 256 + (size_t)n * 400, len = 0;
    char *hdr = malloc(cap);
    if (!hdr) return 1;
    len += (size_t)snprintf(hdr + len, cap - len, "{");
    uint64_t off = 0;
    for (int k = 0; k < n; k++) {
        const char *nm = argv[first + k];
        ta[k] = nemo_find(&A, nm); tb[k] = nemo_find(&B, nm);
        if (!ta[k] || !tb[k] || ta[k]->ndim != 2 || tb[k]->ndim != 2 || ta[k]->count != tb[k]->count
            || ta[k]->shape[0] != tb[k]->shape[0] || !ta[k]->is_bf16 || !tb[k]->is_bf16) {
            fprintf(stderr, "%s: missing, not 2-D bf16, or shapes differ\n", nm); return 1;
        }
        uint64_t bytes = ta[k]->count * 4;
        len += (size_t)snprintf(hdr + len, cap - len, "%s\"%s.refine\":{\"dtype\":\"F32\",\"shape\":[%llu,%llu],\"data_offsets\":[%llu,%llu]}",
                                k ? "," : "", nm, (unsigned long long)ta[k]->shape[0], (unsigned long long)ta[k]->shape[1],
                                (unsigned long long)off, (unsigned long long)(off + bytes));
        off += bytes;
    }
    len += (size_t)snprintf(hdr + len, cap - len, "}");
    while (len % 8) hdr[len++] = ' ';
    FILE *f = fopen(out, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", out); return 1; }
    uint8_t lb[8]; for (int i = 0; i < 8; i++) lb[i] = (uint8_t)((uint64_t)len >> (8 * i));
    if (fwrite(lb, 1, 8, f) != 8 || fwrite(hdr, 1, len, f) != len) { fprintf(stderr, "write failed\n"); return 1; }
    for (int k = 0; k < n; k++) {
        uint16_t *a = nemo_load_bf16(&A, ta[k]), *b = nemo_load_bf16(&B, tb[k]);
        if (!a || !b) { fprintf(stderr, "%s: load failed\n", argv[first + k]); return 1; }
        uint64_t c = ta[k]->count;
        float *r = malloc((size_t)c * 4);
        if (!r) return 1;
        double sa = 0, sr = 0;
        for (uint64_t i = 0; i < c; i++) {
            uint32_t ua = (uint32_t)a[i] << 16, ub = (uint32_t)b[i] << 16; float fa, fb;
            memcpy(&fa, &ua, 4); memcpy(&fb, &ub, 4);
            r[i] = fa - fb;
            sa += (double)fa * fa; sr += (double)r[i] * r[i];
        }
        if (fwrite(r, 4, (size_t)c, f) != (size_t)c) { fprintf(stderr, "write failed\n"); return 1; }
        printf("%s.refine count %llu residual_rel_l2 %.4f\n", argv[first + k], (unsigned long long)c, sa > 0 ? sqrt(sr / sa) : 0.0);
        free(a); free(b); free(r);
    }
    if (fclose(f)) { fprintf(stderr, "close failed\n"); return 1; }
    nemo_checkpoint_close(&A); nemo_checkpoint_close(&B);
    return 0;
}
