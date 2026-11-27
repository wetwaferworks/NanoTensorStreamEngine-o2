/* fid-hess.c ... fidelity lab: activation second moments.
 *
 * Co-implemented by Cortana at JD's direction, 2026-11-27, for the
 * fidelity roadmap (docs/ROADMAP.md, method 1). Pure C99.
 *
 * Reads the float32 rows that nq2-infer-lab --calib-dump wrote for
 * one projection input (N rows of DIM) and writes the mean second
 * moment H = X^T X / N, the proxy Hessian of the layer-output error
 * ||(W - Q) x||^2 that the activation-aware quantizer minimizes.
 *
 * Accumulation: 512-row chunks are multiplied in float with the
 * fixed-tile kernel and added into a double accumulator in chunk
 * order, so the result is bit-identical for any --threads.
 *
 * Output file (.fidh): "FIDH" u32 dim, u64 rows, f64 trace, then
 * dim*dim float32 (full symmetric matrix, row-major).
 *
 * Usage: fid-hess --dim DIM --in X.f32 --out H.fidh [--threads N]
 */
#include "fid-common.h"

typedef struct { float *Hc; int64_t dim; } Mirror;

int main(int argc, char **argv)
{
    const char *in = NULL, *out = NULL;
    int64_t dim = 0;
    int threads = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--dim") && i + 1 < argc) dim = atoll(argv[++i]);
        else if (!strcmp(argv[i], "--in") && i + 1 < argc) in = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 1; }
    }
    if (!in || !out || dim <= 0) {
        fprintf(stderr, "usage: fid-hess --dim DIM --in X.f32 --out H.fidh [--threads N]\n");
        return 1;
    }
    g_fid_threads = threads > 0 ? threads : fid_cpu_count();
    FILE *f = fopen(in, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", in); return 1; }
    uint64_t sz = fid_fsize(f);
    fid_seek(f, 0);
    if (sz % (uint64_t)(dim * 4) != 0) {
        fprintf(stderr, "%s: %llu bytes is not a whole number of %lld-float rows\n",
                in, (unsigned long long)sz, (long long)dim);
        return 1;
    }
    int64_t rows = (int64_t)(sz / (uint64_t)(dim * 4));
    const int64_t R = 512;
    float *X = (float *)fid_alloc((size_t)R * dim * sizeof(float));
    float *XT = (float *)fid_alloc((size_t)R * dim * sizeof(float));
    float *Hc = (float *)fid_alloc((size_t)dim * dim * sizeof(float));
    double *Hd = (double *)fid_calloc((size_t)dim * dim, sizeof(double));
    double t0 = fid_now();
    int64_t nonfinite = 0;
    for (int64_t r0 = 0; r0 < rows; r0 += R) {
        int64_t m = rows - r0 < R ? rows - r0 : R;
        if (fread(X, sizeof(float), (size_t)(m * dim), f) != (size_t)(m * dim)) {
            fprintf(stderr, "short read\n");
            return 1;
        }
        for (int64_t r = 0; r < m; r++)
            for (int64_t c = 0; c < dim; c++) {
                float v = X[r * dim + c];
                if (!isfinite(v)) { nonfinite++; v = 0.0f; X[r * dim + c] = 0.0f; }
                XT[c * m + r] = v;
            }
        memset(Hc, 0, (size_t)dim * dim * sizeof(float));
        fid_gemm_ex(Hc, dim, XT, m, X, dim, dim, m, dim, 1);
        for (int64_t i = 0; i < dim; i++)
            for (int64_t j = i; j < dim; j++)
                Hd[i * dim + j] += Hc[i * dim + j];
    }
    fclose(f);
    double tr = 0.0;
    for (int64_t i = 0; i < dim; i++) {
        for (int64_t j = i; j < dim; j++) {
            float v = (float)(Hd[i * dim + j] / (double)rows);
            Hc[i * dim + j] = v;
            Hc[j * dim + i] = v;
        }
        tr += Hd[i * dim + i] / (double)rows;
    }
    FILE *o = fopen(out, "wb");
    if (!o) { fprintf(stderr, "cannot open %s\n", out); return 1; }
    uint32_t hdr[2] = { 0x48444946u, (uint32_t)dim };   /* "FIDH" */
    uint64_t nr = (uint64_t)rows;
    if (fwrite(hdr, 4, 2, o) != 2 || fwrite(&nr, 8, 1, o) != 1
            || fwrite(&tr, 8, 1, o) != 1
            || fwrite(Hc, sizeof(float), (size_t)dim * dim, o) != (size_t)(dim * dim)) {
        fprintf(stderr, "short write\n");
        return 1;
    }
    fclose(o);
    uint64_t dg = fid_fnv(Hc, (size_t)dim * dim * sizeof(float), FID_FNV_SEED);
    printf("%s: rows %lld dim %lld trace %.6e nonfinite %lld digest %016llx %.1f s\n",
           out, (long long)rows, (long long)dim, tr, (long long)nonfinite,
           (unsigned long long)dg, fid_now() - t0);
    free(X); free(XT); free(Hc); free(Hd);
    return 0;
}
