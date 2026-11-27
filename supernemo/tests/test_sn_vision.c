/* test_sn_vision.c - the RADIO tower + mlp1 from a tiny pack against an independent double-precision reference.
 *   test_sn_vision PACK_PREFIX SHAPES_TSV CONFIG_JSON
 * The reference below is written separately from sn_vision.c (naive loops, doubles, the pixel shuffle as the literal
 * view/permute/view/permute of modeling_nemotron_h_omni.py, bilinear position resampling from first principles), and reads
 * the same expanded weights, so it checks the kernels, the chunking, the lanes and the token order, not the quantizer. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/sn_vision.h"

static int fails, checks;
#define CHECK(c, ...) do { checks++; if (!(c)) { fails++; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static const sn_pack *P; static sn_rd RD;
static double *W(const char *name, uint64_t *cnt)
{
    const sn_tensor *t = sn_pack_find(P, name);
    if (!t) { printf("missing %s\n", name); exit(2); }
    float *f = (float *)malloc(t->count * 4); double *d = (double *)malloc(t->count * 8);
    if (sn_expand(P, &RD, t, 0, t->count, f) != 0) { printf("read fail %s\n", name); exit(2); }
    for (uint64_t i = 0; i < t->count; i++) d[i] = f[i];
    free(f); if (cnt) *cnt = t->count; return d;
}
static double *Wn(const char *fmt, int i) { char n[256]; snprintf(n, sizeof n, fmt, i); return W(n, NULL); }

static void ref_ln(const double *x, double *y, const double *w, const double *b, int N, int D, double eps)
{
    for (int n = 0; n < N; n++) {
        double m = 0, v = 0; for (int i = 0; i < D; i++) m += x[n * D + i]; m /= D;
        for (int i = 0; i < D; i++) v += (x[n * D + i] - m) * (x[n * D + i] - m);
        v /= D;
        for (int i = 0; i < D; i++) y[n * D + i] = (x[n * D + i] - m) / sqrt(v + eps) * w[i] + b[i];
    }
}
static void ref_lin(const double *x, const double *Wt, const double *b, double *y, int N, int K, int R)
{
    for (int n = 0; n < N; n++) for (int r = 0; r < R; r++) { double s = b ? b[r] : 0; for (int k = 0; k < K; k++) s += Wt[(size_t)r * K + k] * x[(size_t)n * K + k]; y[(size_t)n * R + r] = s; }
}
static double gelu(double x) { return 0.5 * x * (1.0 + erf(x / sqrt(2.0))); }

/* torch F.interpolate(bilinear, align_corners=False), one axis, from first principles */
static double bilin(const double *src, int in, int out, int stride, int d)
{
    double s = ((double)d + 0.5) * in / out - 0.5; if (s < 0) s = 0;
    int a = (int)floor(s), b = a + 1 < in ? a + 1 : in - 1; double l = s - a;
    return src[a * stride] * (1 - l) + src[b * stride] * l;
}

static void ref_embed(const sn_vgeom *g, const float *pix, int H, int W_, double *out)
{
    int ps = g->patch, Hd = g->hidden, gh = H / ps, gw = W_ / ps, np = g->ncls + g->nreg, N = np + gh * gw, M = g->max_img / ps, m = gh > gw ? gh : gw;
    double *emb = W("vision_model.radio_model.model.patch_generator.embedder.weight", NULL);
    double *cls = W("vision_model.radio_model.model.patch_generator.cls_token.token", NULL);
    double *tab = W("vision_model.radio_model.model.patch_generator.pos_embed", NULL);
    double *X = (double *)calloc((size_t)N * Hd, 8), *T = (double *)calloc((size_t)N * Hd, 8), *A = (double *)calloc((size_t)N * Hd, 8);
    double *QKV = (double *)calloc((size_t)N * 3 * Hd, 8), *MH = (double *)calloc((size_t)N * g->mlp, 8);
    /* position embedding: [M][M][Hd] -> m x m -> crop gh x gw */
    double *r1 = (double *)malloc((size_t)m * M * Hd * 8), *pos = (double *)malloc((size_t)m * m * Hd * 8);
    for (int i = 0; i < m; i++) for (int c = 0; c < M; c++) for (int h = 0; h < Hd; h++) r1[((size_t)i * M + c) * Hd + h] = bilin(tab + (size_t)c * Hd + h, M, m, M * Hd, i);
    for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) for (int h = 0; h < Hd; h++) pos[((size_t)i * m + j) * Hd + h] = bilin(r1 + (size_t)i * M * Hd + h, M, m, Hd, j);
    memcpy(X, cls, (size_t)np * Hd * 8);
    for (int t = 0; t < gh * gw; t++) {
        int i = t / gw, j = t % gw; double *pv = (double *)malloc((size_t)3 * ps * ps * 8);
        for (int c = 0; c < 3; c++) for (int y = 0; y < ps; y++) for (int x = 0; x < ps; x++) pv[(c * ps + y) * ps + x] = pix[((size_t)c * H + i * ps + y) * W_ + j * ps + x];
        for (int h = 0; h < Hd; h++) { double s = 0; for (int k = 0; k < 3 * ps * ps; k++) s += emb[(size_t)h * 3 * ps * ps + k] * pv[k]; X[(size_t)(np + t) * Hd + h] = s + pos[((size_t)i * m + j) * Hd + h]; }
        free(pv);
    }
    int hd = Hd / g->heads;
    for (int L = 0; L < g->layers; L++) {
        double *n1w = Wn("vision_model.radio_model.model.blocks.%d.norm1.weight", L), *n1b = Wn("vision_model.radio_model.model.blocks.%d.norm1.bias", L);
        double *n2w = Wn("vision_model.radio_model.model.blocks.%d.norm2.weight", L), *n2b = Wn("vision_model.radio_model.model.blocks.%d.norm2.bias", L);
        double *qw = Wn("vision_model.radio_model.model.blocks.%d.attn.qkv.weight", L), *qb = Wn("vision_model.radio_model.model.blocks.%d.attn.qkv.bias", L);
        double *pw = Wn("vision_model.radio_model.model.blocks.%d.attn.proj.weight", L), *pb = Wn("vision_model.radio_model.model.blocks.%d.attn.proj.bias", L);
        double *f1w = Wn("vision_model.radio_model.model.blocks.%d.mlp.fc1.weight", L), *f1b = Wn("vision_model.radio_model.model.blocks.%d.mlp.fc1.bias", L);
        double *f2w = Wn("vision_model.radio_model.model.blocks.%d.mlp.fc2.weight", L), *f2b = Wn("vision_model.radio_model.model.blocks.%d.mlp.fc2.bias", L);
        ref_ln(X, T, n1w, n1b, N, Hd, g->eps);
        ref_lin(T, qw, qb, QKV, N, Hd, 3 * Hd);          /* chunk(3) along the output dim: q | k | v */
        for (int h = 0; h < g->heads; h++) for (int q = 0; q < N; q++) {
            double *s = (double *)malloc((size_t)N * 8), mx = -1e300, sum = 0;
            for (int j = 0; j < N; j++) { double d = 0; for (int k = 0; k < hd; k++) d += QKV[(size_t)q * 3 * Hd + h * hd + k] * QKV[(size_t)j * 3 * Hd + Hd + h * hd + k]; s[j] = d / sqrt((double)hd); if (s[j] > mx) mx = s[j]; }
            for (int j = 0; j < N; j++) { s[j] = exp(s[j] - mx); sum += s[j]; }
            for (int k = 0; k < hd; k++) { double a = 0; for (int j = 0; j < N; j++) a += s[j] / sum * QKV[(size_t)j * 3 * Hd + 2 * Hd + h * hd + k]; A[(size_t)q * Hd + h * hd + k] = a; }
            free(s);
        }
        ref_lin(A, pw, pb, T, N, Hd, Hd);
        for (size_t i = 0; i < (size_t)N * Hd; i++) X[i] += T[i];
        ref_ln(X, T, n2w, n2b, N, Hd, g->eps);
        ref_lin(T, f1w, f1b, MH, N, Hd, g->mlp);
        for (size_t i = 0; i < (size_t)N * g->mlp; i++) MH[i] = gelu(MH[i]);
        ref_lin(MH, f2w, f2b, T, N, g->mlp, Hd);
        for (size_t i = 0; i < (size_t)N * Hd; i++) X[i] += T[i];
        free(n1w); free(n1b); free(n2w); free(n2b); free(qw); free(qb); free(pw); free(pb); free(f1w); free(f1b); free(f2w); free(f2b);
    }
    double *fw = W("vision_projector.vision_final_layernorm.weight", NULL), *fb = W("vision_projector.vision_final_layernorm.bias", NULL);
    int npatch = gh * gw; double *F = (double *)malloc((size_t)npatch * Hd * 8);
    ref_ln(X + (size_t)np * Hd, F, fw, fb, npatch, Hd, g->eps);
    /* pixel_shuffle(scale 0.5), ps v2, literally: view(n,w,h/2,2c) permute(0,2,1,3) view(n,h/2,w/2,4c) permute(0,2,1,3).
       here "w" is the grid row count gh and "h" the grid column count gw (modeling code names them that way). */
    int c = Hd, h2 = gw / 2, w2 = gh / 2, cat = 4 * c, Nt = h2 * w2;
    double *p2 = (double *)malloc((size_t)npatch * Hd * 8), *o4 = (double *)malloc((size_t)Nt * cat * 8);
    for (int i = 0; i < gh; i++) for (int j = 0; j < h2; j++) for (int k = 0; k < 2 * c; k++)
        p2[((size_t)j * gh + i) * 2 * c + k] = F[((size_t)i * h2 + j) * 2 * c + k];      /* A1[i][j][k] -> A2[j][i][k] */
    for (int j = 0; j < h2; j++) for (int I = 0; I < w2; I++) for (int k = 0; k < cat; k++)
        o4[((size_t)I * h2 + j) * cat + k] = p2[((size_t)j * w2 + I) * cat + k];         /* A3[j][I][k] -> A4[I][j][k] */
    double *mnw = W("mlp1.0.weight", NULL), *m1 = W("mlp1.1.weight", NULL), *m3 = W("mlp1.3.weight", NULL);
    double *R = (double *)malloc((size_t)Nt * cat * 8), *H1 = (double *)malloc((size_t)Nt * g->proj_hidden * 8);
    for (int n = 0; n < Nt; n++) { double s = 0; for (int i = 0; i < cat; i++) s += o4[(size_t)n * cat + i] * o4[(size_t)n * cat + i]; s = 1.0 / sqrt(s / cat + 1e-5); for (int i = 0; i < cat; i++) R[(size_t)n * cat + i] = o4[(size_t)n * cat + i] * s * mnw[i]; }
    ref_lin(R, m1, NULL, H1, Nt, cat, g->proj_hidden);
    for (size_t i = 0; i < (size_t)Nt * g->proj_hidden; i++) H1[i] = H1[i] > 0 ? H1[i] * H1[i] : 0;
    ref_lin(H1, m3, NULL, out, Nt, g->proj_hidden, g->llm_hidden);
    free(emb); free(cls); free(tab); free(X); free(T); free(A); free(QKV); free(MH); free(r1); free(pos); free(fw); free(fb); free(F); free(p2); free(o4); free(mnw); free(m1); free(m3); free(R); free(H1);
}

static unsigned rs = 99;
static float rnd(void) { rs = rs * 1664525u + 1013904223u; return ((float)((rs >> 8) & 0xffff) / 32768.0f - 1.0f); }

int main(int argc, char **argv)
{
    if (argc != 4) { fprintf(stderr, "usage: test_sn_vision PACK SHAPES CONFIG\n"); return 2; }
    static sn_pack pk;
    if (sn_pack_open(&pk, argv[1], argv[2]) != 0) return 2;
    P = &pk; if (sn_rd_open(P, &RD) != 0) return 2;
    sn_vgeom g; if (sn_vgeom_load(&g, argv[3], 256) != 0) return 2;
    CHECK(g.hidden == 128 && g.layers == 2 && g.heads == 2 && g.mlp == 512 && g.ncls == 3 && g.nreg == 7 && g.shuffle == 2 && g.proj_hidden == 640, "geometry parsed");
    nemo_pool pa, pb; nemo_pool_init(&pa, 1); nemo_pool_init(&pb, 5);
    sn_vision *va = sn_vision_open(&pk, &pa, 1, &g), *vb = sn_vision_open(&pk, &pb, 5, &g);
    CHECK(va && vb, "open"); if (!va || !vb) return 1;
    int shapes[][2] = { {64, 64}, {32, 96}, {96, 32}, {96, 96}, {32, 32} };
    for (int s = 0; s < 5; s++) {
        int H = shapes[s][0], Wd = shapes[s][1];
        float *pix = (float *)malloc((size_t)3 * H * Wd * 4); for (size_t i = 0; i < (size_t)3 * H * Wd; i++) pix[i] = rnd() * 1.5f;
        int Nt = (H / 32) * (Wd / 32);
        float *oa = (float *)malloc((size_t)Nt * 256 * 4), *ob = (float *)malloc((size_t)Nt * 256 * 4); double *rf = (double *)malloc((size_t)Nt * 256 * 8);
        CHECK(sn_vision_embed(va, pix, H, Wd, oa) == 0 && sn_vision_embed(vb, pix, H, Wd, ob) == 0, "%dx%d embed runs", H, Wd);
        CHECK(memcmp(oa, ob, (size_t)Nt * 256 * 4) == 0, "%dx%d bit-identical on 1 and 5 lanes", H, Wd);
        ref_embed(&g, pix, H, Wd, rf);
        double num = 0, den = 0; int finite = 1;
        for (int i = 0; i < Nt * 256; i++) { double d = oa[i] - rf[i]; num += d * d; den += rf[i] * rf[i]; if (!isfinite(oa[i])) finite = 0; }
        double rel = sqrt(num / (den > 0 ? den : 1));
        printf("  %3dx%-3d tokens %d  rel L2 vs double reference %.3g  (|ref| rms %.3g)\n", H, Wd, Nt, rel, sqrt(den / (Nt * 256)));
        CHECK(finite && rel < 2e-4 && den > 0, "%dx%d matches the reference (rel %.3g)", H, Wd, rel);
        free(pix); free(oa); free(ob); free(rf);
    }
    CHECK(sn_vision_embed(va, (float[3]){0}, 40, 40, (float[1]){0}) == -2, "dimensions not a multiple of 32 are refused");
    sn_vision_close(va); sn_vision_close(vb); nemo_pool_shutdown(&pa); nemo_pool_shutdown(&pb);
    printf("%d checks, %d failed\n", checks, fails);
    return fails ? 1 : 0;
}

