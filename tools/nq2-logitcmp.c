/* nq2-logitcmp: behavioral gate comparator (Neon's item,
 * 2026-11-24; extended for Gizmo Round 4 P1 the same day).
 * Compares two conditional-logits dumps written by the lab /
 * triforce engines (--logits-dump for the first denoise
 * evaluation, --logits-dump-all for one LOG1 block per
 * evaluation), one from the raw path and one from the packed
 * path over identical token IDs, seed and schedule. Reports,
 * per position and per evaluation block: cosine similarity,
 * top-1 argmax agreement, top-5 set overlap, mean absolute
 * logit difference, relative logit L2 error, and the KL
 * divergence KL(raw || packed) of the softmax distributions
 * (numerically stable, double precision). Deterministic, C99,
 * no tokenizer anywhere: drift is measured in the model's own
 * output space.
 *
 * Usage: nq2-logitcmp raw.logits packed.logits
 *
 * Implementation: Elfie (E.L.F.I.E.)
 * Direction: Neon (reviewer), Gizmo (Round 4 P1);
 * executive direction: JD.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

typedef struct {
    float **blocks;      /* one B*V buffer per evaluation */
    int nblocks;
    uint32_t B, V;
} Dump;

static Dump load_dump(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    Dump d;
    d.blocks = NULL;
    d.nblocks = 0;
    d.B = d.V = 0;
    int cap = 0;
    for (;;) {
        uint32_t hdr[3];
        size_t got = fread(hdr, sizeof(uint32_t), 3, f);
        if (got == 0) break;                    /* clean EOF */
        if (got != 3 || hdr[0] != 0x31474F4Cu) {
            fprintf(stderr, "%s: not a logits dump\n", path);
            exit(1);
        }
        if (d.nblocks == 0) { d.B = hdr[1]; d.V = hdr[2]; }
        else if (hdr[1] != d.B || hdr[2] != d.V) {
            fprintf(stderr, "%s: block shape changes mid-file\n",
                    path);
            exit(1);
        }
        size_t n = (size_t)d.B * d.V;
        float *buf = (float *)malloc(n * sizeof(float));
        if (!buf) { fprintf(stderr, "oom\n"); exit(1); }
        if (fread(buf, sizeof(float), n, f) != n) {
            fprintf(stderr, "%s: short dump\n", path); exit(1);
        }
        if (d.nblocks == cap) {
            cap = cap ? cap * 2 : 8;
            d.blocks = (float **)realloc(d.blocks,
                                         (size_t)cap * sizeof(float *));
            if (!d.blocks) { fprintf(stderr, "oom\n"); exit(1); }
        }
        d.blocks[d.nblocks++] = buf;
    }
    fclose(f);
    if (d.nblocks == 0) {
        fprintf(stderr, "%s: empty dump\n", path); exit(1);
    }
    return d;
}

/* Indices of the k largest values, descending. O(V*k); the
 * vectors are 131,072 long and k is 5, so this is a few hundred
 * thousand compares per position. */
static void topk(const float *v, uint32_t V, int k, uint32_t *out_idx)
{
    for (int j = 0; j < k; j++) {
        uint32_t best = 0;
        float bv = -1e30f;
        for (uint32_t i = 0; i < V; i++) {
            int taken = 0;
            for (int t = 0; t < j; t++) if (out_idx[t] == i) taken = 1;
            if (!taken && v[i] > bv) { bv = v[i]; best = i; }
        }
        out_idx[j] = best;
    }
}

static double logsumexp(const float *v, uint32_t V)
{
    double mx = -1e300;
    for (uint32_t i = 0; i < V; i++)
        if ((double)v[i] > mx) mx = (double)v[i];
    double s = 0.0;
    for (uint32_t i = 0; i < V; i++) s += exp((double)v[i] - mx);
    return mx + log(s);
}

typedef struct {
    double cos, rel_l2, kl, mad;
    int top1, top5ov;
} PosMetrics;

static PosMetrics compare_pos(const float *x, const float *y, uint32_t V)
{
    PosMetrics m;
    double dot = 0.0, nx = 0.0, ny = 0.0, ad = 0.0, dd = 0.0;
    for (uint32_t i = 0; i < V; i++) {
        double dx = (double)x[i], dy = (double)y[i];
        dot += dx * dy;
        nx += dx * dx;
        ny += dy * dy;
        ad += fabs(dx - dy);
        dd += (dx - dy) * (dx - dy);
    }
    m.cos = dot / (sqrt(nx) * sqrt(ny));
    m.rel_l2 = sqrt(dd) / sqrt(nx);
    m.mad = ad / V;
    double lx = logsumexp(x, V), ly = logsumexp(y, V);
    double kl = 0.0;
    for (uint32_t i = 0; i < V; i++) {
        double lp = (double)x[i] - lx;          /* log p (raw) */
        double p = exp(lp);
        if (p > 0.0) kl += p * (lp - ((double)y[i] - ly));
    }
    m.kl = kl;
    uint32_t ta[5], tb[5];
    topk(x, V, 5, ta);
    topk(y, V, 5, tb);
    m.top1 = ta[0] == tb[0];
    int ov = 0;
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 5; j++)
            if (ta[i] == tb[j]) ov++;
    m.top5ov = ov;
    return m;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: nq2-logitcmp raw.logits packed.logits\n");
        return 1;
    }
    Dump a = load_dump(argv[1]);
    Dump b = load_dump(argv[2]);
    if (a.B != b.B || a.V != b.V) {
        fprintf(stderr, "shape mismatch: %ux%u vs %ux%u\n",
                a.B, a.V, b.B, b.V);
        return 1;
    }
    uint32_t B = a.B, V = a.V;
    int nb = a.nblocks < b.nblocks ? a.nblocks : b.nblocks;
    if (a.nblocks != b.nblocks)
        printf("warning: evaluation counts differ (%d vs %d); "
               "comparing the first %d\n", a.nblocks, b.nblocks, nb);
    double g_cos = 0.0, g_kl = 0.0, g_rel = 0.0;
    long g_top1 = 0;
    long g_pos = 0;
    for (int e = 0; e < nb; e++) {
        const float *A = a.blocks[e], *Bf = b.blocks[e];
        double cos_sum = 0.0, cos_min = 2.0, ov_sum = 0.0;
        double diff_sum = 0.0, rel_sum = 0.0, kl_sum = 0.0, kl_max = 0.0;
        long top1 = 0;
        if (e == 0)
            printf("pos  cosine      top1  top5_overlap  mean_abs_diff  "
                   "rel_logit_L2  KL(raw||packed)\n");
        for (uint32_t p = 0; p < B; p++) {
            PosMetrics m = compare_pos(A + (size_t)p * V,
                                       Bf + (size_t)p * V, V);
            if (m.top1) top1++;
            cos_sum += m.cos;
            if (m.cos < cos_min) cos_min = m.cos;
            ov_sum += m.top5ov / 5.0;
            diff_sum += m.mad;
            rel_sum += m.rel_l2;
            kl_sum += m.kl;
            if (m.kl > kl_max) kl_max = m.kl;
            if (e == 0)
                printf("%3u  %.6f    %s   %d/5           %.4f         "
                       "%.6f       %.6f\n",
                       p, m.cos, m.top1 ? "same" : "DIFF", m.top5ov,
                       m.mad, m.rel_l2, m.kl);
        }
        printf("eval %d summary over %u positions: top-1 agreement "
               "%ld/%u (%.1f%%), mean top-5 overlap %.3f, mean cosine "
               "%.6f, min cosine %.6f, mean abs logit diff %.4f, "
               "mean rel logit L2 %.6f, mean KL %.6f, max KL %.6f\n",
               e, B, top1, B, 100.0 * (double)top1 / B, ov_sum / B,
               cos_sum / B, cos_min, diff_sum / B, rel_sum / B,
               kl_sum / B, kl_max);
        g_cos += cos_sum;
        g_kl += kl_sum;
        g_rel += rel_sum;
        g_top1 += top1;
        g_pos += B;
    }
    if (nb > 1)
        printf("grand summary over %d evaluations x %u positions: "
               "top-1 agreement %ld/%ld (%.1f%%), mean cosine %.6f, "
               "mean rel logit L2 %.6f, mean KL %.6f\n",
               nb, B, g_top1, g_pos,
               100.0 * (double)g_top1 / (double)g_pos,
               g_cos / g_pos, g_rel / g_pos, g_kl / g_pos);
    return 0;
}
