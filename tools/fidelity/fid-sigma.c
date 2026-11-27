/* fid-sigma.c ... fidelity lab: layer-major teacher-forced evaluator
 * and calibrator for Sigma-family checkpoints.
 *
 * Co-implemented by Cortana at JD's direction, 2026-11-27, for the
 * fidelity roadmap (docs/ROADMAP.md). Pure C99, zero mutexes,
 * bit-identical results for any --threads.
 *
 * The task is the one nq2-infer-lab --teacher scores (and the one the
 * model was trained on): each line of --teacher is a real-text token
 * sequence; its first block is the clean prefix; every later block is
 * noised to each gamma from its own true embedding, z = alpha * E[id]
 * + sigma * eps (eps from the same counter-based generator and keys
 * as the lab engine), denoised once with the clean tokens before it
 * as context, and scored. Because every clean token is known, all
 * rows of all sequences, clean and noisy, go through each layer
 * together, so every weight is read once per evaluation instead of
 * once per block: about 20x less weight traffic, which is what lets a
 * full quantization study run on one workstation.
 *
 * The arithmetic follows nq2-infer-lab: the same RMSNorm, YaRN rotary
 * tables, bf16 key/value cache for keys outside a row's own block,
 * float keys inside it, double softmax, SiLU gate, prior term and
 * gamma embedding. Products run through the fixed-order float kernel
 * in fid-common.h, so logits agree with the lab engine to float
 * rounding (checked by --check against a lab --teacher-out file).
 *
 * Modes:
 *   --write-ref FILE   score against the true tokens only and write
 *                      the logits (the raw model's reference);
 *   --ref FILE         also score against that reference (KL, top-1,
 *                      top-5 agreement);
 *   --hess-out DIR     calibration: accumulate E[x x^T] for every
 *                      projection input (L%02d.qkv, .o, .gu, .down)
 *                      and the head input (head), over all rows, and
 *                      write .fidh files for fid-quant.
 * Order of rows in the reference file: sequence, block, gamma,
 * position, matching the lab engine.
 *
 * Usage:
 *   fid-sigma --dir MODEL --teacher IDS [--gammas 6,1.5] [--block 32]
 *             [--seed 42] [--max-seqs N] [--threads N]
 *             [--write-ref F | --ref F] [--out TSV] [--hess-out DIR]
 */
#include "fid-common.h"

static int H, L, NH, NKV, HD = 128, FFN, V, ED;
static const double EPS = (double)1e-5f;   /* the engine passes a float eps */

typedef struct { FidShard sh[8]; int ns; } Model;

static FidTensor *find_t(Model *m, const char *name, int *shard)
{
    for (int s = 0; s < m->ns; s++)
        for (int i = 0; i < m->sh[s].n; i++)
            if (!strcmp(m->sh[s].t[i].name, name)) { *shard = s; return &m->sh[s].t[i]; }
    return NULL;
}

static float *load_f(Model *m, const char *name, uint64_t need)
{
    int s;
    FidTensor *t = find_t(m, name, &s);
    if (!t) { fprintf(stderr, "missing tensor %s\n", name); exit(1); }
    if (need && t->count != need) {
        fprintf(stderr, "tensor %s has %llu elements, expected %llu\n", name,
                (unsigned long long)t->count, (unsigned long long)need);
        exit(1);
    }
    float *b = (float *)fid_alloc(t->count * sizeof(float));
    if (fid_tensor_read(&m->sh[s], t, 0, t->count, b) != 0) {
        fprintf(stderr, "read failed %s\n", name); exit(1); }
    return b;
}

static float *g_cos, *g_sin;

static void rope_init(int max_pos)
{
    const double theta = 1000000.0, orig_max = 16384.0;
    const double factor = 4096.0 / orig_max;
    const double beta_fast = 32.0, beta_slow = 1.0;
    int half = HD / 2;
    double inv_freq[64];
    double corr = 2.0 * log(theta);
    double low_d = HD * log(orig_max / (beta_fast * 2.0 * 3.141592653589793)) / corr;
    double high_d = HD * log(orig_max / (beta_slow * 2.0 * 3.141592653589793)) / corr;
    int low = (int)floor(low_d); if (low < 0) low = 0;
    int high = (int)ceil(high_d); if (high > HD - 1) high = HD - 1;
    for (int i = 0; i < half; i++) {
        double pos_freq = pow(theta, (double)(2 * i) / HD);
        double inv_extrap = 1.0 / pos_freq;
        double inv_interp = 1.0 / (factor * pos_freq);
        double ramp = ((double)i - low) / (high - low > 0.001 ? high - low : 0.001);
        if (ramp < 0.0) ramp = 0.0;
        if (ramp > 1.0) ramp = 1.0;
        double ef = 1.0 - ramp;
        inv_freq[i] = inv_interp * (1.0 - ef) + inv_extrap * ef;
    }
    g_cos = (float *)fid_alloc((size_t)max_pos * HD * sizeof(float));
    g_sin = (float *)fid_alloc((size_t)max_pos * HD * sizeof(float));
    for (int p = 0; p < max_pos; p++)
        for (int i = 0; i < half; i++) {
            double ang = inv_freq[i] * p;
            float c = (float)cos(ang), s = (float)sin(ang);
            g_cos[(size_t)p * HD + i] = c; g_cos[(size_t)p * HD + half + i] = c;
            g_sin[(size_t)p * HD + i] = s; g_sin[(size_t)p * HD + half + i] = s;
        }
}

static uint16_t bf16r(float f) { return fid_f32_to_bf16(f); }

static void rmsnorm_rows(float *out, const float *x, const float *w, int64_t n, int dim)
{
    for (int64_t i = 0; i < n; i++) {
        const float *row = x + i * dim;
        float *o = out + i * dim;
        double ss = 0;
        for (int k = 0; k < dim; k++) ss += (double)row[k] * row[k];
        double inv = 1.0 / sqrt(ss / dim + EPS);
        for (int k = 0; k < dim; k++) o[k] = (float)(row[k] * inv) * w[k];
    }
}

/* ---------- rows ---------- */

typedef struct {
    int seq, pos, id;
    int noisy;
    int gi, blk;           /* noisy rows */
    int64_t own_lo, own_hi;   /* rows of the same attention group (current-batch keys) */
    int64_t c_lo, c_hi;       /* clean rows whose keys come from the bf16 cache */
} Row;

static Row *R;
static int64_t NR;

/* ---------- attention ---------- */

typedef struct {
    const float *Q, *K, *Vv;     /* float, after rotary */
    const uint16_t *Kb, *Vb;     /* bf16 copies */
    float *Out;
} AttC;

static void att_item(void *vc, int64_t it)
{
    AttC *a = (AttC *)vc;
    int64_t i = it / NH;
    int h = (int)(it % NH);
    int groups = NH / NKV, kh = h / groups;
    int kvd = NKV * HD, qd = NH * HD;
    const Row *r = &R[i];
    const float *qv = a->Q + i * qd + h * HD;
    int64_t nc = r->c_hi - r->c_lo, no = r->own_hi - r->own_lo;
    int64_t tot = nc + no;
    float sc_stack[1024];
    float *scores = tot <= 1024 ? sc_stack : (float *)fid_alloc(sizeof(float) * tot);
    double mx = -1e30;
    for (int64_t j = 0; j < tot; j++) {
        double dot = 0.0;
        int allowed = 1;
        if (j < nc) {
            const uint16_t *kv = a->Kb + (r->c_lo + j) * kvd + kh * HD;
            for (int d = 0; d < HD; d++) dot += (double)qv[d] * fid_bf16_to_f32(kv[d]);
        } else {
            int64_t jr = r->own_lo + (j - nc);
            if (!r->noisy && R[jr].pos > r->pos) allowed = 0;
            const float *kv = a->K + jr * kvd + kh * HD;
            if (allowed) for (int d = 0; d < HD; d++) dot += (double)qv[d] * kv[d];
        }
        if (!allowed) { scores[j] = -1e30f; continue; }
        double s = dot * (1.0 / sqrt((double)HD));
        scores[j] = (float)s;
        if (s > mx) mx = s;
    }
    double sum = 0.0;
    for (int64_t j = 0; j < tot; j++) {
        if (scores[j] <= -1e29f) { scores[j] = 0.0f; continue; }
        scores[j] = (float)exp((double)scores[j] - mx);
        sum += scores[j];
    }
    float *ov = a->Out + i * qd + h * HD;
    for (int d = 0; d < HD; d++) ov[d] = 0.0f;
    if (sum > 0.0) {
        double inv = 1.0 / sum;
        for (int64_t j = 0; j < tot; j++) {
            if (scores[j] == 0.0f) continue;
            double p = scores[j] * inv;
            if (j < nc) {
                const uint16_t *vv = a->Vb + (r->c_lo + j) * kvd + kh * HD;
                for (int d = 0; d < HD; d++) ov[d] += (float)(p * fid_bf16_to_f32(vv[d]));
            } else {
                const float *vv = a->Vv + (r->own_lo + (j - nc)) * kvd + kh * HD;
                for (int d = 0; d < HD; d++) ov[d] += (float)(p * vv[d]);
            }
        }
    }
    if (scores != sc_stack) free(scores);
}

/* ---------- Hessian accumulation ---------- */

typedef struct { double *acc; float *tmp, *xt; int64_t dim, rows; } HAcc;

static void hacc_init(HAcc *h, int64_t dim)
{
    h->dim = dim; h->rows = 0;
    h->acc = (double *)fid_calloc((size_t)dim * dim, sizeof(double));
    h->tmp = (float *)fid_alloc((size_t)dim * dim * sizeof(float));
    h->xt = (float *)fid_alloc((size_t)dim * 512 * sizeof(float));
}

/* add rows of X (n x dim) in 512-row chunks (fixed boundaries) */
static void hacc_add(HAcc *h, const float *X, int64_t n)
{
    int64_t dim = h->dim;
    for (int64_t r0 = 0; r0 < n; r0 += 512) {
        int64_t m = n - r0 < 512 ? n - r0 : 512;
        for (int64_t r = 0; r < m; r++)
            for (int64_t c = 0; c < dim; c++) h->xt[c * m + r] = X[(r0 + r) * dim + c];
        memset(h->tmp, 0, (size_t)dim * dim * sizeof(float));
        fid_gemm_ex(h->tmp, dim, h->xt, m, X + r0 * dim, dim, dim, m, dim, 1);
        for (int64_t i = 0; i < dim; i++)
            for (int64_t j = i; j < dim; j++) h->acc[i * dim + j] += h->tmp[i * dim + j];
    }
    h->rows += n;
}

static void hacc_write(HAcc *h, const char *dir, const char *name)
{
    int64_t dim = h->dim;
    double tr = 0;
    for (int64_t i = 0; i < dim; i++) {
        for (int64_t j = i; j < dim; j++) {
            float v = (float)(h->acc[i * dim + j] / (double)h->rows);
            h->tmp[i * dim + j] = v; h->tmp[j * dim + i] = v;
        }
        tr += h->acc[i * dim + i] / (double)h->rows;
    }
    char path[2200];
    snprintf(path, sizeof(path), "%s/%s.fidh", dir, name);
    FILE *o = fopen(path, "wb");
    if (!o) { fprintf(stderr, "cannot write %s\n", path); exit(1); }
    uint32_t hdr[2] = { 0x48444946u, (uint32_t)dim };
    uint64_t nr = (uint64_t)h->rows;
    fwrite(hdr, 4, 2, o); fwrite(&nr, 8, 1, o); fwrite(&tr, 8, 1, o);
    fwrite(h->tmp, sizeof(float), (size_t)dim * dim, o);
    fclose(o);
    printf("  hessian %s rows %lld trace %.4e digest %016llx\n", name, (long long)h->rows, tr,
           (unsigned long long)fid_fnv(h->tmp, (size_t)dim * dim * 4, FID_FNV_SEED));
    free(h->acc); free(h->tmp); free(h->xt);
}

/* ---------- teacher input ---------- */

typedef struct { int *ids; int n; } Seq;

static int load_seqs(const char *path, Seq **out, int maxs)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    int cap = 16, ns = 0, icap = 0, n = 0, have = 0, c;
    long v = 0;
    Seq *s = (Seq *)fid_calloc(cap, sizeof(Seq));
    int *ids = NULL;
    for (;;) {
        c = fgetc(f);
        if (c >= '0' && c <= '9') { v = v * 10 + (c - '0'); have = 1; continue; }
        if (have) {
            if (n == icap) { icap = icap ? icap * 2 : 1024; ids = (int *)realloc(ids, sizeof(int) * icap); }
            ids[n++] = (int)v; v = 0; have = 0;
        }
        if (c == '\n' || c == EOF) {
            if (n > 0 && ns < maxs) {
                if (ns == cap) { cap *= 2; s = (Seq *)realloc(s, sizeof(Seq) * cap); }
                s[ns].ids = ids; s[ns].n = n; ns++; ids = NULL; n = 0; icap = 0;
            } else { n = 0; }
            if (c == EOF) break;
        }
    }
    fclose(f);
    free(ids);
    *out = s;
    return ns;
}

static uint64_t t_mix(uint64_t x) { return fid_mix(x); }

static double t_normal(uint64_t key, uint64_t i)
{
    uint64_t a = t_mix(key ^ t_mix(2 * (i / 2)));
    uint64_t b = t_mix(key ^ t_mix(2 * (i / 2) + 1));
    double u1 = ((double)(a >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    double u2 = (double)(b >> 11) * (1.0 / 9007199254740992.0);
    double r = sqrt(-2.0 * log(u1));
    double t = 2.0 * 3.14159265358979323846 * u2;
    return (i & 1) ? r * sin(t) : r * cos(t);
}

/* ---------- scoring ---------- */

typedef struct { double n, top1, top5, kl, klmax, ce_ref, ce_q, acc_ref, acc_q; } St;

int main(int argc, char **argv)
{
    const char *dir = NULL, *tpath = NULL, *gam_s = "6", *wref = NULL, *rref = NULL;
    const char *outp = NULL, *hout = NULL;
    int block = 32, threads = 0, maxs = 1 << 20;
    uint64_t seed = 42;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        int more = i + 1 < argc;
        if (!strcmp(a, "--dir") && more) dir = argv[++i];
        else if (!strcmp(a, "--teacher") && more) tpath = argv[++i];
        else if (!strcmp(a, "--gammas") && more) gam_s = argv[++i];
        else if (!strcmp(a, "--block") && more) block = atoi(argv[++i]);
        else if (!strcmp(a, "--seed") && more) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(a, "--max-seqs") && more) maxs = atoi(argv[++i]);
        else if (!strcmp(a, "--threads") && more) threads = atoi(argv[++i]);
        else if (!strcmp(a, "--write-ref") && more) wref = argv[++i];
        else if (!strcmp(a, "--ref") && more) rref = argv[++i];
        else if (!strcmp(a, "--out") && more) outp = argv[++i];
        else if (!strcmp(a, "--hess-out") && more) hout = argv[++i];
        else { fprintf(stderr, "unknown or incomplete option %s\n", a); return 1; }
    }
    if (!dir || !tpath) {
        fprintf(stderr, "usage: fid-sigma --dir MODEL --teacher IDS [--gammas 6,1.5] [--write-ref F | --ref F] [--out TSV] [--hess-out DIR]\n");
        return 1;
    }
    g_fid_threads = threads > 0 ? threads : fid_cpu_count();
    double t0 = fid_now();
    Model m;
    memset(&m, 0, sizeof(m));
    for (int s = 1; s <= 8; s++)
        for (int n = 1; n <= 8; n++) {
            char p[2200];
            snprintf(p, sizeof(p), "%s/model-%05d-of-%05d.safetensors", dir, s, n);
            FILE *f = fopen(p, "rb");
            if (!f) continue;
            fclose(f);
            if (fid_shard_open(&m.sh[m.ns], p) != 0) { fprintf(stderr, "cannot open %s\n", p); return 1; }
            m.ns++;
        }
    if (!m.ns) { fprintf(stderr, "no shards in %s\n", dir); return 1; }
    int sx;
    FidTensor *te = find_t(&m, "encoder.embed_tokens.weight", &sx);
    FidTensor *tq = find_t(&m, "encoder.layers.0.self_attn.q_proj.weight", &sx);
    FidTensor *tk = find_t(&m, "encoder.layers.0.self_attn.k_proj.weight", &sx);
    FidTensor *tg = find_t(&m, "encoder.layers.0.mlp.gate_proj.weight", &sx);
    FidTensor *tE = find_t(&m, "replaid_embedding_matrix.matrix", &sx);
    if (!te || !tq || !tk || !tg || !tE) { fprintf(stderr, "not a Sigma-family model\n"); return 1; }
    V = (int)te->dims[0]; H = (int)te->dims[1]; NH = (int)tq->dims[0] / HD; NKV = (int)tk->dims[0] / HD;
    FFN = (int)tg->dims[0]; ED = (int)tE->dims[1];
    L = 0;
    for (int s = 0; s < m.ns; s++)
        for (int i = 0; i < m.sh[s].n; i++) {
            int li;
            if (sscanf(m.sh[s].t[i].name, "encoder.layers.%d.", &li) == 1 && li + 1 > L) L = li + 1;
        }
    double gam[16];
    int ng = 0;
    {
        char b[256];
        snprintf(b, sizeof(b), "%s", gam_s);
        for (char *t = strtok(b, ","); t && ng < 16; t = strtok(NULL, ",")) gam[ng++] = atof(t);
    }
    Seq *seqs;
    int nseq = load_seqs(tpath, &seqs, maxs);
    if (nseq <= 0) { fprintf(stderr, "no sequences in %s\n", tpath); return 1; }
    /* rows: clean rows of every sequence, then noisy rows in
     * (seq, block, gamma, position) order, the reference order */
    int64_t ncl = 0, nno = 0;
    int maxlen = 0;
    for (int s = 0; s < nseq; s++) {
        int nb = seqs[s].n / block;
        if (nb < 2) continue;
        ncl += (int64_t)nb * block;
        nno += (int64_t)(nb - 1) * block * ng;
        if (nb * block > maxlen) maxlen = nb * block;
    }
    NR = ncl + nno;
    R = (Row *)fid_calloc(NR, sizeof(Row));
    int64_t *cbase = (int64_t *)fid_calloc(nseq, sizeof(int64_t));
    {
        int64_t at = 0;
        for (int s = 0; s < nseq; s++) {
            int nb = seqs[s].n / block;
            if (nb < 2) continue;
            cbase[s] = at;
            for (int p = 0; p < nb * block; p++) {
                Row *r = &R[at + p];
                r->seq = s; r->pos = p; r->id = seqs[s].ids[p]; r->noisy = 0;
                int b = p / block;
                r->c_lo = at; r->c_hi = at + (int64_t)b * block;     /* earlier blocks: cache */
                r->own_lo = at + (int64_t)b * block; r->own_hi = r->own_lo + block;
            }
            at += (int64_t)nb * block;
        }
        for (int s = 0; s < nseq; s++) {
            int nb = seqs[s].n / block;
            if (nb < 2) continue;
            for (int b = 1; b < nb; b++)
                for (int gi = 0; gi < ng; gi++) {
                    int64_t lo = at;
                    for (int i = 0; i < block; i++) {
                        Row *r = &R[at++];
                        r->seq = s; r->pos = b * block + i; r->id = seqs[s].ids[b * block + i];
                        r->noisy = 1; r->gi = gi; r->blk = b;
                        r->c_lo = cbase[s]; r->c_hi = cbase[s] + (int64_t)b * block;
                        r->own_lo = lo; r->own_hi = lo + block;
                    }
                }
        }
    }
    for (int64_t i = 0; i < NR; i++)
        if (R[i].id < 0 || R[i].id >= V) { fprintf(stderr, "token id %d out of range\n", R[i].id); return 1; }
    printf("fid-sigma: %d layers, hidden %d, heads %d/%d, ffn %d, vocab %d; %d sequences, %lld clean + %lld noisy rows, gammas %s, threads %d\n",
           L, H, NH, NKV, FFN, V, nseq, (long long)ncl, (long long)nno, gam_s, g_fid_threads);
    rope_init(maxlen + 8);
    int qd = NH * HD, kvd = NKV * HD;
    /* small tensors */
    float *Enorm = load_f(&m, "replaid_embedding_matrix.matrix", (uint64_t)V * ED);
    for (int v = 0; v < V; v++) {
        double nrm = 0;
        for (int d = 0; d < ED; d++) nrm += (double)Enorm[v * ED + d] * Enorm[v * ED + d];
        double inv = 1.0 / (sqrt(nrm) + 1e-8);
        for (int d = 0; d < ED; d++) Enorm[v * ED + d] = (float)(Enorm[v * ED + d] * inv);
    }
    float *Win = load_f(&m, "replaid_input_linear.weight", (uint64_t)H * ED);
    float *Wgam = load_f(&m, "replaid_gamma_linear.weight", (uint64_t)H * 64);
    float *nfin = load_f(&m, "encoder.norm.weight", H);
    /* initial hidden */
    float *h = (float *)fid_alloc((size_t)NR * H * sizeof(float));
    double *zall = (double *)fid_alloc((size_t)(nno > 0 ? nno : 1) * ED * sizeof(double));
    {
        int s0 = 0;
        FidTensor *emb = find_t(&m, "encoder.embed_tokens.weight", &s0);
        for (int64_t i = 0; i < ncl; i++)
            if (fid_tensor_read(&m.sh[s0], emb, (uint64_t)R[i].id * H, H, h + i * H) != 0) {
                fprintf(stderr, "embed read failed\n"); return 1; }
        float *grow = (float *)fid_alloc(sizeof(float) * ng * H);
        for (int gi = 0; gi < ng; gi++) {
            float ge[64];
            for (int j = 0; j < 32; j++) {
                double fr = exp(-5.0 + (10.0 * j) / 31.0);
                ge[j] = (float)sin(fr * gam[gi]); ge[32 + j] = (float)cos(fr * gam[gi]);
            }
            for (int d = 0; d < H; d++) {
                double acc = 0;
                for (int k = 0; k < 64; k++) acc += (double)ge[k] * Wgam[d * 64 + k];
                grow[gi * H + d] = (float)acc;
            }
        }
        for (int64_t i = ncl; i < NR; i++) {
            Row *r = &R[i];
            double g = gam[r->gi];
            double asq = 1.0 / (1.0 + exp(g)), ssq = 1.0 / (1.0 + exp(-g));
            double alpha = sqrt(asq), sigma = sqrt(ssq);
            double zscale = 1.0 / sqrt(asq / ED + ssq);
            uint64_t key = t_mix(seed ^ t_mix(((uint64_t)r->seq << 40) ^ ((uint64_t)r->blk << 20) ^ (uint64_t)r->gi));
            int ib = r->pos - r->blk * block;
            double *z = zall + (i - ncl) * ED;
            float xin[64];
            for (int d = 0; d < ED; d++) {
                z[d] = alpha * Enorm[(size_t)r->id * ED + d] + sigma * t_normal(key, (uint64_t)ib * ED + d);
                xin[d] = (float)(z[d] * zscale);
            }
            for (int dd = 0; dd < H; dd++) {
                double acc = 0;
                for (int k = 0; k < ED; k++) acc += (double)xin[k] * Win[dd * ED + k];
                h[i * H + dd] = (float)acc + grow[r->gi * H + dd];
            }
        }
        free(grow);
    }
    printf("setup %.1f s\n", fid_now() - t0);
    float *xn = (float *)fid_alloc((size_t)NR * (H > qd ? H : qd) * sizeof(float));
    float *Q = (float *)fid_alloc((size_t)NR * qd * sizeof(float));
    float *K = (float *)fid_alloc((size_t)NR * kvd * sizeof(float));
    float *Vv = (float *)fid_alloc((size_t)NR * kvd * sizeof(float));
    uint16_t *Kb = (uint16_t *)fid_alloc((size_t)NR * kvd * 2);
    uint16_t *Vb = (uint16_t *)fid_alloc((size_t)NR * kvd * 2);
    float *att = (float *)fid_alloc((size_t)NR * qd * sizeof(float));
    const int64_t CH = 1024;
    float *g1 = (float *)fid_alloc((size_t)CH * FFN * sizeof(float));
    float *u1 = (float *)fid_alloc((size_t)CH * FFN * sizeof(float));
    float *tmp = (float *)fid_alloc((size_t)CH * H * sizeof(float));
    for (int l = 0; l < L; l++) {
        double tl = fid_now();
        char nm[256];
        #define LW(var, suffix, cnt) snprintf(nm, sizeof(nm), "encoder.layers.%d.%s", l, suffix); float *var = load_f(&m, nm, cnt)
        LW(nin, "input_layernorm.weight", H);
        LW(npost, "post_attention_layernorm.weight", H);
        LW(Wq, "self_attn.q_proj.weight", (uint64_t)qd * H);
        LW(Wk, "self_attn.k_proj.weight", (uint64_t)kvd * H);
        LW(Wv, "self_attn.v_proj.weight", (uint64_t)kvd * H);
        LW(Wo, "self_attn.o_proj.weight", (uint64_t)H * qd);
        LW(Wg, "mlp.gate_proj.weight", (uint64_t)FFN * H);
        LW(Wu, "mlp.up_proj.weight", (uint64_t)FFN * H);
        LW(Wd, "mlp.down_proj.weight", (uint64_t)H * FFN);
        double tload = fid_now() - tl;
        rmsnorm_rows(xn, h, nin, NR, H);
        if (hout) { HAcc ha; hacc_init(&ha, H); hacc_add(&ha, xn, NR); snprintf(nm, sizeof(nm), "L%02d.qkv", l); hacc_write(&ha, hout, nm); }
        fid_gemm_nt(Q, qd, xn, H, Wq, H, NR, H, qd);
        fid_gemm_nt(K, kvd, xn, H, Wk, H, NR, H, kvd);
        fid_gemm_nt(Vv, kvd, xn, H, Wv, H, NR, H, kvd);
        for (int64_t i = 0; i < NR; i++) {
            const float *cs = g_cos + (size_t)R[i].pos * HD, *sn = g_sin + (size_t)R[i].pos * HD;
            float *vecs[2] = { Q + i * qd, K + i * kvd };
            int dims[2] = { qd, kvd };
            for (int vv = 0; vv < 2; vv++)
                for (int off = 0; off < dims[vv]; off += HD) {
                    float *xh = vecs[vv] + off, hold[64];
                    for (int d = 0; d < HD / 2; d++) hold[d] = xh[d];
                    for (int d = 0; d < HD / 2; d++) {
                        float x1 = hold[d], x2 = xh[HD / 2 + d];
                        xh[d] = x1 * cs[d] - x2 * sn[d];
                        xh[HD / 2 + d] = x2 * cs[HD / 2 + d] + x1 * sn[HD / 2 + d];
                    }
                }
            double sc = 1.0 + 0.1 * log(1.0 + floor(R[i].pos / 16384.0));
            if (sc != 1.0) for (int d = 0; d < qd; d++) Q[i * qd + d] = (float)(Q[i * qd + d] * sc);
        }
        for (int64_t i = 0; i < NR * kvd; i++) { Kb[i] = bf16r(K[i]); Vb[i] = bf16r(Vv[i]); }
        AttC ac;
        ac.Q = Q; ac.K = K; ac.Vv = Vv; ac.Kb = Kb; ac.Vb = Vb; ac.Out = att;
        fid_par_for(NR * NH, att_item, &ac);
        if (hout) { HAcc ha; hacc_init(&ha, qd); hacc_add(&ha, att, NR); snprintf(nm, sizeof(nm), "L%02d.o", l); hacc_write(&ha, hout, nm); }
        for (int64_t r0 = 0; r0 < NR; r0 += CH) {
            int64_t n = NR - r0 < CH ? NR - r0 : CH;
            fid_gemm_nt(tmp, H, att + r0 * qd, qd, Wo, qd, n, qd, H);
            for (int64_t i = 0; i < n * H; i++) h[r0 * H + i] += tmp[i];
        }
        rmsnorm_rows(xn, h, npost, NR, H);
        HAcc hgu, hdn;
        if (hout) { hacc_init(&hgu, H); hacc_add(&hgu, xn, NR); snprintf(nm, sizeof(nm), "L%02d.gu", l); hacc_write(&hgu, hout, nm); hacc_init(&hdn, FFN); }
        for (int64_t r0 = 0; r0 < NR; r0 += CH) {
            int64_t n = NR - r0 < CH ? NR - r0 : CH;
            fid_gemm_nt(g1, FFN, xn + r0 * H, H, Wg, H, n, H, FFN);
            fid_gemm_nt(u1, FFN, xn + r0 * H, H, Wu, H, n, H, FFN);
            for (int64_t i = 0; i < n * FFN; i++) g1[i] = (g1[i] / (1.0f + expf(-g1[i]))) * u1[i];
            if (hout) hacc_add(&hdn, g1, n);
            fid_gemm_nt(tmp, H, g1, FFN, Wd, FFN, n, FFN, H);
            for (int64_t i = 0; i < n * H; i++) h[r0 * H + i] += tmp[i];
        }
        if (hout) { snprintf(nm, sizeof(nm), "L%02d.down", l); hacc_write(&hdn, hout, nm); }
        free(nin); free(npost); free(Wq); free(Wk); free(Wv); free(Wo); free(Wg); free(Wu); free(Wd);
        printf("layer %2d: load %.1f s, total %.1f s\n", l, tload, fid_now() - tl);
        fflush(stdout);
    }
    #undef LW
    free(Q); free(K); free(Vv); free(Kb); free(Vb); free(att); free(g1); free(u1); free(tmp);
    /* final norm for noisy rows, head, prior, scoring */
    float *hf = h + ncl * H;
    rmsnorm_rows(xn, hf, nfin, nno, H);
    if (hout) { HAcc ha; hacc_init(&ha, H); hacc_add(&ha, xn, nno); hacc_write(&ha, hout, "head"); }
    if (hout && !wref && !rref && !outp) {
        printf("calibration done, wall %.1f s\n", fid_now() - t0);
        return 0;
    }
    /* the head is held as bf16 (its storage type: exact) and widened a
     * tile at a time, so it costs 0.8 GB instead of 1.6 GB */
    uint16_t *Whb = (uint16_t *)fid_alloc((size_t)V * H * 2);
    {
        int s0 = 0;
        FidTensor *th = find_t(&m, "diffusion_head.weight", &s0);
        if (!th || th->count != (uint64_t)V * H) { fprintf(stderr, "bad diffusion_head.weight\n"); return 1; }
        float *tb = (float *)fid_alloc((size_t)4096 * H * sizeof(float));
        for (int v0 = 0; v0 < V; v0 += 4096) {
            int nv = V - v0 < 4096 ? V - v0 : 4096;
            if (fid_tensor_read(&m.sh[s0], th, (uint64_t)v0 * H, (uint64_t)nv * H, tb) != 0) {
                fprintf(stderr, "head read failed\n"); return 1; }
            for (int64_t i = 0; i < (int64_t)nv * H; i++) Whb[(int64_t)v0 * H + i] = fid_f32_to_bf16(tb[i]);
        }
        free(tb);
    }
    float *Wt = (float *)fid_alloc((size_t)4096 * H * sizeof(float));
    FILE *wf = wref ? fopen(wref, "wb") : NULL;
    FILE *rf = rref ? fopen(rref, "rb") : NULL;
    if ((wref && !wf) || (rref && !rf)) { fprintf(stderr, "cannot open reference file\n"); return 1; }
    FILE *tsv = outp ? fopen(outp, "w") : NULL;
    if (tsv) fprintf(tsv, "seq\tblock\tgamma\tpos\ttrue\tref_top1\tq_top1\ttop5_overlap\tkl\tce_ref\tce_q\n");
    St st[16];
    memset(st, 0, sizeof(st));
    const int64_t HC = 512;
    float *lg = (float *)fid_alloc((size_t)HC * V * sizeof(float));
    float *rl = (float *)fid_alloc((size_t)V * sizeof(float));
    uint64_t digest = FID_FNV_SEED;
    for (int64_t r0 = 0; r0 < nno; r0 += HC) {
        int64_t n = nno - r0 < HC ? nno - r0 : HC;
        for (int v0 = 0; v0 < V; v0 += 4096) {
            int nv = V - v0 < 4096 ? V - v0 : 4096;
            for (int64_t i = 0; i < (int64_t)nv * H; i++) Wt[i] = fid_bf16_to_f32(Whb[(int64_t)v0 * H + i]);
            fid_gemm_nt(lg + v0, V, xn + r0 * H, H, Wt, H, n, H, nv);
        }
        for (int64_t i = 0; i < n; i++) {
            const Row *r = &R[ncl + r0 + i];
            double g = gam[r->gi];
            double alpha = sqrt(1.0 / (1.0 + exp(g))), ssq = 1.0 / (1.0 + exp(-g));
            double prior = alpha / ssq;
            const double *z = zall + (r0 + i) * ED;
            float zf[64];
            for (int d = 0; d < ED; d++) zf[d] = (float)z[d];
            float *l1 = lg + i * V;
            for (int v = 0; v < V; v++) {
                double acc = 0;
                for (int d = 0; d < ED; d++) acc += (double)zf[d] * Enorm[(size_t)v * ED + d];
                l1[v] += (float)(prior * (float)acc);
            }
        }
        digest = fid_fnv(lg, (size_t)n * V * sizeof(float), digest);
        for (int64_t i = 0; i < n; i++) {
            const Row *r = &R[ncl + r0 + i];
            const float *lq = lg + i * V;
            if (wf) { if (fwrite(lq, sizeof(float), V, wf) != (size_t)V) { fprintf(stderr, "short write\n"); return 1; } memcpy(rl, lq, sizeof(float) * V); }
            else if (rf) { if (fread(rl, sizeof(float), V, rf) != (size_t)V) { fprintf(stderr, "reference too short\n"); return 1; } }
            else memcpy(rl, lq, sizeof(float) * V);
            double mq = -1e300, mr = -1e300;
            int aq = 0, ar = 0;
            for (int v = 0; v < V; v++) {
                if (lq[v] > mq) { mq = lq[v]; aq = v; }
                if (rl[v] > mr) { mr = rl[v]; ar = v; }
            }
            double sq = 0, sr = 0;
            for (int v = 0; v < V; v++) { sq += exp((double)lq[v] - mq); sr += exp((double)rl[v] - mr); }
            double lseq = mq + log(sq), lser = mr + log(sr), kl = 0;
            for (int v = 0; v < V; v++) {
                double lr = (double)rl[v] - lser, p = exp(lr);
                if (p > 0) kl += p * (lr - ((double)lq[v] - lseq));
            }
            if (kl < 0) kl = 0;
            int t5q[5], t5r[5];
            for (int k = 0; k < 5; k++) { t5q[k] = -1; t5r[k] = -1; }
            for (int v = 0; v < V; v++) {
                for (int k = 0; k < 5; k++)
                    if (t5q[k] < 0 || lq[v] > lq[t5q[k]]) { for (int mm = 4; mm > k; mm--) t5q[mm] = t5q[mm - 1]; t5q[k] = v; break; }
                for (int k = 0; k < 5; k++)
                    if (t5r[k] < 0 || rl[v] > rl[t5r[k]]) { for (int mm = 4; mm > k; mm--) t5r[mm] = t5r[mm - 1]; t5r[k] = v; break; }
            }
            int ov = 0;
            for (int a = 0; a < 5; a++) for (int c = 0; c < 5; c++) if (t5q[a] == t5r[c]) ov++;
            double ceq = lseq - lq[r->id], cer = lser - rl[r->id];
            St *S = &st[r->gi];
            S->n += 1; S->top1 += (aq == ar); S->top5 += ov / 5.0; S->kl += kl;
            if (kl > S->klmax) S->klmax = kl;
            S->ce_ref += cer; S->ce_q += ceq; S->acc_ref += (ar == r->id); S->acc_q += (aq == r->id);
            if (tsv) fprintf(tsv, "%d\t%d\t%g\t%d\t%d\t%d\t%d\t%d\t%.6f\t%.6f\t%.6f\n",
                             r->seq, r->blk, gam[r->gi], r->pos, r->id, ar, aq, ov, kl, cer, ceq);
        }
    }
    printf("logits digest %016llx\n", (unsigned long long)digest);
    double all_n = 0, all_t1 = 0, all_kl = 0;
    for (int gi = 0; gi < ng; gi++) {
        St *S = &st[gi];
        double n = S->n > 0 ? S->n : 1;
        char line[600];
        snprintf(line, sizeof(line),
                 "summary gamma %g: positions %.0f top1 %.4f top5 %.4f kl_mean %.6f kl_max %.6f ce_ref %.6f ce_q %.6f ppl_ref %.3f ppl_q %.3f acc_ref %.4f acc_q %.4f",
                 gam[gi], S->n, S->top1 / n, S->top5 / n, S->kl / n, S->klmax, S->ce_ref / n, S->ce_q / n,
                 exp(S->ce_ref / n), exp(S->ce_q / n), S->acc_ref / n, S->acc_q / n);
        printf("%s\n", line);
        if (tsv) fprintf(tsv, "# %s\n", line);
        all_n += S->n; all_t1 += S->top1; all_kl += S->kl;
    }
    printf("summary all: positions %.0f top1 %.4f kl_mean %.6f\n", all_n, all_t1 / all_n, all_kl / all_n);
    if (tsv) { fprintf(tsv, "# logits digest %016llx\n", (unsigned long long)digest); fclose(tsv); }
    if (wf) fclose(wf);
    if (rf) fclose(rf);
    printf("wall %.1f s\n", fid_now() - t0);
    return 0;
}
