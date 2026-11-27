/* fid-quant.c ... fidelity lab quantizer: methods 1 to 4 of the
 * fidelity roadmap on a Sigma-family checkpoint.
 *
 * Co-implemented by Cortana at JD's direction, 2026-11-27. Pure C99,
 * zero mutexes, bit-identical results for any --threads.
 *
 * Reads the raw safetensors shards, quantizes the scope tensors (the
 * 182 layer projections and the diffusion head) with the framed PQ
 * family in fid-pq.h, and writes a variant model directory: the same
 * shard files with every quantized tensor replaced by its
 * reconstruction (bf16, exactly what a decoder would produce), so
 * the unmodified engine can be judged on it against the raw model.
 * Everything outside the scope is copied raw, as in every other run.
 * A per-tensor report gives the rate (bits per weight, counting
 * indices, float16 frame scales and float16 books), the weight error
 * and the layer-output error tr(dW H dW^T) / tr(W H W^T).
 *
 * Usage:
 *   fid-quant --model DIR --out DIR --report FILE
 *             [--hess DIR] [--d 4] [--K 256] [--K2 0]
 *             [--fit weight|act] [--gptq] [--refit N] [--scale-refit]
 *             [--rounds N] [--damp 0.01] [--km-iters 10] [--sample 32768]
 *             [--plan FILE] [--only a,b] [--share-books] [--no-write]
 *             [--state DIR] [--from-state DIR] [--threads N] [--seed N]
 *
 * --plan FILE: lines "substring raw" or "substring d K [K2]"; the first
 *   line whose substring occurs in a tensor's name overrides the
 *   defaults for it (method 2: critical tensors raw, others squeezed).
 * --only a,b: quantize only scope tensors whose names contain one of
 *   the substrings; every other tensor stays raw (sensitivity sweeps).
 * --share-books (method 4): for each projection type, one book set is
 *   learned from all layers' tensors pooled (weighted by each layer's
 *   own diag(H) when --fit act), charged once, and every layer then
 *   quantizes against it; with --refit each layer refits a private
 *   copy (the per-layer delta), charged as a delta only for entries
 *   that changed in float16.
 * --state DIR: also write each quantized tensor's decoder state
 *   (books, scales, codes) as DIR/<tensor>.fidq.
 * --from-state DIR (method 5's output): no quantization; every scope
 *   tensor with a DIR/<tensor>.fidq is rebuilt from it, every other
 *   one stays raw. With --hess the report still gives the errors.
 */
#include "fid-pq.h"

#define MAXT 512

typedef struct {
    int shard;
    FidTensor *t;
    char group[64];   /* Hessian group: L07.qkv, head, ... */
    char type[32];    /* q_proj, ..., head */
    int layer;
} ScopeT;

typedef struct { char sub[200]; int raw; int d, K, K2; } PlanLine;

static PlanLine g_plan[256];
static int g_nplan = 0;

static int in_scope(const char *name, char *group, char *type, int *layer)
{
    int L;
    char kind[64];
    if (!strcmp(name, "diffusion_head.weight")) {
        strcpy(group, "head"); strcpy(type, "head"); *layer = -1; return 1;
    }
    if (sscanf(name, "encoder.layers.%d.%63s", &L, kind) != 2) return 0;
    const char *p = strchr(kind, '.');
    if (!p) return 0;
    char t[32];
    size_t n = strcspn(p + 1, ".");
    if (n >= sizeof(t)) return 0;
    memcpy(t, p + 1, n); t[n] = 0;
    if (strcmp(p + 1 + n, ".weight") != 0) return 0;
    const char *g = NULL;
    if (!strcmp(t, "q_proj") || !strcmp(t, "k_proj") || !strcmp(t, "v_proj")) g = "qkv";
    else if (!strcmp(t, "o_proj")) g = "o";
    else if (!strcmp(t, "gate_proj") || !strcmp(t, "up_proj")) g = "gu";
    else if (!strcmp(t, "down_proj")) g = "down";
    else return 0;
    snprintf(group, 64, "L%02d.%s", L, g);
    snprintf(type, 32, "%s", t);
    *layer = L;
    return 1;
}

static int list_match(const char *name, const char *csv)
{
    if (!csv) return 1;
    char buf[4096];
    snprintf(buf, sizeof(buf), "%s", csv);
    for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ","))
        if (*tok && strstr(name, tok)) return 1;
    return 0;
}

static float *load_hess(const char *dir, const char *group, int64_t dim, uint64_t *rows)
{
    char path[1200];
    snprintf(path, sizeof(path), "%s/%s.fidh", dir, group);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "missing Hessian %s\n", path); exit(1); }
    uint32_t hdr[2];
    double tr;
    if (fread(hdr, 4, 2, f) != 2 || hdr[0] != 0x48444946u || hdr[1] != (uint32_t)dim
            || fread(rows, 8, 1, f) != 1 || fread(&tr, 8, 1, f) != 1) {
        fprintf(stderr, "bad Hessian file %s (want dim %lld)\n", path, (long long)dim);
        exit(1);
    }
    float *H = (float *)fid_alloc((size_t)dim * dim * sizeof(float));
    if (fread(H, sizeof(float), (size_t)dim * dim, f) != (size_t)(dim * dim)) {
        fprintf(stderr, "short Hessian %s\n", path); exit(1);
    }
    fclose(f);
    return H;
}

/* U cache next to the Hessian, keyed by the damping */
static float *load_or_make_u(const char *dir, const char *group, const float *H,
                             int64_t dim, double damp)
{
    char path[1200];
    snprintf(path, sizeof(path), "%s/%s.d%g.fidu", dir, group, damp);
    float *U = (float *)fid_alloc((size_t)dim * dim * sizeof(float));
    FILE *f = fopen(path, "rb");
    if (f) {
        uint32_t hdr[2];
        double dmp;
        if (fread(hdr, 4, 2, f) == 2 && hdr[0] == 0x55444946u && hdr[1] == (uint32_t)dim
                && fread(&dmp, 8, 1, f) == 1 && dmp == damp
                && fread(U, sizeof(float), (size_t)dim * dim, f) == (size_t)(dim * dim)) {
            fclose(f);
            return U;
        }
        fclose(f);
    }
    double t0 = fid_now();
    if (fid_inv_upper_factor(H, dim, damp, U) != 0) {
        fprintf(stderr, "Hessian %s is not positive definite at damp %g\n", group, damp);
        exit(1);
    }
    f = fopen(path, "wb");
    if (f) {
        uint32_t hdr[2] = { 0x55444946u, (uint32_t)dim };
        fwrite(hdr, 4, 2, f); fwrite(&damp, 8, 1, f);
        fwrite(U, sizeof(float), (size_t)dim * dim, f);
        fclose(f);
    }
    printf("  factor %s (%lld) %.1f s\n", group, (long long)dim, fid_now() - t0);
    return U;
}

static int copy_file(const char *src, const char *dst)
{
    FILE *a = fopen(src, "rb"), *b = fopen(dst, "wb");
    if (!a || !b) { if (a) fclose(a); if (b) fclose(b); return -1; }
    const size_t CH = 64u << 20;
    char *buf = (char *)fid_alloc(CH);
    size_t n;
    while ((n = fread(buf, 1, CH, a)) > 0)
        if (fwrite(buf, 1, n, b) != n) { free(buf); fclose(a); fclose(b); return -1; }
    free(buf);
    fclose(a);
    return fclose(b) == 0 ? 0 : -1;
}

static int write_tensor(const char *path, const FidTensor *t, const float *Q)
{
    FILE *f = fopen(path, "r+b");
    if (!f) return -1;
    if (fid_seek(f, t->off) != 0) { fclose(f); return -1; }
    const uint64_t CH = 1u << 22;
    int esz = t->dtype == FID_F32 ? 4 : 2;
    uint8_t *buf = (uint8_t *)fid_alloc(CH * 4);
    for (uint64_t d = 0; d < t->count; d += CH) {
        uint64_t m = t->count - d < CH ? t->count - d : CH;
        for (uint64_t i = 0; i < m; i++) {
            float v = Q[d + i];
            if (t->dtype == FID_BF16) ((uint16_t *)buf)[i] = fid_f32_to_bf16(v);
            else if (t->dtype == FID_F16) ((uint16_t *)buf)[i] = fid_f32_to_f16(v);
            else ((float *)buf)[i] = v;
        }
        if (fwrite(buf, esz, m, f) != m) { free(buf); fclose(f); return -1; }
    }
    free(buf);
    return fclose(f) == 0 ? 0 : -1;
}

static int cmp_scope(const void *a, const void *b)
{
    const ScopeT *x = (const ScopeT *)a, *y = (const ScopeT *)b;
    int c = strcmp(x->group, y->group);
    return c ? c : strcmp(x->t->name, y->t->name);
}

int main(int argc, char **argv)
{
    const char *model = NULL, *out = NULL, *report = NULL, *hess = NULL;
    const char *plan = NULL, *only = NULL;
    PqCfg cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.d = 4; cfg.K = 256; cfg.K2 = 0; cfg.km_iters = 10; cfg.sample = 32768;
    cfg.damp = 0.01; cfg.rounds = 1; cfg.seed = 20261127;
    int threads = 0, no_write = 0, share = 0;
    const char *state_out = NULL, *state_in = NULL;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        int more = i + 1 < argc;
        if (!strcmp(a, "--model") && more) model = argv[++i];
        else if (!strcmp(a, "--out") && more) out = argv[++i];
        else if (!strcmp(a, "--report") && more) report = argv[++i];
        else if (!strcmp(a, "--hess") && more) hess = argv[++i];
        else if (!strcmp(a, "--d") && more) cfg.d = atoi(argv[++i]);
        else if (!strcmp(a, "--K") && more) cfg.K = atoi(argv[++i]);
        else if (!strcmp(a, "--K2") && more) cfg.K2 = atoi(argv[++i]);
        else if (!strcmp(a, "--fit") && more) cfg.fit_act = !strcmp(argv[++i], "act");
        else if (!strcmp(a, "--gptq")) cfg.gptq = 1;
        else if (!strcmp(a, "--refit") && more) cfg.refit = atoi(argv[++i]);
        else if (!strcmp(a, "--scale-refit")) cfg.scale_refit = 1;
        else if (!strcmp(a, "--rounds") && more) cfg.rounds = atoi(argv[++i]);
        else if (!strcmp(a, "--damp") && more) cfg.damp = atof(argv[++i]);
        else if (!strcmp(a, "--km-iters") && more) cfg.km_iters = atoi(argv[++i]);
        else if (!strcmp(a, "--sample") && more) cfg.sample = atoll(argv[++i]);
        else if (!strcmp(a, "--seed") && more) cfg.seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(a, "--plan") && more) plan = argv[++i];
        else if (!strcmp(a, "--only") && more) only = argv[++i];
        else if (!strcmp(a, "--share-books")) share = 1;
        else if (!strcmp(a, "--no-write")) no_write = 1;
        else if (!strcmp(a, "--state") && more) state_out = argv[++i];
        else if (!strcmp(a, "--from-state") && more) state_in = argv[++i];
        else if (!strcmp(a, "--threads") && more) threads = atoi(argv[++i]);
        else { fprintf(stderr, "unknown or incomplete option %s\n", a); return 1; }
    }
    if (!model || !report || (!out && !no_write)) {
        fprintf(stderr, "usage: fid-quant --model DIR --out DIR --report FILE [options]\n");
        return 1;
    }
    int needs_h = cfg.fit_act || cfg.gptq || cfg.refit || cfg.scale_refit;
    if (state_in) needs_h = 0;
    if (state_in && (share || plan)) { fprintf(stderr, "--from-state takes no --share-books or --plan\n"); return 1; }
    if (needs_h && !hess) { fprintf(stderr, "this method needs --hess DIR\n"); return 1; }
    g_fid_threads = threads > 0 ? threads : fid_cpu_count();
    if (plan) {
        FILE *pf = fopen(plan, "r");
        if (!pf) { fprintf(stderr, "cannot open plan %s\n", plan); return 1; }
        char line[512];
        while (fgets(line, sizeof(line), pf) && g_nplan < 256) {
            if (line[0] == '#' || line[0] == '\n') continue;
            PlanLine *p = &g_plan[g_nplan];
            memset(p, 0, sizeof(*p));
            char w2[32];
            int n = sscanf(line, "%199s %31s %d %d", p->sub, w2, &p->K, &p->K2);
            if (n < 2) continue;
            if (!strcmp(w2, "raw")) p->raw = 1;
            else { p->d = atoi(w2); if (n < 3) p->K = cfg.K; }
            g_nplan++;
        }
        fclose(pf);
    }
    /* shards, sorted */
    char names[16][512];
    int ns = 0;
    {
        char cmdpath[1100];
        (void)cmdpath;
        for (int s = 1; s <= 16 && ns < 16; s++) {
            /* Sigma shards follow the Hugging Face pattern; probe 1..16 of N */
            for (int n = 1; n <= 16; n++) {
                char p[2200];
                snprintf(p, sizeof(p), "%s/model-%05d-of-%05d.safetensors", model, s, n);
                FILE *f = fopen(p, "rb");
                if (f) { fclose(f); snprintf(names[ns++], 512, "model-%05d-of-%05d.safetensors", s, n); break; }
            }
        }
        if (ns == 0) {
            char p[2200];
            snprintf(p, sizeof(p), "%s/model.safetensors", model);
            FILE *f = fopen(p, "rb");
            if (f) { fclose(f); snprintf(names[ns++], 512, "model.safetensors"); }
        }
    }
    if (ns == 0) { fprintf(stderr, "no safetensors shards in %s\n", model); return 1; }
    FidShard *sh = (FidShard *)fid_calloc(ns, sizeof(FidShard));
    ScopeT *sc = (ScopeT *)fid_calloc(MAXT, sizeof(ScopeT));
    int nsc = 0;
    uint64_t all_w = 0;
    for (int s = 0; s < ns; s++) {
        char p[2200];
        fid_join(p, sizeof(p), model, names[s]);
        if (fid_shard_open(&sh[s], p) != 0) { fprintf(stderr, "cannot open %s\n", p); return 1; }
        for (int i = 0; i < sh[s].n; i++) {
            FidTensor *t = &sh[s].t[i];
            all_w += t->count;
            ScopeT x;
            memset(&x, 0, sizeof(x));
            if (nsc < MAXT && in_scope(t->name, x.group, x.type, &x.layer)) {
                x.shard = s; x.t = t;
                sc[nsc++] = x;
            }
        }
    }
    qsort(sc, nsc, sizeof(ScopeT), cmp_scope);
    if (!no_write) {
        for (int s = 0; s < ns; s++) {
            char a[2200], b[2200];
            fid_join(a, sizeof(a), model, names[s]);
            fid_join(b, sizeof(b), out, names[s]);
            double t0 = fid_now();
            if (copy_file(a, b) != 0) { fprintf(stderr, "cannot copy %s to %s\n", a, b); return 1; }
            printf("copied %s (%.1f s)\n", names[s], fid_now() - t0);
        }
    }
    FILE *rf = fopen(report, "w");
    if (!rf) { fprintf(stderr, "cannot open %s\n", report); return 1; }
    fprintf(rf, "# fid-quant d=%d K=%d K2=%d fit=%s gptq=%d refit=%d scale_refit=%d rounds=%d damp=%g km_iters=%d sample=%lld share=%d plan=%s only=%s from_state=%s seed=%llu threads=%d\n",
            cfg.d, cfg.K, cfg.K2, cfg.fit_act ? "act" : "weight", cfg.gptq, cfg.refit,
            cfg.scale_refit, cfg.rounds, cfg.damp, cfg.km_iters, (long long)cfg.sample,
            share, plan ? plan : "-", only ? only : "-", state_in ? state_in : "-", (unsigned long long)cfg.seed, g_fid_threads);
    fprintf(rf, "tensor\trows\tcols\tmode\td\tK\tK2\tbits\tbpw\twrel\torel\tstages\tsecs\tdigest\n");
    /* method 4: shared books per projection type */
    typedef struct { char type[32]; float *b1, *b2; int d, K, K2; int64_t c; } Shared;
    Shared shared[8];
    int nshared = 0;
    double shared_bits = 0;
    char cur_group[64] = "";
    float *H = NULL, *U = NULL;
    int64_t Hdim = 0;
    double tot_bits = 0, scope_w = 0, raw_scope_w = 0, wsum_e = 0, wsum_w = 0;
    double t_all = fid_now();
    for (int pass = (share && !state_in) ? 0 : 1; pass < 2; pass++) {
        /* pass 0 (sharing only): learn pooled books per type */
        if (pass == 0) {
            const char *types[8] = { "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", NULL };
            for (int ti = 0; types[ti]; ti++) {
                int64_t c = 0, rtot = 0;
                int nl = 0;
                for (int i = 0; i < nsc; i++)
                    if (!strcmp(sc[i].type, types[ti]) && list_match(sc[i].t->name, only)) {
                        c = (int64_t)sc[i].t->dims[1]; rtot += (int64_t)sc[i].t->dims[0]; nl++;
                    }
                if (!nl) continue;
                /* pool: stack every layer's rows, each scaled to unit
                 * diag(H) weighting by folding sqrt(diag) into a
                 * per-layer weight matrix copy is not exact for the
                 * normalized subvectors, so the pooled k-means runs on
                 * the stacked weights with the mean diag(H). */
                float *Wp = (float *)fid_alloc((size_t)rtot * c * sizeof(float));
                float *Hm = cfg.fit_act ? (float *)fid_calloc((size_t)c * c, sizeof(float)) : NULL;
                int64_t ro = 0;
                for (int i = 0; i < nsc; i++) {
                    if (strcmp(sc[i].type, types[ti]) || !list_match(sc[i].t->name, only)) continue;
                    int64_t r = (int64_t)sc[i].t->dims[0];
                    if (fid_tensor_read(&sh[sc[i].shard], sc[i].t, 0, sc[i].t->count, Wp + ro * c) != 0) {
                        fprintf(stderr, "read failed %s\n", sc[i].t->name); return 1; }
                    ro += r;
                    if (Hm) {
                        uint64_t hr;
                        float *Hl = load_hess(hess, sc[i].group, c, &hr);
                        for (int64_t j = 0; j < c; j++) Hm[j * c + j] += Hl[j * c + j] / nl;
                        free(Hl);
                    }
                }
                PqCfg lc = cfg;
                lc.learn_only = 1;
                lc.sample = cfg.sample * 4;
                PqState st; PqStats ps;
                pq_quantize(Wp, rtot, c, Hm, NULL, &lc, NULL, &st, &ps);
                Shared *S = &shared[nshared++];
                snprintf(S->type, sizeof(S->type), "%s", types[ti]);
                S->b1 = st.b1; st.b1 = NULL; S->b2 = st.b2; st.b2 = NULL;
                S->d = cfg.d; S->K = cfg.K; S->K2 = cfg.K2; S->c = c;
                pq_state_free(&st);
                double bb = (double)(FID_BLOCK / cfg.d) * (cfg.K + cfg.K2) * cfg.d * 16.0;
                shared_bits += bb;
                printf("shared book %s: %d layers, %lld rows, %.0f bits, %.1f s\n", types[ti], nl,
                       (long long)rtot, bb, ps.secs);
                fprintf(rf, "# shared %s layers %d bits %.0f\n", types[ti], nl, bb);
                free(Wp); free(Hm);
            }
            continue;
        }
        for (int i = 0; i < nsc; i++) {
            FidTensor *t = sc[i].t;
            int64_t r = (int64_t)t->dims[0], c = (int64_t)t->dims[1];
            scope_w += (double)t->count;
            PqCfg tc = cfg;
            int raw = !list_match(t->name, only);
            for (int p = 0; p < g_nplan; p++)
                if (strstr(t->name, g_plan[p].sub)) {
                    if (g_plan[p].raw) raw = 1;
                    else { tc.d = g_plan[p].d; tc.K = g_plan[p].K; tc.K2 = g_plan[p].K2; }
                    break;
                }
            if (t->ndim != 2 || c % FID_BLOCK != 0 || FID_BLOCK % tc.d != 0 || tc.d > 16 || tc.K > 65536) raw = 1;
            PqState fst;
            int have_fst = 0;
            if (state_in) {
                char sp[2200];
                snprintf(sp, sizeof(sp), "%s/%s.fidq", state_in, t->name);
                int rc = pq_state_read(sp, &fst);
                if (rc < 0) { fprintf(stderr, "bad state file %s\n", sp); return 1; }
                if (rc == 0 && (fst.r != r || fst.c != c)) { fprintf(stderr, "state %s has the wrong shape\n", sp); return 1; }
                have_fst = rc == 0;
                raw = !have_fst;
                if (have_fst) { tc.d = fst.d; tc.K = fst.K; tc.K2 = fst.K2; }
            }
            if (raw) {
                tot_bits += 16.0 * t->count;
                raw_scope_w += (double)t->count;
                fprintf(rf, "%s\t%lld\t%lld\traw\t-\t-\t-\t%.0f\t16.0000\t0\t0\t-\t0\t-\n", t->name,
                        (long long)r, (long long)c, 16.0 * t->count);
                continue;
            }
            float *W = (float *)fid_alloc((size_t)t->count * sizeof(float));
            if (fid_tensor_read(&sh[sc[i].shard], t, 0, t->count, W) != 0) {
                fprintf(stderr, "read failed %s\n", t->name); return 1; }
            int use_h = hess != NULL;   /* fitting uses it only as the method says; metrics always */
            if (use_h && strcmp(cur_group, sc[i].group) != 0) {
                free(H); free(U); H = U = NULL;
                uint64_t hr;
                H = load_hess(hess, sc[i].group, c, &hr);
                Hdim = c;
                if (tc.gptq) U = load_or_make_u(hess, sc[i].group, H, c, tc.damp);
                snprintf(cur_group, sizeof(cur_group), "%s", sc[i].group);
            }
            if (use_h && Hdim != c) { fprintf(stderr, "Hessian dim mismatch for %s\n", t->name); return 1; }
            if (use_h && tc.gptq && !U) U = load_or_make_u(hess, sc[i].group, H, c, tc.damp);
            const Shared *S = NULL;
            if (share)
                for (int k = 0; k < nshared; k++) if (!strcmp(shared[k].type, sc[i].type)) S = &shared[k];
            if (S && S->d == tc.d && S->K == tc.K && S->K2 == tc.K2) {
                tc.ext_b1 = S->b1; tc.ext_b2 = S->b2;
            }
            float *Q = (float *)fid_alloc((size_t)t->count * sizeof(float));
            PqState st; PqStats ps;
            if (have_fst) {
                double tq = fid_now();
                st = fst;
                memset(&ps, 0, sizeof(ps));
                for (int k = 0; k < 8; k++) ps.orel_stage[k] = -1;
                pq_finish(W, use_h ? H : NULL, &st, Q, &ps);
                ps.bits = pq_bits(r, c, st.d, st.K, st.K2, 1);
                ps.secs = fid_now() - tq;
            } else {
                pq_quantize(W, r, c, use_h ? H : NULL, use_h ? U : NULL, &tc, Q, &st, &ps);
            }
            if (state_out) {
                char sp[2200];
                snprintf(sp, sizeof(sp), "%s/%s.fidq", state_out, t->name);
                if (pq_state_write(sp, &st) != 0) { fprintf(stderr, "cannot write %s\n", sp); return 1; }
            }
            double bits = ps.bits;
            if (S && tc.ext_b1 && tc.refit > 0) {
                /* per-layer delta: entries whose float16 value moved */
                int64_t n1 = st.nsub * (int64_t)st.K * st.d, n2 = st.nsub * (int64_t)st.K2 * st.d, ch = 0;
                for (int64_t k = 0; k < n1; k++) ch += st.b1[k] != S->b1[k];
                for (int64_t k = 0; k < n2; k++) ch += st.b2[k] != S->b2[k];
                /* changed entries: a 1-bit presence map over entries plus 16 bits per changed value */
                bits += (double)(st.nsub * (int64_t)(st.K + st.K2)) + 16.0 * ch;
            }
            tot_bits += bits;
            for (int64_t k = 0; k < r * c; k++) {
                double dd = (double)W[k] - Q[k];
                wsum_e += dd * dd; wsum_w += (double)W[k] * W[k];
            }
            char stg[256] = "";
            for (int k = 0; k < 8 && ps.orel_stage[k] >= 0; k++) {
                char b[32];
                snprintf(b, sizeof(b), "%s%.5f", k ? "," : "", ps.orel_stage[k]);
                strcat(stg, b);
            }
            uint64_t dg = fid_fnv(Q, (size_t)t->count * sizeof(float), FID_FNV_SEED);
            fprintf(rf, "%s\t%lld\t%lld\t%s\t%d\t%d\t%d\t%.0f\t%.4f\t%.5f\t%.5f\t%s\t%.1f\t%016llx\n",
                    t->name, (long long)r, (long long)c, have_fst ? "pq-state" : tc.ext_b1 ? "pq-shared" : "pq", tc.d, tc.K, tc.K2,
                    bits, bits / (double)t->count, ps.wrel, ps.orel, stg[0] ? stg : "-", ps.secs,
                    (unsigned long long)dg);
            fflush(rf);
            printf("%-48s %5lldx%-5lld bpw %.4f wrel %.4f orel %.5f [%s] %.1f s\n", t->name,
                   (long long)r, (long long)c, bits / (double)t->count, ps.wrel, ps.orel,
                   stg[0] ? stg : "-", ps.secs);
            fflush(stdout);
            if (!no_write) {
                char p[2200];
                fid_join(p, sizeof(p), out, names[sc[i].shard]);
                if (write_tensor(p, t, Q) != 0) { fprintf(stderr, "write failed %s\n", t->name); return 1; }
            }
            pq_state_free(&st);
            free(W); free(Q);
        }
    }
    tot_bits += shared_bits;
    double other_w = (double)all_w - scope_w;
    fprintf(rf, "# scope weights %.0f (raw inside scope %.0f), bits %.0f (shared books %.0f), scope bpw %.5f\n",
            scope_w, raw_scope_w, tot_bits, shared_bits, tot_bits / scope_w);
    fprintf(rf, "# weighted scope wrel %.5f\n", wsum_w > 0 ? sqrt(wsum_e / wsum_w) : 0.0);
    fprintf(rf, "# outside scope (raw bf16): %.0f weights; complete bpw with them at 16: %.5f\n",
            other_w, (tot_bits + 16.0 * other_w) / (double)all_w);
    fprintf(rf, "# wall %.1f s\n", fid_now() - t_all);
    fclose(rf);
    printf("scope bpw %.5f, weighted wrel %.5f, wall %.1f s\n", tot_bits / scope_w,
           wsum_w > 0 ? sqrt(wsum_e / wsum_w) : 0.0, fid_now() - t_all);
    free(H); free(U);
    return 0;
}
