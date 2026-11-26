/* test_accel_real.c - accel lanes on REAL packed tensors.
 * Co-implemented by Cortana. The pack reader and the safetensors
 * walk are Elfie's hardened code from tools/nq2-dc.c, included here
 * (its main renamed) rather than copied, so the geometry checks are
 * the shipped ones.
 *
 *   test_accel_real <v1.nq2> <v1.nq2book> <shard> [shard...]
 *                   [--match q_proj,k_proj,...] [--lane L]
 *
 * For each matched 2-D tensor (first match per pattern): validate,
 * upload, and run every operator on the chosen lane against the CPU
 * reference and the fp64 oracle; three repeats must be bit-identical.
 */
#define main nq2dc_probe_main
#include "../../tools/nq2-dc.c"
#undef main

#include "../include/ntse_accel.h"

static int checks = 0, fails = 0;
static void expect(const char *n, int ok)
{
    checks++;
    if (!ok) fails++;
    printf("%s %s\n", ok ? "ok  " : "FAIL", n);
    fflush(stdout);
}

static uint8_t *slurp(const char *p, uint64_t *n)
{
    FILE *f = fopen(p, "rb");
    if (!f) return NULL;
    fseeko(f, 0, SEEK_END);
    off_t s = ftello(f);
    fseeko(f, 0, SEEK_SET);
    if (s <= 0) { fclose(f); return NULL; }
    uint8_t *b = (uint8_t *)malloc((size_t)s);
    if (!b || fread(b, 1, (size_t)s, f) != (size_t)s) { free(b); fclose(f); return NULL; }
    fclose(f);
    *n = (uint64_t)s;
    return b;
}

int main(int argc, char **argv)
{
    const char *match = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj";
    ntse_lane lane = NTSE_LANE_AUTO;
    const char *pos[24];
    int npos = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--match") && i + 1 < argc) match = argv[++i];
        else if (!strcmp(argv[i], "--lane") && i + 1 < argc) {
            i++;
            if (!strcmp(argv[i], "cuda")) lane = NTSE_LANE_CUDA;
            else if (!strcmp(argv[i], "igpu")) lane = NTSE_LANE_CL_IGPU;
            else if (!strcmp(argv[i], "gpu-cl")) lane = NTSE_LANE_CL_GPU;
        } else if (npos < 24) pos[npos++] = argv[i];
    }
    if (npos < 3) {
        fprintf(stderr, "usage: test_accel_real v1.nq2 v1.nq2book shard... [--match a,b] [--lane L]\n");
        return 1;
    }
    uint64_t ssz = 0, bsz = 0;
    uint8_t *stream = slurp(pos[0], &ssz), *bookf = slurp(pos[1], &bsz);
    if (!stream || !bookf || ssz < 32 || memcmp(stream, "NQ2\0", 4) != 0
        || rd32(stream + 4) != 1) {
        fprintf(stderr, "need a readable v1 pack\n");
        return 1;
    }
    uint32_t block = rd32(stream + 8), entries = rd32(stream + 12);
    uint64_t n_blocks = rd64(stream + 24);
    if (block == 0 || block > 4096 || entries == 0 || entries > 256
        || n_blocks > (UINT64_MAX - 32) / 5 || ssz != 32 + 5 * n_blocks
        || bsz != 12 + (uint64_t)entries * block * 4 || memcmp(bookf, "NQ2B", 4) != 0
        || rd32(bookf + 4) != block || rd32(bookf + 8) != entries) {
        fprintf(stderr, "pack/book geometry mismatch: refused\n");
        return 1;
    }
    float *book = (float *)malloc((size_t)entries * block * 4);
    memcpy(book, bookf + 12, (size_t)entries * block * 4);
    const uint8_t *idx_sec = stream + 32, *mn_sec = stream + 32 + n_blocks,
                  *sc_sec = stream + 32 + 3 * n_blocks;

    ntse_accel_config cfg;
    ntse_accel_config_default(&cfg);
    cfg.lane = lane;
    cfg.verbose = 1;
    ntse_accel *a = NULL;
    ntse_accel_open(&a, &cfg);
    printf("== real-tensor lane %s: %s\n", ntse_lane_name(ntse_accel_lane(a)), ntse_accel_device(a));

    char pats[512];
    snprintf(pats, sizeof pats, "%s", match);
    char *save = NULL;
    for (char *pat = strtok_r(pats, ",", &save); pat; pat = strtok_r(NULL, ",", &save)) {
        uint64_t cursor = 0, b0 = 0;
        TensorInfo tgt;
        int found = 0;
        memset(&tgt, 0, sizeof tgt);
        for (int s = 0; s < npos - 2; s++) {
            Shard sh;
            if (shard_open(&sh, pos[2 + s]) != 0) { fprintf(stderr, "shard refused\n"); return 1; }
            for (int ti = 0; ti < sh.n_tensors; ti++) {
                TensorInfo *t = &sh.tensors[ti];
                if (!found && strstr(t->name, pat) && t->ndims == 2
                    && t->dims[1] % block == 0 && t->count % block == 0) {
                    tgt = *t;
                    tgt.name = strdup(t->name);
                    found = 1;
                    b0 = cursor;
                }
                cursor += t->count / block;
            }
            fclose(sh.file);
            shard_free_meta(&sh);
        }
        if (cursor != n_blocks) {
            expect("tensor walk reconciles with the header", 0);
            break;
        }
        if (!found) { printf("SKIP %s: no 2-D tensor matches\n", pat); continue; }
        uint64_t M = tgt.dims[0], N = tgt.dims[1], nb = tgt.count / block;
        if (b0 + nb > n_blocks) { expect("target inside the stream", 0); continue; }
        uint16_t *mn = (uint16_t *)malloc(nb * 2), *sc = (uint16_t *)malloc(nb * 2);
        for (uint64_t b = 0; b < nb; b++) {
            mn[b] = rd16(mn_sec + 2 * (b0 + b));
            sc[b] = rd16(sc_sec + 2 * (b0 + b));
        }
        ntse_dtc_tensor t = { M, N, block, entries, book, idx_sec + b0, mn, sc };
        {
            /* weight-space fingerprint (as nq2-dc): a walk misaligned by
             * even one block reads near sqrt(2); aligned, it reproduces
             * the packer's own per-tensor distortion */
            Shard fs;
            int fsi = -1;
            for (int s = 0; s < npos - 2 && fsi < 0; s++) {
                if (shard_open(&fs, pos[2 + s]) != 0) break;
                for (int ti = 0; ti < fs.n_tensors; ti++)
                    if (!strcmp(fs.tensors[ti].name, tgt.name)) { fsi = ti; break; }
                if (fsi < 0) { fclose(fs.file); shard_free_meta(&fs); }
            }
            /* aligned figure, plus a control with the walk deliberately
             * shifted one block: a correct walk must beat its own shift
             * and correlate positively with the raw tensor (relL2 alone
             * cannot tell a misaligned walk from a low-norm reconstruction) */
            double fp = -1, fps = -1, cosw = 0;
            float *raw = fsi >= 0 ? (float *)malloc(tgt.count * 4) : NULL;
            if (raw && tensor_read(&fs, &fs.tensors[fsi], raw) == 0 && nb > 1) {
                double d2 = 0, r2 = 0, w2 = 0, wr = 0, s2d = 0, s2r = 0;
                for (uint64_t b = 0; b < nb; b++) {
                    float m2 = half_to_float(mn[b]), sc2 = half_to_float(sc[b]);
                    const float *pr = book + (size_t)idx_sec[b0 + b] * block;
                    const float *rs = raw + ((b + 1) % nb) * block;
                    for (uint32_t i = 0; i < block; i++) {
                        double w = (double)m2 + (double)sc2 * pr[i], rv = raw[b * block + i];
                        d2 += (w - rv) * (w - rv); r2 += rv * rv;
                        w2 += w * w; wr += w * rv;
                        s2d += (w - rs[i]) * (w - rs[i]); s2r += (double)rs[i] * rs[i];
                    }
                }
                fp = r2 > 0 ? sqrt(d2 / r2) : -1;
                fps = s2r > 0 ? sqrt(s2d / s2r) : -1;
                cosw = (w2 > 0 && r2 > 0) ? wr / sqrt(w2 * r2) : 0;
            }
            free(raw);
            if (fsi >= 0) { fclose(fs.file); shard_free_meta(&fs); }
            printf("     %s weight-space fingerprint: relL2 %.4f aligned vs %.4f "
                   "shifted-one-block control | cos(recon,raw) %.4f\n",
                   tgt.name, fp, fps, cosw);
            expect("walk aligned (beats one-block shift, cos > 0.05)",
                   fp >= 0 && fps > fp && cosw > 0.05);
        }
        char why[200], name[300];
        int vr = ntse_dtc_validate(&t, why, sizeof why);
        snprintf(name, sizeof name, "%s [%llu x %llu] validates", tgt.name,
                 (unsigned long long)M, (unsigned long long)N);
        expect(name, vr == NTSE_OK);
        if (vr != NTSE_OK) { printf("     %s\n", why); free(mn); free(sc); continue; }
        float *x = (float *)malloc(N * 4), *ref = (float *)malloc(M * 4),
              *got = (float *)malloc(M * 4), *rep = (float *)malloc(M * 4);
        double *orc = (double *)malloc(M * 8);
        rng_state = 7;
        for (uint64_t i = 0; i < N; i++) x[i] = (float)randn();
        ntse_cpu_oracle(&t, x, 1, orc);
        ntse_dtensor *d = NULL;
        expect("upload", ntse_accel_upload(a, &t, &d) == NTSE_OK);
        for (int op = 0; op <= NTSE_OP_DTC_LAZY; op++) {
            ntse_lane used;
            ntse_cpu_matvec(&t, (ntse_op)op, x, 1, ref);
            int rc = ntse_accel_matvec(a, d, (ntse_op)op, x, 1, got, &used);
            double d2 = 0, r2 = 0, mx = 0, o2 = 0, oo = 0;
            uint64_t differ = 0;
            for (uint64_t r = 0; r < M; r++) {
                double e = (double)got[r] - ref[r];
                d2 += e * e; r2 += (double)ref[r] * ref[r];
                if (fabs(e) > mx) mx = fabs(e);
                if (memcmp(&got[r], &ref[r], 4)) differ++;
                double eo = (double)got[r] - orc[r];
                o2 += eo * eo; oo += orc[r] * orc[r];
            }
            int det = 1;
            for (int k = 0; k < 3 && det; k++)
                if (ntse_accel_matvec(a, d, (ntse_op)op, x, 1, rep, NULL) != NTSE_OK
                    || memcmp(rep, got, M * 4)) det = 0;
            double rl2 = sqrt(d2 / r2);
            double tol = op == NTSE_OP_EXPANDED ? 1e-5 : 1e-6;
            snprintf(name, sizeof name, "%s %s on %s", tgt.name, ntse_op_name((ntse_op)op),
                     ntse_lane_name(used));
            expect(name, rc == NTSE_OK && rl2 <= tol && det);
            printf("     vs CPU relL2 %.3e max_abs %.3e differ %llu/%llu%s | vs fp64 "
                   "oracle relL2 %.3e | deterministic %s\n", rl2, mx,
                   (unsigned long long)differ, (unsigned long long)M,
                   differ == 0 ? " BITEXACT" : "", sqrt(o2 / oo), det ? "yes" : "NO");
        }
        ntse_accel_release(a, d);
        free(x); free(ref); free(got); free(rep); free(orc); free(mn); free(sc);
        free(tgt.name);
    }
    ntse_accel_close(a);
    free(book); free(stream); free(bookf);
    printf("RESULT: %d/%d checks passed\n", checks - fails, checks);
    return fails ? 1 : 0;
}
