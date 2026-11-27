/* parakeet-c unit tests (no weights needed; extra checks run if a model dir is given) */
#include "../src/pk.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdatomic.h>

static int fails, passes;
#define CHECK(c, ...) do { if (c) { passes++; printf("PASS "); } else { fails++; printf("FAIL "); } printf(__VA_ARGS__); printf("\n"); } while (0)
static uint32_t rs = 12345; static float rnd(void) { rs = rs * 1664525u + 1013904223u; return ((rs >> 8) & 0xffff) / 32768.f - 1.f; }

static void test_fft(void) {
    float in[512], re[257], im[257]; for (int i = 0; i < 512; i++) in[i] = rnd();
    pk_fft_real512(in, re, im);
    double md = 0;
    for (int k = 0; k < 257; k += 7) { double r = 0, q = 0; for (int n = 0; n < 512; n++) { r += in[n] * cos(2 * M_PI * k * n / 512); q -= in[n] * sin(2 * M_PI * k * n / 512); }
        double d = fabs(r - re[k]) + fabs(q - im[k]); if (d > md) md = d; }
    CHECK(md < 1e-3, "fft512 vs naive DFT, max err %.2e", md);
}
static void test_gemm(int threads) {
    int M = 37, N = 70, K = 1032;
    float *A = malloc(4 * M * K), *W = malloc(4 * N * K), *b = malloc(4 * N), *C = malloc(4 * M * N), *R = malloc(4 * M * N);
    uint16_t *Wb = malloc(2 * N * K);
    for (int i = 0; i < M * K; i++) A[i] = rnd();
    for (int i = 0; i < N * K; i++) { float w = rnd(); uint32_t u; memcpy(&u, &w, 4); Wb[i] = (uint16_t)(u >> 16); W[i] = pk_bf16_to_f32(Wb[i]); }
    for (int i = 0; i < N; i++) b[i] = rnd();
    pk_pool_start(threads);
    pk_gemm_ref(R, A, W, b, M, N, K);
    pk_gemm_f32w(C, A, W, b, M, N, K);
    double e1 = 0; for (int i = 0; i < M * N; i++) { double d = fabs(C[i] - R[i]); if (d > e1) e1 = d; }
    pk_gemm_bf16w(C, A, Wb, b, M, N, K);
    double e2 = 0; for (int i = 0; i < M * N; i++) { double d = fabs(C[i] - R[i]); if (d > e2) e2 = d; }
    pk_pool_stop();
    CHECK(e1 < 2e-3, "gemm f32 weights (%d threads) max err %.2e", threads, e1);
    CHECK(e2 < 2e-3, "gemm bf16 weights (%d threads) max err %.2e", threads, e2);
    free(A); free(W); free(b); free(C); free(R); free(Wb);
}
static _Atomic long g_sum; static void add_task(void *c, int i) { (void)c; atomic_fetch_add(&g_sum, i + 1); }
static void test_pool(void) {
    pk_pool_start(8); int ok = 1;
    for (int r = 0; r < 20000; r++) { atomic_store(&g_sum, 0); int n = 1 + r % 97; pk_parallel_for(n, add_task, NULL); if (atomic_load(&g_sum) != (long)n * (n + 1) / 2) { ok = 0; break; } }
    pk_pool_stop();
    CHECK(ok, "lock-free pool: 20000 fork/join jobs, sums exact");
}
static void test_resample(void) {
    int n = 48000; float *x = malloc(4 * n); for (int i = 0; i < n; i++) x[i] = 0.5f * sinf(2 * M_PI * 1000 * i / 48000.0);
    int no; float *y = pk_resample(x, n, 48000, 16000, &no);
    double e = 0, c = 0, s = 0; for (int i = 1000; i < no - 1000; i++) { e += y[i] * y[i]; c += y[i] * cos(2 * M_PI * 1000 * i / 16000.0); s += y[i] * sin(2 * M_PI * 1000 * i / 16000.0); }
    int m = no - 2000; double amp = sqrt(2 * e / m), pure = 2 * sqrt(c * c + s * s) / m;
    CHECK(no == 16000 && fabs(amp - 0.5) < 0.01 && fabs(pure - 0.5) < 0.01, "resample 48k->16k: %d samples, amp %.4f, 1 kHz component %.4f", no, amp, pure);
    /* aliasing: 10 kHz tone at 48k must vanish at 16k */
    for (int i = 0; i < n; i++) x[i] = 0.5f * sinf(2 * M_PI * 10000 * i / 48000.0);
    free(y); y = pk_resample(x, n, 48000, 16000, &no); e = 0; for (int i = 1000; i < no - 1000; i++) e += y[i] * y[i];
    double db = 10 * log10(e / (no - 2000) / 0.125);
    CHECK(db < -40, "resample anti-alias: 10 kHz tone attenuated %.1f dB", db);
    free(x); free(y);
}
static void test_wav(void) {
    float x[1600]; for (int i = 0; i < 1600; i++) x[i] = 0.25f * sinf(i * 0.1f);
    pk_wav_write16("pk_test.wav", x, 1600); pk_audio a; int rc = pk_wav_read("pk_test.wav", &a);
    double e = 0; if (!rc) for (int i = 0; i < 1600 && i < a.n; i++) { double d = fabs(a.x[i] - x[i]); if (d > e) e = d; }
    CHECK(!rc && a.n == 1600 && e < 1e-4, "wav 16-bit write/read round trip (n=%d err %.1e)", rc ? -1 : a.n, e);
    remove("pk_test.wav");
}
static void test_mel_synthetic(void) {
    pk_mel m; pk_mel_default_fb(&m, 128);
    int n = 16000; float *x = malloc(4 * n); for (int i = 0; i < n; i++) x[i] = 0.3f * sinf(2 * M_PI * 440 * i / 16000.0) + 0.01f * rnd();
    float *f; int T = pk_mel_compute(&m, x, n, &f);
    double mu = 0, var = 0; for (int t = 0; t < T; t++) mu += f[t * 128 + 20]; mu /= T; for (int t = 0; t < T; t++) var += (f[t * 128 + 20] - mu) * (f[t * 128 + 20] - mu); var /= T - 1;
    CHECK(T == 100 && fabs(mu) < 1e-4 && fabs(var - 1) < 1e-2, "mel 128: T=%d per-feature mean %.1e var %.4f", T, mu, var);
    /* fb row sums: slaney norm -> each filter integrates to ~2/(f_hi-f_lo)*area */
    int empty = 0; for (int j = 0; j < 128; j++) { double r = 0; for (int k = 0; k < 257; k++) r += m.fb[j * 257 + k]; if (r <= 0) empty++; }
    CHECK(empty == 0, "slaney mel filterbank: all 128 filters non-empty (%d empty)", empty);
    free(x); pk_free(f);
}
static void test_spm_synth(void) {
    char *p[] = { "<unk>", "\xe2\x96\x81Hello", ",", "\xe2\x96\x81J", "D", ".", "\xe2\x96\x81" };
    int ids[] = { 1, 2, 3, 4, 5 };
    char *s = pk_spm_detok(p, 7, ids, 5);
    CHECK(!strcmp(s, "Hello, JD."), "spm detok synthetic -> '%s'", s); free(s);
}


static int write_st(const char *path, const char *json, const void *data, size_t dn) {
    uint64_t hl = (uint64_t)strlen(json);
    FILE *f = fopen(path, "wb"); if (!f) return -1;
    uint8_t b[8]; for (int i = 0; i < 8; i++) b[i] = (uint8_t)(hl >> (8 * i));
    int bad = fwrite(b, 1, 8, f) != 8 || fwrite(json, 1, (size_t)hl, f) != (size_t)hl
        || (dn && fwrite(data, 1, dn, f) != dn);
    if (fclose(f)) bad = 1;
    return bad ? -1 : 0;
}
static void test_st_bounds(void) {
    const char *okj = "{\"w\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
    float one = 1.0f;
    pk_st st;
    if (write_st("pk_st_ok.st", okj, &one, 4)) { CHECK(0, "write valid safetensors"); return; }
    int rc = pk_st_open(&st, "pk_st_ok.st");
    CHECK(rc == 0 && st.n == 1 && st.t[0].nbytes == 4, "pk_st_open accepts a valid one-tensor file (rc=%d n=%d nbytes=%zu)", rc, rc ? -1 : st.n, rc ? 0 : st.t[0].nbytes);
    pk_st_close(&st);
    remove("pk_st_ok.st");

    const char *rev = "{\"w\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,0]}}";
    if (write_st("pk_st_rev.st", rev, &one, 4) == 0) {
        rc = pk_st_open(&st, "pk_st_rev.st");
        CHECK(rc == -7, "pk_st_open refuses off0 > off1 (rc=%d)", rc);
        pk_st_close(&st);
        remove("pk_st_rev.st");
    }
    const char *past = "{\"w\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,64]}}";
    if (write_st("pk_st_past.st", past, &one, 4) == 0) {
        rc = pk_st_open(&st, "pk_st_past.st");
        CHECK(rc == -7, "pk_st_open refuses off1 past the file (rc=%d)", rc);
        pk_st_close(&st);
        remove("pk_st_past.st");
    }
    {
        FILE *f = fopen("pk_st_wrap.st", "wb");
        uint64_t hl = ~(uint64_t)0; uint8_t b[8];
        for (int i = 0; i < 8; i++) b[i] = (uint8_t)(hl >> (8 * i));
        /* 8-byte header plus 8 dummy bytes: the old hl+8 check wraps and would accept this. */
        if (f && fwrite(b, 1, 8, f) == 8 && fwrite("xxxxxxxx", 1, 8, f) == 8 && fclose(f) == 0) {
            rc = pk_st_open(&st, "pk_st_wrap.st");
            CHECK(rc == -2, "pk_st_open refuses a wrapping header length (rc=%d)", rc);
            pk_st_close(&st);
        } else CHECK(0, "write wrapping-header file");
        remove("pk_st_wrap.st");
    }
}

static void test_with_model(const char *dir) {
    pk_model m; if (pk_model_load(&m, dir)) { CHECK(0, "model load %s", dir); return; }
    CHECK(m.n_layers == 24 && m.d_model == 1024 && m.n_heads == 8 && m.vocab == 1024 && m.is_tdt == 1 && m.n_dur == 5,
          "model dims: %d layers d%d h%d ff%d vocab %d tdt %d durs %d feat %d", m.n_layers, m.d_model, m.n_heads, m.d_ff, m.vocab, m.is_tdt, m.n_dur, m.feat_in);
    pk_mel d; pk_mel_default_fb(&d, m.mel.nmel);
    double e = 0, w = 0; for (int i = 0; i < m.mel.nmel * 257; i++) { double x = fabs(d.fb[i] - m.mel.fb[i]); if (x > e) e = x; }
    for (int i = 0; i < 400; i++) { double x = fabs(d.window[i] - m.mel.window[i]); if (x > w) w = x; }
    CHECK(e < 1e-5, "our slaney mel fb matches checkpoint fb, max err %.2e", e);
    CHECK(w < 1e-6, "our Hann(periodic=False) matches checkpoint window, max err %.2e", w);
    CHECK(m.n_pieces == 1024, "tokenizer.model pieces %d (piece[1]='%s')", m.n_pieces, m.n_pieces > 1 ? m.pieces[1] : "");
    int ids[] = { 5, 40 }; char *s = pk_detok(&m, ids, 2);
    CHECK(!strcmp(s, "the you"), "detok real vocab [5,40] -> '%s'", s); free(s);
}

int main(int argc, char **argv) {
    test_fft(); test_gemm(1); test_gemm(8); test_pool(); test_resample(); test_wav(); test_mel_synthetic(); test_spm_synth(); test_st_bounds();
    if (argc > 1) test_with_model(argv[1]);
    printf("%d PASS / %d FAIL\n", passes, fails);
    return fails ? 1 : 0;
}
