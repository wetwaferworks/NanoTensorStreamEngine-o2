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

/* ---------- hostile inputs: each must be refused cleanly, never read past a buffer ---------- */
static int write_bytes(const char *path, const void *b, size_t n) {
    FILE *f = fopen(path, "wb"); if (!f) return -1;
    int bad = fwrite(b, 1, n, f) != n; if (fclose(f)) bad = 1; return bad ? -1 : 0;
}
static size_t put32(uint8_t *b, uint32_t v) { for (int i = 0; i < 4; i++) b[i] = (uint8_t)(v >> (8 * i)); return 4; }
static size_t put16(uint8_t *b, uint16_t v) { b[0] = (uint8_t)v; b[1] = (uint8_t)(v >> 8); return 2; }
/* RIFF/WAVE with one fmt chunk (declared length fmt_len, 16 bytes written) and one data chunk */
static size_t make_wav(uint8_t *b, uint16_t fmt, uint16_t ch, uint32_t sr, uint16_t bits, uint32_t fmt_len, uint32_t data_len_decl, uint32_t data_bytes) {
    size_t n = 0;
    memcpy(b, "RIFF", 4); n += 4; n += put32(b + n, 0); memcpy(b + n, "WAVE", 4); n += 4;
    memcpy(b + n, "fmt ", 4); n += 4; n += put32(b + n, fmt_len);
    n += put16(b + n, fmt); n += put16(b + n, ch); n += put32(b + n, sr); n += put32(b + n, sr * ch * (bits / 8u));
    n += put16(b + n, (uint16_t)(ch * (bits / 8u))); n += put16(b + n, bits);
    memcpy(b + n, "data", 4); n += 4; n += put32(b + n, data_len_decl);
    memset(b + n, 0x11, data_bytes); n += data_bytes;
    return n;
}
static void test_wav_hostile(void) {
    uint8_t b[512]; size_t n; pk_audio a; int rc;
    struct { const char *what; uint16_t fmt, ch; uint32_t sr; uint16_t bits; uint32_t fmt_len, dlen; } cases[] = {
        { "PCM with 4 bits per sample (old divide by zero)", 1, 1, 16000, 4, 16, 64 },
        { "PCM with 0 bits per sample", 1, 1, 16000, 0, 16, 64 },
        { "PCM with 12 bits per sample (not a byte multiple)", 1, 1, 16000, 12, 16, 64 },
        { "float with 16 bits per sample", 3, 1, 16000, 16, 16, 64 },
        { "zero channels", 1, 0, 16000, 16, 16, 64 },
        { "sample rate 0", 1, 1, 0, 16, 16, 64 },
        { "sample rate 1 Hz (resampled length would overflow)", 1, 1, 1, 16, 16, 64 },
        { "fmt chunk shorter than 16 bytes", 1, 1, 16000, 16, 8, 64 },
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        n = make_wav(b, cases[i].fmt, cases[i].ch, cases[i].sr, cases[i].bits, cases[i].fmt_len, cases[i].dlen, 64);
        write_bytes("pk_hostile.wav", b, n);
        rc = pk_wav_read("pk_hostile.wav", &a);
        CHECK(rc != 0, "wav refuses %s (rc=%d)", cases[i].what, rc);
    }
    /* data chunk claiming ~4 GB: clamped to the bytes present, no wrap */
    n = make_wav(b, 1, 1, 16000, 16, 16, 0xFFFFFFF0u, 64);
    write_bytes("pk_hostile.wav", b, n);
    rc = pk_wav_read("pk_hostile.wav", &a);
    CHECK(rc == 0 && a.n == 32, "wav clamps an oversized data length to the file (rc=%d n=%d)", rc, rc ? -1 : a.n);
    if (!rc) free(a.x);
    /* an unknown chunk claiming ~4 GB before fmt: the walk ends, nothing is read past the file */
    n = 0; memcpy(b, "RIFF", 4); n = 8; memcpy(b + n, "WAVE", 4); n += 4;
    memcpy(b + n, "junk", 4); n += 4; n += put32(b + n, 0xFFFFFFF8u); memset(b + n, 0, 16); n += 16;
    write_bytes("pk_hostile.wav", b, n);
    rc = pk_wav_read("pk_hostile.wav", &a);
    CHECK(rc != 0, "wav with a 4 GB junk chunk and no fmt is refused (rc=%d)", rc);
    remove("pk_hostile.wav");
}
static void test_spm_hostile(void) {
    char **pieces = NULL; int np = 0, rc;
    /* valid: two pieces, "a" (normal) and "<0x41>" (byte) */
    const uint8_t ok[] = { 0x0A, 0x03, 0x0A, 0x01, 'a',
                           0x0A, 0x0A, 0x0A, 0x06, '<', '0', 'x', '4', '1', '>', 0x18, 0x06 };
    write_bytes("pk_spm.model", ok, sizeof ok);
    rc = pk_spm_load("pk_spm.model", &pieces, &np);
    CHECK(rc == 0 && np == 2 && !strcmp(pieces[0], "a") && !strcmp(pieces[1], "<0x41>"), "spm loads a valid two-piece model (rc=%d n=%d)", rc, np);
    if (!rc) { int ids[] = { 0, 1 }; char *s = pk_spm_detok(pieces, np, ids, 2); CHECK(!strcmp(s, "aA"), "spm byte piece decodes ('%s')", s); free(s); }
    struct { const char *what; uint8_t b[12]; size_t n; } bad[] = {
        { "piece message longer than the file", { 0x0A, 0x7F, 0x0A, 0x01, 'a' }, 5 },
        { "piece string longer than its message", { 0x0A, 0x03, 0x0A, 0x40, 'a' }, 5 },
        { "string length of 2^63 (pointer wrap)", { 0x0A, 0x0B, 0x0A, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80 }, 12 },
        { "skipped field longer than the file", { 0x12, 0x7F, 'x' }, 3 },
        { "fixed64 field cut short", { 0x09, 1, 2, 3 }, 4 },
        { "invalid wire type", { 0x0F, 0x00 }, 2 },
    };
    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; i++) {
        write_bytes("pk_spm.model", bad[i].b, bad[i].n);
        char **p2 = NULL; int n2 = 0;
        rc = pk_spm_load("pk_spm.model", &p2, &n2);
        CHECK(rc != 0, "spm refuses %s (rc=%d)", bad[i].what, rc);
    }
    remove("pk_spm.model");
}
static void good_dims(pk_model *m) {
    memset(m, 0, sizeof *m);
    m->d_model = 1024; m->n_heads = 8; m->n_layers = 24; m->d_ff = 4096; m->conv_ks = 9; m->sub_ch = 256; m->feat_in = 128;
    m->vocab = 1024; m->is_tdt = 1; m->n_dur = 5; m->pred_hidden = 640; m->pred_layers = 2; m->joint_hidden = 640;
    m->j_enc.N = 640; m->j_enc.K = 1024; m->j_pred.N = 640; m->j_pred.K = 640; m->j_out.N = 1024 + 1 + 5; m->j_out.K = 640;
}
static void test_model_dims(void) {
    pk_model m; const int ALL = PK_CHECK_ENCODER | PK_CHECK_DECODER;
    good_dims(&m); CHECK(pk_model_check_dims(&m, ALL) == 0, "dims: parakeet-tdt-0.6b shapes accepted");
    struct { const char *what; int field; int value; } bad[] = {
        { "prediction width 2048 (LSTM state holds 1024)", 0, 2048 },
        { "joint width 4096 (joint buffers hold 1024)", 1, 4096 },
        { "9 TDT durations (table holds 8)", 2, 9 },
        { "head width 512 (attention rows hold 256)", 3, 2 },
        { "3 LSTM layers", 4, 3 },
        { "heads that do not divide d_model", 5, 7 },
        { "even conv kernel", 6, 8 },
        { "zero d_model", 7, 0 },
        { "vocabulary 0", 8, 0 },
    };
    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; i++) {
        good_dims(&m);
        switch (bad[i].field) {
        case 0: m.pred_hidden = bad[i].value; m.j_pred.K = bad[i].value; break;
        case 1: m.joint_hidden = bad[i].value; m.j_enc.N = m.j_pred.N = m.j_out.K = bad[i].value; break;
        case 2: m.n_dur = bad[i].value; m.j_out.N = m.vocab + 1 + bad[i].value; break;
        case 3: m.n_heads = bad[i].value; break;
        case 4: m.pred_layers = bad[i].value; break;
        case 5: m.n_heads = bad[i].value; break;
        case 6: m.conv_ks = bad[i].value; break;
        case 7: m.d_model = bad[i].value; break;
        case 8: m.vocab = bad[i].value; m.j_out.N = 1 + m.n_dur; break;
        }
        CHECK(pk_model_check_dims(&m, ALL) != 0, "dims: rejects %s", bad[i].what);
    }
    good_dims(&m); m.j_out.N = 1024 + 1 + 4;
    CHECK(pk_model_check_dims(&m, ALL) != 0, "dims: rejects a joint output that is not vocab + blank + durations");
    good_dims(&m); m.j_pred.K = 320;
    CHECK(pk_model_check_dims(&m, ALL) != 0, "dims: rejects a joint prediction input that is not the LSTM width");
    pk_layer L; memset(&L, 0, sizeof L);
    good_dims(&m);
    const int D = m.d_model, F = m.d_ff;
    pk_lin *ls[] = { &L.ff1_1, &L.ff1_2, &L.ff2_1, &L.ff2_2, &L.q, &L.k, &L.v, &L.o, &L.pos, &L.pw1, &L.pw2 };
    int nk[][2] = { {F, D}, {D, F}, {F, D}, {D, F}, {D, D}, {D, D}, {D, D}, {D, D}, {D, D}, {2 * D, D}, {D, D} };
    for (int i = 0; i < 11; i++) { ls[i]->N = nk[i][0]; ls[i]->K = nk[i][1]; }
    CHECK(pk_model_check_layer(&m, &L) == 0, "dims: a consistent encoder layer is accepted");
    L.pw1.N = D;   /* GLU would read 2*D per row from a D-wide output */
    CHECK(pk_model_check_layer(&m, &L) != 0, "dims: rejects pointwise_conv1 that is not 2*d_model wide");
}
static void test_st_shapes(void) {
    float four[4] = { 1, 2, 3, 4 }; pk_st st; int rc;
    const char *more = "{\"w\":{\"dtype\":\"F32\",\"shape\":[8],\"data_offsets\":[0,16]}}";
    if (write_st("pk_st_shape.st", more, four, 16) == 0) {
        rc = pk_st_open(&st, "pk_st_shape.st");
        CHECK(rc == -9, "pk_st_open refuses a shape larger than its bytes (rc=%d)", rc);
        pk_st_close(&st);
    }
    const char *five = "{\"w\":{\"dtype\":\"F32\",\"shape\":[1,1,1,1,4],\"data_offsets\":[0,16]}}";
    if (write_st("pk_st_shape.st", five, four, 16) == 0) {
        rc = pk_st_open(&st, "pk_st_shape.st");
        CHECK(rc == -8, "pk_st_open refuses more than four dimensions (rc=%d)", rc);
        pk_st_close(&st);
    }
    const char *okj = "{\"w\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}";
    if (write_st("pk_st_shape.st", okj, four, 16) == 0) {
        rc = pk_st_open(&st, "pk_st_shape.st");
        CHECK(rc == 0, "pk_st_open accepts a shape that matches its bytes (rc=%d)", rc);
        pk_st_close(&st);
    }
    remove("pk_st_shape.st");
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
    test_wav_hostile(); test_spm_hostile(); test_model_dims(); test_st_shapes();
    if (argc > 1) test_with_model(argv[1]);
    printf("%d PASS / %d FAIL\n", passes, fails);
    return fails ? 1 : 0;
}
