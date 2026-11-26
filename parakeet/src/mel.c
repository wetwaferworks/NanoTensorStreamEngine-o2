/* parakeet-c mel front end, matching NeMo FilterbankFeatures (inference):
 * preemphasis 0.97 -> STFT (n_fft 512, win 400 Hann periodic=False, hop 160,
 * center=True, zero pad) -> power -> mel (slaney) -> log(x + 2^-24) ->
 * per-feature normalize over valid frames (unbiased std + 1e-5).
 * Valid frames T = n / 160 (NeMo get_seq_len); dither is training-only. */
#include "pk.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float tw_re[256], tw_im[256]; static int rev[512]; static int fft_ready;
static void fft_init(void) {
    for (int i = 0; i < 256; i++) { double a = -2.0 * M_PI * i / 512.0; tw_re[i] = (float)cos(a); tw_im[i] = (float)sin(a); }
    for (int i = 0; i < 512; i++) { int r = 0; for (int b = 0; b < 9; b++) if (i & (1 << b)) r |= 1 << (8 - b); rev[i] = r; }
    fft_ready = 1;
}
void pk_fft_real512(const float *in, float *re, float *im) {
    if (!fft_ready) fft_init();
    float xr[512], xi[512];
    for (int i = 0; i < 512; i++) { xr[rev[i]] = in[i]; xi[rev[i]] = 0; }
    for (int len = 2; len <= 512; len <<= 1) {
        int h = len >> 1, step = 512 / len;
        for (int s = 0; s < 512; s += len)
            for (int k = 0; k < h; k++) {
                float wr = tw_re[k * step], wi = tw_im[k * step];
                int a = s + k, b = a + h;
                float tr = xr[b] * wr - xi[b] * wi, ti = xr[b] * wi + xi[b] * wr;
                xr[b] = xr[a] - tr; xi[b] = xi[a] - ti; xr[a] += tr; xi[a] += ti;
            }
    }
    for (int i = 0; i < PK_NBIN; i++) { re[i] = xr[i]; im[i] = xi[i]; }
}

static double hz_to_mel(double f) {
    const double f_sp = 200.0 / 3, min_log_hz = 1000.0, min_log_mel = min_log_hz / f_sp, logstep = log(6.4) / 27.0;
    return f >= min_log_hz ? min_log_mel + log(f / min_log_hz) / logstep : f / f_sp;
}
static double mel_to_hz(double m) {
    const double f_sp = 200.0 / 3, min_log_hz = 1000.0, min_log_mel = min_log_hz / f_sp, logstep = log(6.4) / 27.0;
    return m >= min_log_mel ? min_log_hz * exp(logstep * (m - min_log_mel)) : f_sp * m;
}
void pk_mel_default_fb(pk_mel *m, int nmel) {
    m->nmel = nmel; m->preemph = 0.97f; m->log_guard = 5.9604644775390625e-08f;
    m->fb = pk_alloc(sizeof(float) * nmel * PK_NBIN);
    m->window = pk_alloc(sizeof(float) * PK_WIN);
    for (int i = 0; i < PK_WIN; i++) m->window[i] = (float)(0.5 - 0.5 * cos(2.0 * M_PI * i / (PK_WIN - 1)));
    double mlo = hz_to_mel(0), mhi = hz_to_mel(PK_SR / 2.0);
    double *mf = malloc(sizeof(double) * (nmel + 2));
    for (int i = 0; i < nmel + 2; i++) mf[i] = mel_to_hz(mlo + (mhi - mlo) * i / (nmel + 1));
    for (int i = 0; i < nmel; i++) {
        double enorm = 2.0 / (mf[i + 2] - mf[i]);
        for (int k = 0; k < PK_NBIN; k++) {
            double f = (PK_SR / 2.0) * k / (PK_NBIN - 1);
            double lo = (f - mf[i]) / (mf[i + 1] - mf[i]), hi = (mf[i + 2] - f) / (mf[i + 2] - mf[i + 1]);
            double w = lo < hi ? lo : hi; if (w < 0) w = 0;
            m->fb[i * PK_NBIN + k] = (float)(w * enorm);
        }
    }
    free(mf);
}

int pk_mel_compute(const pk_mel *m, const float *x, int n, float **out) {
    int T = n / PK_HOP;
    if (T < 1) { *out = NULL; return 0; }
    const int nm = m->nmel;
    float *pe = malloc(sizeof(float) * n);
    pe[0] = x[0];
    for (int i = 1; i < n; i++) pe[i] = x[i] - m->preemph * x[i - 1];
    float *o = pk_alloc(sizeof(float) * (size_t)T * nm);
    float fr[512], re[PK_NBIN], im[PK_NBIN], pw[PK_NBIN];
    const int off = (PK_NFFT - PK_WIN) / 2; /* 56 */
    for (int t = 0; t < T; t++) {
        int s0 = t * PK_HOP - PK_NFFT / 2;
        memset(fr, 0, sizeof fr);
        for (int i = 0; i < PK_WIN; i++) { int s = s0 + off + i; fr[off + i] = (s >= 0 && s < n) ? pe[s] * m->window[i] : 0.f; }
        pk_fft_real512(fr, re, im);
        for (int k = 0; k < PK_NBIN; k++) { float mag = sqrtf(re[k] * re[k] + im[k] * im[k]); pw[k] = mag * mag; }
        for (int j = 0; j < nm; j++) {
            const float *w = m->fb + j * PK_NBIN; float s = 0;
            for (int k = 0; k < PK_NBIN; k++) s += w[k] * pw[k];
            o[(size_t)t * nm + j] = logf(s + m->log_guard);
        }
    }
    free(pe);
    for (int j = 0; j < nm; j++) {
        double mean = 0, var = 0;
        for (int t = 0; t < T; t++) mean += o[(size_t)t * nm + j];
        mean /= T;
        for (int t = 0; t < T; t++) { double d = o[(size_t)t * nm + j] - mean; var += d * d; }
        double sd = T > 1 ? sqrt(var / (T - 1)) : 0.0;
        float inv = (float)(1.0 / (sd + 1e-5)), mu = (float)mean;
        for (int t = 0; t < T; t++) o[(size_t)t * nm + j] = (o[(size_t)t * nm + j] - mu) * inv;
    }
    *out = o;
    return T;
}
