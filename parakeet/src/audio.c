/* parakeet-c audio: WAV reader (PCM 8/16/24/32, float32/64, any channel count,
 * WAVE_FORMAT_EXTENSIBLE), windowed-sinc resampler to 16 kHz mono, WAV writer. */
#include "pk.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static uint32_t rd32(const uint8_t *p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }
static uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] | p[1] << 8); }

float *pk_resample(const float *x, int n, int sr_in, int sr_out, int *n_out) {
    if (sr_in == sr_out) { float *o = malloc(sizeof(float) * (n ? n : 1)); memcpy(o, x, sizeof(float) * n); *n_out = n; return o; }
    double ratio = (double)sr_out / sr_in;
    int no = (int)floor(n * ratio);
    float *o = malloc(sizeof(float) * (no ? no : 1));
    double cutoff = ratio < 1 ? ratio : 1.0;   /* normalized to input Nyquist */
    const int half = 32;                        /* taps each side (in output-scaled units) */
    double scale = cutoff;
    for (int i = 0; i < no; i++) {
        double t = i / ratio;                   /* position in input samples */
        int c = (int)floor(t);
        double acc = 0, wsum = 0;
        int r = (int)ceil(half / cutoff);
        for (int k = c - r + 1; k <= c + r; k++) {
            double d = (t - k) * scale;
            double s = fabs(d) < 1e-9 ? 1.0 : sin(M_PI * d) / (M_PI * d);
            double wx = (t - k) / (r + 1e-9);
            double win = fabs(wx) >= 1 ? 0 : 0.5 + 0.5 * cos(M_PI * wx); /* Hann */
            double w = s * win;
            wsum += w;
            if (k >= 0 && k < n) acc += x[k] * w;
        }
        (void)wsum;
        o[i] = (float)(acc * scale);
    }
    *n_out = no;
    return o;
}

int pk_wav_read(const char *path, pk_audio *out) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc(sz);
    if (fread(b, 1, sz, f) != (size_t)sz) { fclose(f); free(b); return -1; }
    fclose(f);
    if (sz < 12 || memcmp(b, "RIFF", 4) || memcmp(b + 8, "WAVE", 4)) { free(b); return -2; }
    int fmt = 0, ch = 0, sr = 0, bits = 0; const uint8_t *data = NULL; uint32_t dlen = 0;
    long p = 12;
    while (p + 8 <= sz) {
        uint32_t cl = rd32(b + p + 4);
        if (!memcmp(b + p, "fmt ", 4)) {
            fmt = rd16(b + p + 8); ch = rd16(b + p + 10); sr = (int)rd32(b + p + 12); bits = rd16(b + p + 22);
            if (fmt == 0xFFFE && cl >= 40) fmt = rd16(b + p + 8 + 24);
        } else if (!memcmp(b + p, "data", 4)) {
            data = b + p + 8; dlen = cl;
            if ((long)(p + 8 + dlen) > sz) dlen = (uint32_t)(sz - p - 8);
        }
        p += 8 + cl + (cl & 1);
    }
    if (!data || ch <= 0 || sr <= 0 || (fmt != 1 && fmt != 3)) { free(b); return -3; }
    int bps = bits / 8, frames = (int)(dlen / (bps * ch));
    float *mono = malloc(sizeof(float) * (frames ? frames : 1));
    for (int i = 0; i < frames; i++) {
        double s = 0;
        for (int c = 0; c < ch; c++) {
            const uint8_t *q = data + ((size_t)i * ch + c) * bps; double v = 0;
            if (fmt == 1) {
                if (bps == 1) v = (q[0] - 128) / 128.0;
                else if (bps == 2) v = (int16_t)rd16(q) / 32768.0;
                else if (bps == 3) v = ((int32_t)((uint32_t)q[0] << 8 | (uint32_t)q[1] << 16 | (uint32_t)q[2] << 24) >> 8) / 8388608.0;
                else if (bps == 4) v = (int32_t)rd32(q) / 2147483648.0;
            } else {
                if (bps == 4) { float fv; memcpy(&fv, q, 4); v = fv; }
                else if (bps == 8) { double dv; memcpy(&dv, q, 8); v = dv; }
            }
            s += v;
        }
        mono[i] = (float)(s / ch);
    }
    free(b);
    out->x = pk_resample(mono, frames, sr, PK_SR, &out->n);
    free(mono);
    return 0;
}

int pk_wav_write16(const char *path, const float *x, int n) {
    FILE *f = fopen(path, "wb"); if (!f) return -1;
    uint32_t dl = (uint32_t)n * 2, v32; uint16_t v16;
    fwrite("RIFF", 1, 4, f); v32 = 36 + dl; fwrite(&v32, 4, 1, f); fwrite("WAVEfmt ", 1, 8, f);
    v32 = 16; fwrite(&v32, 4, 1, f); v16 = 1; fwrite(&v16, 2, 1, f); fwrite(&v16, 2, 1, f);
    v32 = PK_SR; fwrite(&v32, 4, 1, f); v32 = PK_SR * 2; fwrite(&v32, 4, 1, f);
    v16 = 2; fwrite(&v16, 2, 1, f); v16 = 16; fwrite(&v16, 2, 1, f);
    fwrite("data", 1, 4, f); fwrite(&dl, 4, 1, f);
    for (int i = 0; i < n; i++) { float s = x[i] * 32767.f; if (s > 32767) s = 32767; if (s < -32768) s = -32768; int16_t q = (int16_t)lrintf(s); fwrite(&q, 2, 1, f); }
    fclose(f); return 0;
}
