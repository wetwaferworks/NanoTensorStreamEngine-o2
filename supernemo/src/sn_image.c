/* sn_image.c - see sn_image.h. */
#include "sn_image.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_BMP
#define STBI_ONLY_GIF
#define STBI_NO_LINEAR
#define STBI_NO_HDR
#define STBI_WINDOWS_UTF8
#include "../third_party/stb_image.h"

int sn_image_load(const char *path, unsigned char **rgb, int *w, int *h, char *err, int errlen)
{
    int comp = 0;
    *rgb = stbi_load(path, w, h, &comp, 3);
    if (!*rgb) {
        const char *why = stbi_failure_reason();
        snprintf(err, (size_t)errlen, "cannot decode image (%s); PNG, JPEG, BMP and GIF are supported", why ? why : "unknown");
        return -1;
    }
    return 0;
}

/* Python round(): half to even */
static double py_round(double x) { return nearbyint(x); }

void sn_image_plan(int w, int h, int patch, int shuffle, int min_patches, int budget, int *wp, int *hp)
{
    int ch = (int)py_round((double)h / patch + 0.5), cw = (int)py_round((double)w / patch + 0.5);
    if (ch < 1) ch = 1;
    if (cw < 1) cw = 1;
    double patches = (double)ch * cw;
    double factor = sqrt((double)budget / patches); if (factor > 1.0) factor = 1.0;
    int th = (int)floor(factor * ch), tw = (int)floor(factor * cw);
    if (th < 1) th = 1;
    if (tw < 1) tw = 1;
    if (budget > min_patches && th * tw < min_patches) {
        double up = sqrt((double)min_patches / ((double)th * tw));
        th = (int)ceil(up * th); tw = (int)ceil(up * tw);
    }
    int rem = th % shuffle;
    if (rem) { int inc = shuffle - rem; if ((long)(th + inc) * tw <= budget) th += inc; else th = th - rem > shuffle ? th - rem : shuffle; }
    rem = tw % shuffle;
    if (rem) { int inc = shuffle - rem; if ((long)th * (tw + inc) <= budget) tw += inc; else tw = tw - rem > shuffle ? tw - rem : shuffle; }
    *wp = tw; *hp = th;
}

/* cubic convolution kernel, a = -0.5 (torch bicubic) */
static double cubic(double x)
{
    const double a = -0.5;
    x = fabs(x);
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    return 0.0;
}

typedef struct { int xmin, n; double *w; } taps_t;

/* torch's separable antialias weights for one axis: support grows with the downscale factor */
static taps_t *make_taps(int in, int out)
{
    double scale = (double)in / out;
    double support = scale >= 1.0 ? 2.0 * scale : 2.0, invs = scale >= 1.0 ? 1.0 / scale : 1.0;
    int maxn = (int)ceil(support) * 2 + 2;
    taps_t *t = (taps_t *)malloc((size_t)out * sizeof(taps_t));
    double *pool = (double *)malloc((size_t)out * maxn * sizeof(double));
    if (!t || !pool) { free(t); free(pool); return NULL; }
    for (int i = 0; i < out; i++) {
        double center = scale * (i + 0.5);
        int xmin = (int)(center - support + 0.5); if (xmin < 0) xmin = 0;
        int xmax = (int)(center + support + 0.5); if (xmax > in) xmax = in;
        int n = xmax - xmin; if (n > maxn) n = maxn;
        double *w = pool + (size_t)i * maxn, tot = 0;
        for (int j = 0; j < n; j++) { w[j] = cubic((j + xmin - center + 0.5) * invs); tot += w[j]; }
        if (tot != 0) for (int j = 0; j < n; j++) w[j] /= tot;
        t[i].xmin = xmin; t[i].n = n; t[i].w = w;
    }
    return t;
}

void sn_image_resize_plane(const float *src, int sw, int sh, float *dst, int dw, int dh)
{
    if (sw == dw && sh == dh) { memcpy(dst, src, (size_t)sw * sh * sizeof(float)); return; }
    taps_t *tx = make_taps(sw, dw), *ty = make_taps(sh, dh);
    float *mid = (float *)malloc((size_t)dw * sh * sizeof(float));
    if (!tx || !ty || !mid) { if (tx) { free(tx[0].w); free(tx); } if (ty) { free(ty[0].w); free(ty); } free(mid); return; }
    for (int y = 0; y < sh; y++)                         /* horizontal pass first, as torch does */
        for (int x = 0; x < dw; x++) {
            double s = 0; const float *r = src + (size_t)y * sw + tx[x].xmin;
            for (int j = 0; j < tx[x].n; j++) s += tx[x].w[j] * r[j];
            mid[(size_t)y * dw + x] = (float)s;
        }
    for (int y = 0; y < dh; y++)
        for (int x = 0; x < dw; x++) {
            double s = 0;
            for (int j = 0; j < ty[y].n; j++) s += ty[y].w[j] * mid[(size_t)(ty[y].xmin + j) * dw + x];
            dst[(size_t)y * dw + x] = (float)s;
        }
    free(tx[0].w); free(tx); free(ty[0].w); free(ty); free(mid);
}

float *sn_image_prepare(const unsigned char *rgb, int w, int h, int wp, int hp, int patch, const float mean[3], const float stdv[3])
{
    int dw = wp * patch, dh = hp * patch;
    float *out = (float *)malloc((size_t)3 * dw * dh * sizeof(float));
    float *plane = (float *)malloc((size_t)w * h * sizeof(float));
    if (!out || !plane) { free(out); free(plane); return NULL; }
    for (int c = 0; c < 3; c++) {
        for (size_t i = 0; i < (size_t)w * h; i++) plane[i] = (float)rgb[i * 3 + c];
        float *dst = out + (size_t)c * dw * dh;
        sn_image_resize_plane(plane, w, h, dst, dw, dh);
        for (size_t i = 0; i < (size_t)dw * dh; i++) dst[i] = (dst[i] / 255.0f - mean[c]) / stdv[c];
    }
    free(plane);
    return out;
}
