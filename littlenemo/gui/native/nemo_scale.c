/* nemo_scale.c - premultiplied RGBA8 scaler. See nemo_scale.h.
 *
 * Safe-critical rules:
 *   - no globals, no locks, no threads
 *   - every dimension is range-checked before use
 *   - every size (stride, pixel count, workspace) is overflow-checked
 *   - heap is used only for a per-call workspace in nemo_scale_rgba8,
 *     and is freed on every path
 */
#include "nemo_scale.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define NEMO_SCALE_VERSION_STR "ntse-scale 1.0"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ----------------- size / argument checks ----------------- */

static int mul_ok(size_t a, size_t b, size_t *out)
{
    if (b != 0 && a > (size_t)-1 / b)
        return 0;
    *out = a * b;
    return 1;
}

/* Tight packed stride for width pixels of RGBA8, or 0 on overflow. */
static int packed_stride(int width, int *out)
{
    if (width <= 0 || width > NEMO_SCALE_MAX_DIM)
        return NEMO_SCALE_ERR_DIM;
    if (width > (2147483647 / 4))
        return NEMO_SCALE_ERR_OVERFLOW;
    *out = width * 4;
    return NEMO_SCALE_OK;
}

static int check_dims(int src_w, int src_h, int dst_w, int dst_h)
{
    if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0)
        return NEMO_SCALE_ERR_DIM;
    if (src_w > NEMO_SCALE_MAX_DIM || src_h > NEMO_SCALE_MAX_DIM ||
        dst_w > NEMO_SCALE_MAX_DIM || dst_h > NEMO_SCALE_MAX_DIM)
        return NEMO_SCALE_ERR_DIM;
    {
        uint64_t sp = (uint64_t)src_w * (uint64_t)src_h;
        uint64_t dp = (uint64_t)dst_w * (uint64_t)dst_h;
        if (sp > (uint64_t)NEMO_SCALE_MAX_PIXELS ||
            dp > (uint64_t)NEMO_SCALE_MAX_PIXELS)
            return NEMO_SCALE_ERR_OVERFLOW;
    }
    return NEMO_SCALE_OK;
}

static int resolve_stride(int width, int stride, int *out)
{
    int packed, st;
    int err = packed_stride(width, &packed);
    if (err != NEMO_SCALE_OK)
        return err;
    st = (stride == 0) ? packed : stride;
    if (st < packed)
        return NEMO_SCALE_ERR_BUF;
    *out = st;
    return NEMO_SCALE_OK;
}

/* ----------------- kernels ----------------- */

static double sinc(double x)
{
    if (x == 0.0)
        return 1.0;
    x *= M_PI;
    return sin(x) / x;
}

static double kernel_lanczos3(double x)
{
    double ax = fabs(x);
    if (ax >= 3.0)
        return 0.0;
    return sinc(x) * sinc(x / 3.0);
}

/* Mitchell-Netravali, B = C = 1/3. */
static double kernel_mitchell(double x)
{
    const double B = 1.0 / 3.0;
    const double C = 1.0 / 3.0;
    double ax = fabs(x);
    double ax2 = ax * ax;
    double ax3 = ax2 * ax;
    if (ax < 1.0) {
        return ((12.0 - 9.0 * B - 6.0 * C) * ax3 +
                (-18.0 + 12.0 * B + 6.0 * C) * ax2 +
                (6.0 - 2.0 * B)) / 6.0;
    }
    if (ax < 2.0) {
        return ((-B - 6.0 * C) * ax3 +
                (6.0 * B + 30.0 * C) * ax2 +
                (-12.0 * B - 48.0 * C) * ax +
                (8.0 * B + 24.0 * C)) / 6.0;
    }
    return 0.0;
}

static double kernel_weight(int filter, double x)
{
    if (filter == NEMO_SCALE_LANCZOS3)
        return kernel_lanczos3(x);
    return kernel_mitchell(x);
}

static double kernel_radius(int filter)
{
    return (filter == NEMO_SCALE_LANCZOS3) ? 3.0 : 2.0;
}

/* ----------------- row / pixel helpers ----------------- */

static void copy_rows(const uint8_t *src, int src_stride,
                      uint8_t *dst, int dst_stride,
                      int row_bytes, int rows)
{
    int y;
    for (y = 0; y < rows; y++)
        memcpy(dst + (size_t)y * (size_t)dst_stride,
               src + (size_t)y * (size_t)src_stride,
               (size_t)row_bytes);
}

static void store_premul(uint8_t *p, double r, double g, double b, double a)
{
    /* Clamp to a valid premultiplied pixel: 0 <= rgb <= a <= 255. */
    if (a < 0.0) a = 0.0;
    if (a > 255.0) a = 255.0;
    if (r < 0.0) r = 0.0;
    if (g < 0.0) g = 0.0;
    if (b < 0.0) b = 0.0;
    if (r > a) r = a;
    if (g > a) g = a;
    if (b > a) b = a;
    p[0] = (uint8_t)(r + 0.5);
    p[1] = (uint8_t)(g + 0.5);
    p[2] = (uint8_t)(b + 0.5);
    p[3] = (uint8_t)(a + 0.5);
}

/* ----------------- nearest (integer-snap) ----------------- */

static int scale_nearest(const uint8_t *src, int src_w, int src_h, int src_st,
                         uint8_t *dst, int dst_w, int dst_h, int dst_st)
{
    int x, y;
    /* Floor mapping so integer 2x/3x is exact pixel replication:
     * dest x maps to src (x * src_w) / dst_w. */
    for (y = 0; y < dst_h; y++) {
        int sy = (int)(((int64_t)y * (int64_t)src_h) / (int64_t)dst_h);
        const uint8_t *srow;
        uint8_t *drow;
        if (sy < 0) sy = 0;
        if (sy >= src_h) sy = src_h - 1;
        srow = src + (size_t)sy * (size_t)src_st;
        drow = dst + (size_t)y * (size_t)dst_st;
        for (x = 0; x < dst_w; x++) {
            int sx = (int)(((int64_t)x * (int64_t)src_w) / (int64_t)dst_w);
            const uint8_t *sp;
            if (sx < 0) sx = 0;
            if (sx >= src_w) sx = src_w - 1;
            sp = srow + (size_t)sx * 4u;
            drow[(size_t)x * 4u + 0] = sp[0];
            drow[(size_t)x * 4u + 1] = sp[1];
            drow[(size_t)x * 4u + 2] = sp[2];
            drow[(size_t)x * 4u + 3] = sp[3];
        }
    }
    return NEMO_SCALE_OK;
}

/* ----------------- separable filter ----------------- */

/* Scale one axis of a tightly-packed RGBA8 buffer (src_len along the
 * axis, n_lines orthogonal). src_is_horizontal: pixels along the row.
 * When vertical, each "line" is a column walked with stride `across`. */
static int pass_axis(const uint8_t *src, int src_len, int src_stride,
                     uint8_t *dst, int dst_len, int dst_stride,
                     int n_lines, int filter)
{
    double radius = kernel_radius(filter);
    double scale = (double)src_len / (double)dst_len;
    /* When shrinking, widen the kernel so we still sample the box. */
    double support = (scale > 1.0) ? radius * scale : radius;
    int line, i;

    for (i = 0; i < dst_len; i++) {
        /* Map the centre of dest pixel i into source space. */
        double center = ((double)i + 0.5) * scale - 0.5;
        double lo = center - support;
        double hi = center + support;
        int i0 = (int)floor(lo);
        int i1 = (int)ceil(hi);
        int k, n;
        /* Bounded: support is at most 3 * (MAX_DIM) in theory if someone
         * scaled 1 -> MAX, but src_len is already <= MAX_DIM. */
        if (i0 < 0) i0 = 0;
        if (i1 >= src_len) i1 = src_len - 1;
        if (i1 < i0) i1 = i0;
        n = i1 - i0 + 1;
        if (n <= 0)
            return NEMO_SCALE_ERR_DIM;

        /* Weights on the stack; n is at most src_len, but typically
         * a handful (radius 3, or a few tens when shrinking a lot).
         * Cap the stack table and fall back to a short local compute
         * in the inner loop if n is huge — still no extra heap. */
        {
            /* Recompute weights per dest pixel (cheap at GUI sizes). */
            for (line = 0; line < n_lines; line++) {
                double r = 0.0, g = 0.0, b = 0.0, a = 0.0, wsum = 0.0;
                uint8_t *dp;
                for (k = i0; k <= i1; k++) {
                    double w, x;
                    const uint8_t *sp;
                    x = ((double)k - center) / ((scale > 1.0) ? scale : 1.0);
                    w = kernel_weight(filter, x);
                    if (w == 0.0)
                        continue;
                    sp = src + (size_t)line * (size_t)src_stride + (size_t)k * 4u;
                    r += w * (double)sp[0];
                    g += w * (double)sp[1];
                    b += w * (double)sp[2];
                    a += w * (double)sp[3];
                    wsum += w;
                }
                if (wsum > 0.0) {
                    r /= wsum;
                    g /= wsum;
                    b /= wsum;
                    a /= wsum;
                } else {
                    r = g = b = a = 0.0;
                }
                dp = dst + (size_t)line * (size_t)dst_stride + (size_t)i * 4u;
                store_premul(dp, r, g, b, a);
            }
        }
    }
    return NEMO_SCALE_OK;
}

static int scale_filter(const uint8_t *src, int src_w, int src_h, int src_st,
                        uint8_t *dst, int dst_w, int dst_h, int dst_st,
                        int filter, void *work, size_t work_bytes)
{
    /* Intermediate: dst_w x src_h, tightly packed. */
    size_t tmp_stride, tmp_bytes;
    uint8_t *tmp;
    int err;

    if (!mul_ok((size_t)dst_w, 4u, &tmp_stride))
        return NEMO_SCALE_ERR_OVERFLOW;
    if (!mul_ok(tmp_stride, (size_t)src_h, &tmp_bytes))
        return NEMO_SCALE_ERR_OVERFLOW;
    if (work == NULL || work_bytes < tmp_bytes)
        return NEMO_SCALE_ERR_NOMEM;
    tmp = (uint8_t *)work;

    /* Horizontal: each of src_h rows, src_w -> dst_w. */
    err = pass_axis(src, src_w, src_st, tmp, dst_w, (int)tmp_stride,
                    src_h, filter);
    if (err != NEMO_SCALE_OK)
        return err;

    /* Vertical: for each of dst_w columns, src_h -> dst_h.
     * pass_axis walks n_lines "rows" with pixels along the axis at
     * 4-byte steps and a line stride. For a column, the "pixels"
     * along the axis are `tmp_stride` apart, and adjacent columns
     * are 4 bytes apart. Easier: treat the intermediate as rows
     * and run a vertical gather. */
    {
        int x, y;
        double radius = kernel_radius(filter);
        double scale = (double)src_h / (double)dst_h;
        double support = (scale > 1.0) ? radius * scale : radius;
        for (y = 0; y < dst_h; y++) {
            double center = ((double)y + 0.5) * scale - 0.5;
            int i0 = (int)floor(center - support);
            int i1 = (int)ceil(center + support);
            uint8_t *drow = dst + (size_t)y * (size_t)dst_st;
            if (i0 < 0) i0 = 0;
            if (i1 >= src_h) i1 = src_h - 1;
            if (i1 < i0) i1 = i0;
            for (x = 0; x < dst_w; x++) {
                int k;
                double r = 0.0, g = 0.0, b = 0.0, a = 0.0, wsum = 0.0;
                for (k = i0; k <= i1; k++) {
                    double w, t;
                    const uint8_t *sp;
                    t = ((double)k - center) / ((scale > 1.0) ? scale : 1.0);
                    w = kernel_weight(filter, t);
                    if (w == 0.0)
                        continue;
                    sp = tmp + (size_t)k * tmp_stride + (size_t)x * 4u;
                    r += w * (double)sp[0];
                    g += w * (double)sp[1];
                    b += w * (double)sp[2];
                    a += w * (double)sp[3];
                    wsum += w;
                }
                if (wsum > 0.0) {
                    r /= wsum; g /= wsum; b /= wsum; a /= wsum;
                } else {
                    r = g = b = a = 0.0;
                }
                store_premul(drow + (size_t)x * 4u, r, g, b, a);
            }
        }
    }
    (void)err;
    return NEMO_SCALE_OK;
}

/* ----------------- public API ----------------- */

const char *nemo_scale_version(void)
{
    return NEMO_SCALE_VERSION_STR;
}

size_t nemo_scale_workspace_size(int src_w, int src_h, int dst_w, int dst_h,
                                 int filter)
{
    size_t tmp_stride, tmp_bytes;
    if (check_dims(src_w, src_h, dst_w, dst_h) != NEMO_SCALE_OK)
        return 0;
    if (filter != NEMO_SCALE_LANCZOS3 && filter != NEMO_SCALE_MITCHELL)
        return 0;
    if (src_w == dst_w && src_h == dst_h)
        return 0;
    if (!mul_ok((size_t)dst_w, 4u, &tmp_stride))
        return 0;
    if (!mul_ok(tmp_stride, (size_t)src_h, &tmp_bytes))
        return 0;
    return tmp_bytes;
}

int nemo_scale_rgba8_ex(const uint8_t *src, int src_w, int src_h, int src_stride,
                        uint8_t *dst, int dst_w, int dst_h, int dst_stride,
                        int filter, void *work, size_t work_bytes)
{
    int err, src_st, dst_st, packed;

    if (src == NULL || dst == NULL)
        return NEMO_SCALE_ERR_NULL;
    err = check_dims(src_w, src_h, dst_w, dst_h);
    if (err != NEMO_SCALE_OK)
        return err;
    if (filter != NEMO_SCALE_NEAREST &&
        filter != NEMO_SCALE_LANCZOS3 &&
        filter != NEMO_SCALE_MITCHELL)
        return NEMO_SCALE_ERR_FILTER;
    err = resolve_stride(src_w, src_stride, &src_st);
    if (err != NEMO_SCALE_OK)
        return err;
    err = resolve_stride(dst_w, dst_stride, &dst_st);
    if (err != NEMO_SCALE_OK)
        return err;
    /* Identity: exact byte copy of each row, any filter. */
    if (src_w == dst_w && src_h == dst_h) {
        err = packed_stride(src_w, &packed);
        if (err != NEMO_SCALE_OK)
            return err;
        copy_rows(src, src_st, dst, dst_st, packed, src_h);
        return NEMO_SCALE_OK;
    }
    if (filter == NEMO_SCALE_NEAREST)
        return scale_nearest(src, src_w, src_h, src_st, dst, dst_w, dst_h, dst_st);
    return scale_filter(src, src_w, src_h, src_st, dst, dst_w, dst_h, dst_st,
                        filter, work, work_bytes);
}

int nemo_scale_rgba8(const uint8_t *src, int src_w, int src_h, int src_stride,
                     uint8_t *dst, int dst_w, int dst_h, int dst_stride,
                     int filter)
{
    size_t need = nemo_scale_workspace_size(src_w, src_h, dst_w, dst_h, filter);
    void *work = NULL;
    int err;
    if (need > 0) {
        work = malloc(need);
        if (work == NULL)
            return NEMO_SCALE_ERR_NOMEM;
    }
    err = nemo_scale_rgba8_ex(src, src_w, src_h, src_stride,
                              dst, dst_w, dst_h, dst_stride,
                              filter, work, need);
    free(work);
    return err;
}
