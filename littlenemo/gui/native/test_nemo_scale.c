/* test_nemo_scale.c - unit tests for the NTSE GUI scaler.
 * Build: gcc -O2 -std=c11 -Wall -Wextra -Werror -o test_nemo_scale \
 *            test_nemo_scale.c nemo_scale.c -lm
 */
#include "nemo_scale.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;

static void expect(int cond, const char *name)
{
    printf("%s %s\n", cond ? "ok  " : "FAIL", name);
    if (!cond)
        fails++;
}

static uint8_t *buf(int w, int h)
{
    size_t n = (size_t)w * (size_t)h * 4u;
    uint8_t *p = (uint8_t *)calloc(1, n);
    if (p == NULL) {
        fprintf(stderr, "oom\n");
        exit(2);
    }
    return p;
}

static void fill_pattern(uint8_t *p, int w, int h)
{
    int x, y;
    for (y = 0; y < h; y++) {
        for (x = 0; x < w; x++) {
            uint8_t *q = p + ((size_t)y * (size_t)w + (size_t)x) * 4u;
            /* Premultiplied: RGB already scaled by A. */
            uint8_t a = (uint8_t)(64 + ((x * 13 + y * 7) % 192));
            uint8_t r = (uint8_t)((x * a) / (w > 0 ? w : 1));
            uint8_t g = (uint8_t)((y * a) / (h > 0 ? h : 1));
            uint8_t b = (uint8_t)((a / 2));
            if (r > a) r = a;
            if (g > a) g = a;
            if (b > a) b = a;
            q[0] = r; q[1] = g; q[2] = b; q[3] = a;
        }
    }
}

static int equal_buf(const uint8_t *a, const uint8_t *b, size_t n)
{
    return memcmp(a, b, n) == 0;
}

static int premul_valid(const uint8_t *p, int w, int h)
{
    int x, y;
    for (y = 0; y < h; y++) {
        for (x = 0; x < w; x++) {
            const uint8_t *q = p + ((size_t)y * (size_t)w + (size_t)x) * 4u;
            if (q[0] > q[3] || q[1] > q[3] || q[2] > q[3])
                return 0;
        }
    }
    return 1;
}

int main(void)
{
    uint8_t *src, *dst, *got;
    int rc;

    expect(strcmp(nemo_scale_version(), "ntse-scale 1.0") == 0,
           "version string");

    /* ---- dimension bounds ---- */
    src = buf(4, 4);
    dst = buf(4, 4);
    rc = nemo_scale_rgba8(NULL, 4, 4, 0, dst, 4, 4, 0, NEMO_SCALE_NEAREST);
    expect(rc == NEMO_SCALE_ERR_NULL, "null src rejected");
    rc = nemo_scale_rgba8(src, 4, 4, 0, NULL, 4, 4, 0, NEMO_SCALE_NEAREST);
    expect(rc == NEMO_SCALE_ERR_NULL, "null dst rejected");
    rc = nemo_scale_rgba8(src, 0, 4, 0, dst, 4, 4, 0, NEMO_SCALE_NEAREST);
    expect(rc == NEMO_SCALE_ERR_DIM, "zero width rejected");
    rc = nemo_scale_rgba8(src, 4, -1, 0, dst, 4, 4, 0, NEMO_SCALE_NEAREST);
    expect(rc == NEMO_SCALE_ERR_DIM, "negative height rejected");
    rc = nemo_scale_rgba8(src, NEMO_SCALE_MAX_DIM + 1, 1, 0, dst, 4, 4, 0,
                          NEMO_SCALE_NEAREST);
    expect(rc == NEMO_SCALE_ERR_DIM, "over-max dimension rejected");
    /* 8192 x 8192 is over MAX_PIXELS (4096^2) */
    rc = nemo_scale_rgba8(src, 8192, 8192, 0, dst, 4, 4, 0, NEMO_SCALE_NEAREST);
    expect(rc == NEMO_SCALE_ERR_OVERFLOW, "over-max pixel count rejected");
    rc = nemo_scale_rgba8(src, 4, 4, 0, dst, 4, 4, 0, 99);
    expect(rc == NEMO_SCALE_ERR_FILTER, "unknown filter rejected");
    rc = nemo_scale_rgba8(src, 4, 4, 4, dst, 4, 4, 0, NEMO_SCALE_NEAREST);
    expect(rc == NEMO_SCALE_ERR_BUF, "too-small src stride rejected");
    rc = nemo_scale_rgba8(src, 4, 4, 0, dst, 4, 4, 8, NEMO_SCALE_NEAREST);
    expect(rc == NEMO_SCALE_ERR_BUF, "too-small dst stride rejected");
    expect(nemo_scale_workspace_size(0, 4, 4, 4, NEMO_SCALE_LANCZOS3) == 0,
           "workspace of a bad dim is 0");
    expect(nemo_scale_workspace_size(4, 4, 4, 4, NEMO_SCALE_LANCZOS3) == 0,
           "workspace of an identity scale is 0");
    expect(nemo_scale_workspace_size(4, 4, 8, 8, NEMO_SCALE_NEAREST) == 0,
           "workspace of nearest is 0");
    expect(nemo_scale_workspace_size(4, 4, 8, 8, NEMO_SCALE_LANCZOS3) ==
               (size_t)8 * 4u * 4u,
           "Lanczos workspace is dst_w * 4 * src_h");
    free(src);
    free(dst);

    /* ---- identity scale (every filter, exact) ---- */
    src = buf(7, 5);
    dst = buf(7, 5);
    fill_pattern(src, 7, 5);
    rc = nemo_scale_rgba8(src, 7, 5, 0, dst, 7, 5, 0, NEMO_SCALE_NEAREST);
    expect(rc == NEMO_SCALE_OK && equal_buf(src, dst, 7u * 5u * 4u),
           "nearest identity is exact");
    memset(dst, 0x5a, 7u * 5u * 4u);
    rc = nemo_scale_rgba8(src, 7, 5, 0, dst, 7, 5, 0, NEMO_SCALE_LANCZOS3);
    expect(rc == NEMO_SCALE_OK && equal_buf(src, dst, 7u * 5u * 4u),
           "Lanczos3 identity is exact");
    memset(dst, 0x5a, 7u * 5u * 4u);
    rc = nemo_scale_rgba8(src, 7, 5, 0, dst, 7, 5, 0, NEMO_SCALE_MITCHELL);
    expect(rc == NEMO_SCALE_OK && equal_buf(src, dst, 7u * 5u * 4u),
           "Mitchell identity is exact");
    free(src);
    free(dst);

    /* Identity with a wider stride. */
    {
        uint8_t srow[5 * 8]; /* 5 px, stride 8 (> 20 would be packed; use 24) */
        uint8_t drow[5 * 8];
        uint8_t *sbig = buf(5, 2);
        uint8_t *dbig = buf(5, 2);
        int y, x;
        /* Use stride 24 for width 5 (packed 20). */
        uint8_t sbuf[2 * 24];
        uint8_t dbuf[2 * 24];
        memset(sbuf, 0x11, sizeof sbuf);
        memset(dbuf, 0x22, sizeof dbuf);
        (void)srow; (void)drow; (void)sbig; (void)dbig;
        for (y = 0; y < 2; y++) {
            for (x = 0; x < 5; x++) {
                uint8_t *q = sbuf + y * 24 + x * 4;
                q[0] = (uint8_t)x; q[1] = (uint8_t)y; q[2] = 0; q[3] = 255;
            }
        }
        rc = nemo_scale_rgba8(sbuf, 5, 2, 24, dbuf, 5, 2, 24, NEMO_SCALE_NEAREST);
        expect(rc == NEMO_SCALE_OK, "strided identity returns OK");
        {
            int y, x, pix_ok = 1, pad_ok = 1;
            for (y = 0; y < 2; y++) {
                if (memcmp(sbuf + y * 24, dbuf + y * 24, 20) != 0)
                    pix_ok = 0;
                for (x = 20; x < 24; x++) {
                    if (dbuf[y * 24 + x] != 0x22)
                        pad_ok = 0;
                }
            }
            expect(pix_ok, "strided identity copies only the pixel bytes");
            expect(pad_ok, "strided identity does not write past the packed row");
        }
        free(sbig);
        free(dbig);
    }

    /* ---- integer 2x / 3x nearest exactness ---- */
    src = buf(3, 2);
    /* Distinct premul pixels. */
    {
        const uint8_t pix[3 * 2 * 4] = {
            10, 0, 0, 20,    0, 30, 0, 40,    0, 0, 50, 60,
            7, 8, 9, 10,     11, 12, 13, 14,  1, 2, 3, 4
        };
        memcpy(src, pix, sizeof pix);
    }
    dst = buf(6, 4);
    rc = nemo_scale_rgba8(src, 3, 2, 0, dst, 6, 4, 0, NEMO_SCALE_NEAREST);
    expect(rc == NEMO_SCALE_OK, "nearest 2x returns OK");
    {
        int x, y, ok = 1;
        for (y = 0; y < 4 && ok; y++) {
            for (x = 0; x < 6 && ok; x++) {
                const uint8_t *s = src + ((y / 2) * 3 + (x / 2)) * 4;
                const uint8_t *d = dst + (y * 6 + x) * 4;
                if (memcmp(s, d, 4) != 0)
                    ok = 0;
            }
        }
        expect(ok, "nearest 2x is exact pixel replication");
    }
    free(dst);
    dst = buf(9, 6);
    rc = nemo_scale_rgba8(src, 3, 2, 0, dst, 9, 6, 0, NEMO_SCALE_NEAREST);
    expect(rc == NEMO_SCALE_OK, "nearest 3x returns OK");
    {
        int x, y, ok = 1;
        for (y = 0; y < 6 && ok; y++) {
            for (x = 0; x < 9 && ok; x++) {
                const uint8_t *s = src + ((y / 3) * 3 + (x / 3)) * 4;
                const uint8_t *d = dst + (y * 9 + x) * 4;
                if (memcmp(s, d, 4) != 0)
                    ok = 0;
            }
        }
        expect(ok, "nearest 3x is exact pixel replication");
    }
    free(src);
    free(dst);

    /* ---- alpha edges ---- */
    src = buf(4, 1);
    dst = buf(8, 1);
    /* fully transparent, opaque red (premul), half-alpha blue, opaque white */
    src[0] = 0;   src[1] = 0;   src[2] = 0;   src[3] = 0;
    src[4] = 255; src[5] = 0;   src[6] = 0;   src[7] = 255;
    src[8] = 0;   src[9] = 0;   src[10] = 64; src[11] = 128;
    src[12] = 200; src[13] = 200; src[14] = 200; src[15] = 200;
    rc = nemo_scale_rgba8(src, 4, 1, 0, dst, 8, 1, 0, NEMO_SCALE_NEAREST);
    expect(rc == NEMO_SCALE_OK, "alpha-edge 2x nearest OK");
    expect(dst[0] == 0 && dst[3] == 0 && dst[4] == 0 && dst[7] == 0,
           "transparent stays fully transparent under nearest 2x");
    expect(dst[8] == 255 && dst[9] == 0 && dst[11] == 255 &&
           dst[12] == 255 && dst[15] == 255,
           "opaque red is replicated");
    expect(dst[8 + 8] == 0 && dst[8 + 10] == 64 && dst[8 + 11] == 128,
           "half-alpha premul blue is replicated exactly");
    expect(premul_valid(dst, 8, 1), "nearest 2x result stays premultiplied");

    got = buf(8, 2);
    rc = nemo_scale_rgba8(src, 4, 1, 0, got, 8, 2, 0, NEMO_SCALE_LANCZOS3);
    expect(rc == NEMO_SCALE_OK, "Lanczos3 on an alpha edge returns OK");
    expect(premul_valid(got, 8, 2),
           "Lanczos3 result is valid premultiplied (rgb <= a)");
    /* The far-left pixel is dominated by the transparent source; alpha
     * should stay small, and rgb must not exceed it (already checked). */
    expect(got[3] < 64, "Lanczos3 left edge stays mostly transparent");

    memset(got, 0, 8u * 2u * 4u);
    rc = nemo_scale_rgba8(src, 4, 1, 0, got, 8, 2, 0, NEMO_SCALE_MITCHELL);
    expect(rc == NEMO_SCALE_OK && premul_valid(got, 8, 2),
           "Mitchell on an alpha edge is valid premul");
    free(src);
    free(dst);
    free(got);

    /* Opaque 2x2 colour block, Lanczos/Mitchell upscale stays opaque
     * in the interior and premul-valid everywhere. */
    src = buf(2, 2);
    dst = buf(16, 16);
    src[0] = 40; src[1] = 80; src[2] = 120; src[3] = 255;
    src[4] = 40; src[5] = 80; src[6] = 120; src[7] = 255;
    src[8] = 40; src[9] = 80; src[10] = 120; src[11] = 255;
    src[12] = 40; src[13] = 80; src[14] = 120; src[15] = 255;
    rc = nemo_scale_rgba8(src, 2, 2, 0, dst, 16, 16, 0, NEMO_SCALE_LANCZOS3);
    expect(rc == NEMO_SCALE_OK && premul_valid(dst, 16, 16),
           "Lanczos3 upscale of an opaque field stays premul-valid");
    /* Centre pixel should stay close to the source colour. */
    {
        const uint8_t *c = dst + (8 * 16 + 8) * 4;
        expect(c[3] == 255 && abs((int)c[0] - 40) <= 2 &&
               abs((int)c[1] - 80) <= 2 && abs((int)c[2] - 120) <= 2,
               "Lanczos3 interior of a flat field matches the source");
    }
    rc = nemo_scale_rgba8(src, 2, 2, 0, dst, 16, 16, 0, NEMO_SCALE_MITCHELL);
    expect(rc == NEMO_SCALE_OK && premul_valid(dst, 16, 16),
           "Mitchell upscale of an opaque field stays premul-valid");
    free(src);
    free(dst);

    /* Workspace too small is rejected (no overrun). */
    src = buf(4, 4);
    dst = buf(8, 8);
    fill_pattern(src, 4, 4);
    {
        unsigned char tiny[8];
        rc = nemo_scale_rgba8_ex(src, 4, 4, 0, dst, 8, 8, 0,
                                 NEMO_SCALE_LANCZOS3, tiny, sizeof tiny);
        expect(rc == NEMO_SCALE_ERR_NOMEM, "short workspace rejected");
        rc = nemo_scale_rgba8_ex(src, 4, 4, 0, dst, 8, 8, 0,
                                 NEMO_SCALE_LANCZOS3, NULL, 0);
        expect(rc == NEMO_SCALE_ERR_NOMEM, "null workspace for Lanczos rejected");
    }
    /* And a correctly sized workspace works, no heap in _ex. */
    {
        size_t need = nemo_scale_workspace_size(4, 4, 8, 8, NEMO_SCALE_MITCHELL);
        void *work = malloc(need);
        expect(work != NULL && need > 0, "Mitchell 4->8 needs a workspace");
        rc = nemo_scale_rgba8_ex(src, 4, 4, 0, dst, 8, 8, 0,
                                 NEMO_SCALE_MITCHELL, work, need);
        expect(rc == NEMO_SCALE_OK && premul_valid(dst, 8, 8),
               "ex + caller workspace scales correctly");
        free(work);
    }
    free(src);
    free(dst);

    printf("RESULT: %s\n", fails ? "FAILURES" : "ALL PASS");
    return fails ? 1 : 0;
}
