/* test_sn_image.c - target-grid rule, resize invariants, and a real PNG decode. test_sn_image [PNG] */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "../src/sn_image.h"

static int fails, checks;
#define CHECK(c, ...) do { checks++; if (!(c)) { fails++; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static void plan(int w, int h, int budget, int ew, int eh)
{
    int wp, hp; sn_image_plan(w, h, 16, 2, 1024, budget, &wp, &hp);
    CHECK(wp == ew && hp == eh, "plan %dx%d budget %d: got %dx%d patches, want %dx%d", w, h, budget, wp, hp, ew, eh);
}

int main(int argc, char **argv)
{
    /* values worked out by hand from _compute_target_patches (python round = half to even) */
    plan(512, 512, 13312, 32, 32);      /* 32.5 -> 32 both ways, 1024 patches */
    plan(100, 100, 13312, 32, 32);      /* 7x7 -> up by sqrt(1024/49) = 32/7, so ceil(32.0) = 32 */
    plan(1920, 1080, 13312, 120, 68);   /* 8160 patches fit the budget */
    plan(1920, 1080, 1024, 42, 24);     /* the CPU cap: factor sqrt(1024/8160) */
    plan(100, 70, 13312, 38, 28);       /* 7x5 -> up by sqrt(1024/35)=5.41 -> ceil to 38x28 */
    /* constant plane stays constant under antialiased bicubic (weights are normalised), up and down */
    float *src = (float *)malloc(37 * 29 * 4), *dst = (float *)malloc(200 * 150 * 4);
    for (int i = 0; i < 37 * 29; i++) src[i] = 123.5f;
    sn_image_resize_plane(src, 37, 29, dst, 200, 150);
    double md = 0; for (int i = 0; i < 200 * 150; i++) md = fmax(md, fabs(dst[i] - 123.5));
    CHECK(md < 1e-3, "constant up: max dev %g", md);
    float *big = (float *)malloc(400 * 300 * 4), *sm = (float *)malloc(200 * 150 * 4);
    for (int i = 0; i < 400 * 300; i++) big[i] = 77.0f;
    sn_image_resize_plane(big, 400, 300, sm, 50, 40);
    md = 0; for (int i = 0; i < 50 * 40; i++) md = fmax(md, fabs(sm[i] - 77.0));
    CHECK(md < 1e-3, "constant down: max dev %g", md);
    /* a horizontal ramp keeps its mean and monotonicity when halved */
    for (int y = 0; y < 300; y++) for (int x = 0; x < 400; x++) big[y * 400 + x] = (float)x;
    sn_image_resize_plane(big, 400, 300, sm, 200, 150);
    int mono = 1; for (int x = 1; x < 200; x++) if (sm[x] <= sm[x - 1]) mono = 0;
    CHECK(mono && fabs(sm[100] - 200.0) < 1.5, "ramp halves cleanly (mid %g, want ~200)", sm[100]);
    if (argc > 1) {
        unsigned char *rgb; int w, h; char err[200];
        CHECK(sn_image_load(argv[1], &rgb, &w, &h, err, sizeof err) == 0 && w == 100 && h == 70, "png decodes to 100x70 (%s)", err);
        if (rgb) { CHECK(rgb[0] == 0 && rgb[1] == 0, "first pixel"); free(rgb); }
        CHECK(sn_image_load("no-such-file.png", &rgb, &w, &h, err, sizeof err) != 0, "a missing file is refused");
    }
    printf("%d checks, %d failed\n", checks, fails);
    return fails ? 1 : 0;
}


