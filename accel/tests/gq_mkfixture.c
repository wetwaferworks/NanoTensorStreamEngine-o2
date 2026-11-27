/* gq_mkfixture.c - synthetic safetensors shards for the GPU
 * quantizer tests. Pure C, no model data: every value comes from a
 * seeded generator, so the fixtures are reproducible anywhere.
 *
 * Co-implemented by Cortana at JD's direction.
 *
 * usage: gq_mkfixture OUT.safetensors SEED [normal|edge|bad]
 *   normal  ordinary weight shapes in BF16, F16 and F32, one tensor
 *           past 8192 blocks (reservoir sampling and chunked packing)
 *   edge    ties, duplicate blocks, grid values, zeros, constants,
 *           tiny and subnormal values, a tail that is not a whole
 *           block, heavy tails
 *   bad     values past the float16 frame range and a NaN, which
 *           the quantizer must refuse to publish
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static uint64_t rs;
static uint64_t rnd(void)
{
    uint64_t x = rs;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    rs = x;
    return x * 0x2545F4914F6CDD1Dull;
}
static double uni(void) { return (double)(rnd() >> 11) * (1.0 / 9007199254740992.0); }
static float gauss(void)
{
    double s = 0.0;
    for (int i = 0; i < 12; i++) s += uni();
    return (float)(s - 6.0);
}

enum { BF16, F16, F32 };
typedef struct {
    const char *name;
    int dtype;
    uint64_t rows, cols;
    int kind;
    float scale;
} Spec;

static uint16_t to_bf16(float f)
{
    uint32_t u; memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
    uint32_t r = ((u >> 16) & 1u) + 0x7fffu;
    return (uint16_t)((u + r) >> 16);
}
static uint16_t to_f16(float f)
{
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t e = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffffu;
    if (((x >> 23) & 0xff) == 0xff) return (uint16_t)(sign | 0x7c00u | (m ? 0x200u : 0));
    if (e >= 31) return (uint16_t)(sign | 0x7c00u);
    if (e <= 0) {
        if (e < -10) return (uint16_t)sign;
        m |= 0x800000u;
        uint32_t sh = (uint32_t)(14 - e);
        uint32_t h = m >> sh;
        uint32_t rem = m & ((1u << sh) - 1u), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (h & 1u))) h++;
        return (uint16_t)(sign | h);
    }
    uint32_t h = ((uint32_t)e << 10) | (m >> 13);
    uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h++;
    return (uint16_t)(sign | h);
}

/* kinds */
enum { K_GAUSS, K_HEAVY, K_ZERO, K_CONST, K_DUPROWS, K_GRID, K_TINY,
       K_SUBN, K_RAMP, K_SPARSE, K_HUGE, K_NAN };

static float value(int kind, float scale, uint64_t i, uint64_t cols, float *rowbuf)
{
    switch (kind) {
    case K_GAUSS: return gauss() * scale;
    case K_HEAVY: { float g = gauss(); return g * g * g * scale; }
    case K_ZERO: return 0.0f;
    case K_CONST: return scale;
    case K_DUPROWS: {   /* 256-wide blocks drawn from 6 patterns */
        uint64_t blk = i / 256, at = i % 256;
        (void)cols;
        return rowbuf[(blk * 2654435761u % 6) * 256 + at];
    }
    case K_GRID: return (float)((int)(rnd() % 7) - 3) * scale;
    case K_TINY: return gauss() * 1e-30f;
    case K_SUBN: return gauss() * 1e-41f;
    case K_RAMP: return (float)(i % 256) * scale;
    case K_SPARSE: return (rnd() % 16 == 0) ? gauss() * scale : 0.0f;
    case K_HUGE: return gauss() * 1e6f;
    default: return (i == 77) ? (0.0f / 0.0f) : gauss() * scale;
    }
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: gq_mkfixture OUT SEED [normal|edge|bad]\n"); return 1; }
    rs = strtoull(argv[2], NULL, 10) * 0x9E3779B97F4A7C15ull + 1u;
    const char *prof = argc > 3 ? argv[3] : "normal";
    static const Spec normal[] = {
        { "model.embed_tokens.weight", BF16, 1000, 256, K_GAUSS, 0.02f },
        { "model.layers.0.self_attn.q_proj.weight", F16, 512, 512, K_GAUSS, 0.03f },
        { "model.layers.0.mlp.up_proj.weight", BF16, 1024, 64, K_GAUSS, 0.02f },
        { "model.layers.0.mlp.down_proj.weight", F32, 768, 1024, K_HEAVY, 0.01f },
        { "model.layers.1.mlp.experts.gate_up_proj", BF16, 2048, 2048, K_GAUSS, 0.015f },
        { "model.layers.1.input_layernorm.weight", F32, 1, 3072, K_GAUSS, 1.0f },
        { "lm_head.weight", BF16, 300, 256, K_GAUSS, 0.02f },
    };
    static const Spec edge[] = {
        { "model.layers.0.zeros.weight", F32, 16, 256, K_ZERO, 0.0f },
        { "model.layers.0.const.weight", BF16, 10, 256, K_CONST, 0.5f },
        { "model.layers.0.duprows.weight", BF16, 600, 256, K_DUPROWS, 0.02f },
        { "model.layers.0.grid.weight", F16, 400, 256, K_GRID, 0.125f },
        { "model.layers.0.tiny.weight", F32, 64, 256, K_TINY, 1.0f },
        { "model.layers.0.subnormal.weight", F32, 32, 256, K_SUBN, 1.0f },
        { "model.layers.0.ramp.weight", F32, 50, 256, K_RAMP, 0.001f },
        { "model.layers.0.sparse.weight", BF16, 900, 256, K_SPARSE, 0.05f },
        { "model.layers.0.tail.weight", BF16, 1, 1000, K_GAUSS, 0.02f },
        { "model.layers.0.heavy.weight", F16, 1200, 256, K_HEAVY, 0.01f },
        { "model.layers.0.one_block.weight", F32, 1, 256, K_GAUSS, 0.1f },
    };
    static const Spec bad[] = {
        { "model.layers.0.ok.weight", BF16, 64, 256, K_GAUSS, 0.02f },
        { "model.layers.0.huge.weight", F32, 8, 256, K_HUGE, 1.0f },
        { "model.layers.0.nan.weight", F32, 8, 256, K_NAN, 0.02f },
    };
    const Spec *sp; int ns;
    if (strcmp(prof, "edge") == 0) { sp = edge; ns = (int)(sizeof edge / sizeof edge[0]); }
    else if (strcmp(prof, "bad") == 0) { sp = bad; ns = (int)(sizeof bad / sizeof bad[0]); }
    else { sp = normal; ns = (int)(sizeof normal / sizeof normal[0]); }

    char *hdr = (char *)malloc(65536);
    size_t hl = 0;
    uint64_t off = 0;
    hl += (size_t)sprintf(hdr + hl, "{\"__metadata__\":{\"format\":\"pt\"}");
    for (int t = 0; t < ns; t++) {
        uint64_t cnt = sp[t].rows * sp[t].cols;
        uint64_t esz = sp[t].dtype == F32 ? 4 : 2;
        hl += (size_t)sprintf(hdr + hl,
            ",\"%s\":{\"dtype\":\"%s\",\"shape\":[%llu,%llu],\"data_offsets\":[%llu,%llu]}",
            sp[t].name, sp[t].dtype == BF16 ? "BF16" : sp[t].dtype == F16 ? "F16" : "F32",
            (unsigned long long)sp[t].rows, (unsigned long long)sp[t].cols,
            (unsigned long long)off, (unsigned long long)(off + cnt * esz));
        off += cnt * esz;
    }
    hl += (size_t)sprintf(hdr + hl, "}");
    while (hl % 8) hdr[hl++] = ' ';
    FILE *f = fopen(argv[1], "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", argv[1]); return 1; }
    uint8_t lb[8];
    for (int i = 0; i < 8; i++) lb[i] = (uint8_t)((uint64_t)hl >> (8 * i));
    fwrite(lb, 1, 8, f);
    fwrite(hdr, 1, hl, f);
    float *rowbuf = (float *)malloc(6 * 256 * sizeof(float));
    for (int i = 0; i < 6 * 256; i++) rowbuf[i] = gauss() * 0.02f;
    for (int t = 0; t < ns; t++) {
        uint64_t cnt = sp[t].rows * sp[t].cols;
        for (uint64_t i = 0; i < cnt; i++) {
            float v = value(sp[t].kind, sp[t].scale, i, sp[t].cols, rowbuf);
            if (sp[t].dtype == F32) fwrite(&v, 4, 1, f);
            else {
                uint16_t h = sp[t].dtype == BF16 ? to_bf16(v) : to_f16(v);
                fwrite(&h, 2, 1, f);
            }
        }
    }
    if (fclose(f) != 0) { fprintf(stderr, "write failed\n"); return 1; }
    printf("%s: %d tensors, %llu data bytes (%s, seed %s)\n", argv[1], ns,
           (unsigned long long)off, prof, argv[2]);
    return 0;
}
