/* sn_mkvis.c - write a tiny random RADIO + mlp1 checkpoint (same tensor names as SuperNemo) for the vision test.
 *   sn_mkvis OUT_DIR
 * hidden 128, 2 heads (head dim 64), 2 layers, mlp 512, patch 16, position table 6x6 (max_img 96), 3 CLS + 7 registers,
 * 2x2 pixel shuffle, projector 512 -> 640 -> 256. Pure C99. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VH 128
#define NLAY 2
#define MLP 512
#define PS 16
#define MAXIMG 96
#define NPOS ((MAXIMG / PS) * (MAXIMG / PS))
#define NPFX 10
#define CAT (VH * 4)
#define PROJ 640
#define LLM 256

static unsigned rs = 777;
static float rnd(void) { rs = rs * 1664525u + 1013904223u; return ((float)((rs >> 8) & 0xffff) / 32768.0f - 1.0f); }
static uint16_t bf16(float v) { uint32_t b; memcpy(&b, &v, 4); uint32_t r = 0x7fff + ((b >> 16) & 1); return (uint16_t)((b + r) >> 16); }

typedef struct { char name[200]; int nd; uint64_t shape[3]; uint64_t cnt; float *data; } T;
static T *ts; static int nt, cap;
static void add(const char *fmt, int a, int nd, uint64_t s0, uint64_t s1, uint64_t s2, float scale, float offset)
{
    if (nt == cap) { cap = cap ? cap * 2 : 64; ts = (T *)realloc(ts, sizeof(T) * (size_t)cap); }
    T *t = &ts[nt++]; memset(t, 0, sizeof *t);
    snprintf(t->name, sizeof t->name, fmt, a);
    t->nd = nd; t->shape[0] = s0; t->shape[1] = s1; t->shape[2] = s2;
    t->cnt = s0 * (nd > 1 ? s1 : 1) * (nd > 2 ? s2 : 1);
    t->data = (float *)malloc(t->cnt * 4);
    for (uint64_t i = 0; i < t->cnt; i++) t->data[i] = offset + scale * rnd();
}

int main(int argc, char **argv)
{
    if (argc != 2) { fprintf(stderr, "usage: sn_mkvis OUT_DIR\n"); return 2; }
    const char *P = "vision_model.radio_model.model.";
    char fmt[256];
#define A(suffix, a, nd, s0, s1, s2, sc, off) do { snprintf(fmt, sizeof fmt, "%s%s", P, suffix); add(fmt, a, nd, s0, s1, s2, sc, off); } while (0)
    A("patch_generator.embedder.weight%.0d", 0, 2, VH, 3 * PS * PS, 0, 0.08f, 0);
    A("patch_generator.cls_token.token%.0d", 0, 2, NPFX, VH, 0, 0.5f, 0);
    A("patch_generator.pos_embed%.0d", 0, 3, 1, NPOS, VH, 0.5f, 0);
    for (int i = 0; i < NLAY; i++) {
        A("blocks.%d.norm1.weight", i, 1, VH, 0, 0, 0.1f, 1.0f);  A("blocks.%d.norm1.bias", i, 1, VH, 0, 0, 0.1f, 0);
        A("blocks.%d.attn.qkv.weight", i, 2, 3 * VH, VH, 0, 0.25f, 0); A("blocks.%d.attn.qkv.bias", i, 1, 3 * VH, 0, 0, 0.1f, 0);
        A("blocks.%d.attn.proj.weight", i, 2, VH, VH, 0, 0.2f, 0);  A("blocks.%d.attn.proj.bias", i, 1, VH, 0, 0, 0.1f, 0);
        A("blocks.%d.norm2.weight", i, 1, VH, 0, 0, 0.1f, 1.0f);  A("blocks.%d.norm2.bias", i, 1, VH, 0, 0, 0.1f, 0);
        A("blocks.%d.mlp.fc1.weight", i, 2, MLP, VH, 0, 0.2f, 0);  A("blocks.%d.mlp.fc1.bias", i, 1, MLP, 0, 0, 0.1f, 0);
        A("blocks.%d.mlp.fc2.weight", i, 2, VH, MLP, 0, 0.12f, 0); A("blocks.%d.mlp.fc2.bias", i, 1, VH, 0, 0, 0.1f, 0);
    }
    add("vision_projector.vision_final_layernorm.weight%.0d", 0, 1, VH, 0, 0, 0.1f, 1.0f);
    add("vision_projector.vision_final_layernorm.bias%.0d", 0, 1, VH, 0, 0, 0.1f, 0);
    add("mlp1.0.weight%.0d", 0, 1, CAT, 0, 0, 0.1f, 1.0f);
    add("mlp1.1.weight%.0d", 0, 2, PROJ, CAT, 0, 0.1f, 0);
    add("mlp1.3.weight%.0d", 0, 2, LLM, PROJ, 0, 0.1f, 0);
    char path[1024]; snprintf(path, sizeof path, "%s/model-00001-of-00001.safetensors", argv[1]);
    FILE *f = fopen(path, "wb"); if (!f) { fprintf(stderr, "cannot write %s\n", path); return 1; }
    size_t hc = 1 << 20; char *h = (char *)malloc(hc); size_t hl = 0; uint64_t off = 0;
    hl += (size_t)snprintf(h + hl, hc - hl, "{");
    for (int i = 0; i < nt; i++) {
        uint64_t bytes = ts[i].cnt * 2;
        hl += (size_t)snprintf(h + hl, hc - hl, "%s\"%s\":{\"dtype\":\"BF16\",\"shape\":[", i ? "," : "", ts[i].name);
        for (int d = 0; d < ts[i].nd; d++) hl += (size_t)snprintf(h + hl, hc - hl, "%s%llu", d ? "," : "", (unsigned long long)ts[i].shape[d]);
        hl += (size_t)snprintf(h + hl, hc - hl, "],\"data_offsets\":[%llu,%llu]}", (unsigned long long)off, (unsigned long long)(off + bytes));
        off += bytes;
    }
    hl += (size_t)snprintf(h + hl, hc - hl, "}");
    uint8_t lb[8]; for (int i = 0; i < 8; i++) lb[i] = (uint8_t)((uint64_t)hl >> (8 * i));
    fwrite(lb, 1, 8, f); fwrite(h, 1, hl, f);
    for (int i = 0; i < nt; i++) for (uint64_t k = 0; k < ts[i].cnt; k++) { uint16_t b = bf16(ts[i].data[k]); fwrite(&b, 2, 1, f); }
    fclose(f);
    snprintf(path, sizeof path, "%s/shapes.tsv", argv[1]);
    f = fopen(path, "wb"); fprintf(f, "name\tdtype\tshape\tcount\n");
    for (int i = 0; i < nt; i++) {
        fprintf(f, "%s\tBF16\t", ts[i].name);
        for (int d = 0; d < ts[i].nd; d++) fprintf(f, "%s%llu", d ? "x" : "", (unsigned long long)ts[i].shape[d]);
        fprintf(f, "\t%llu\n", (unsigned long long)ts[i].cnt);
    }
    fclose(f);
    snprintf(path, sizeof path, "%s/config.json", argv[1]);
    f = fopen(path, "wb");
    fprintf(f, "{\"downsample_ratio\":0.5,\"projector_hidden_size\":%d,\"llm_config\":{\"hidden_size\":%d},"
               "\"vision_config\":{\"hidden_size\":%d,\"layer_norm_eps\":1e-06,\"max_img_size\":%d,\"mlp_ratio\":%.1f,\"num_attention_heads\":2,"
               "\"num_cls_tokens\":3,\"num_hidden_layers\":%d,\"num_registers\":7,\"patch_size\":%d}}\n",
            PROJ, LLM, VH, MAXIMG, (double)MLP / VH, NLAY, PS);
    fclose(f);
    printf("tiny vision checkpoint: %d tensors\n", nt);
    return 0;
}
