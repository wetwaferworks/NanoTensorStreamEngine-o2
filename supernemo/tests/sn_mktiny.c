/* sn_mktiny.c - write a tiny random SuperNemo-shaped checkpoint for tests.
 *   sn_mktiny OUT_DIR
 * Produces OUT_DIR/model-00001-of-00001.safetensors (bf16 + one f32), model.safetensors.index.json
 * and config.json (llm_config only), with the same tensor names as the real model and a
 * geometry small enough for a laptop test but still exercising: unaligned expert down rows
 * (384 columns = 1.5 blocks), grouped Mamba B/C, GQA attention, latent MoE. Pure C99. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HID 256
#define VOCAB 512
#define NL 5
static const char *TYPES[NL] = { "mamba", "moe", "attention", "mamba", "moe" };
#define HEADS 4
#define HDIM 64
#define STATE 64
#define GROUPS 2
#define INNER 256
#define CONV (INNER + 2 * GROUPS * STATE)
#define INPROJ (INNER + CONV + HEADS)
#define NH 4
#define NKV 2
#define AHD 64
#define NE 8
#define TOPK 3
#define LAT 256
#define INTER 384
#define SHARED 512

static unsigned rs = 20260;
static float rnd(void) { rs = rs * 1664525u + 1013904223u; return ((float)((rs >> 8) & 0xffff) / 32768.0f - 1.0f); }
static uint16_t bf16(float v) { uint32_t b; memcpy(&b, &v, 4); uint32_t r = 0x7fff + ((b >> 16) & 1); return (uint16_t)((b + r) >> 16); }

typedef struct { char name[200]; int f32; int nd; uint64_t shape[3]; uint64_t cnt; float *data; } T;
static T *ts; static int nt, cap;
static void add(const char *fmt, int a, int b, int f32, int nd, uint64_t s0, uint64_t s1, uint64_t s2, float scale, float offset)
{
    if (nt == cap) { cap = cap ? cap * 2 : 64; ts = (T *)realloc(ts, sizeof(T) * (size_t)cap); }
    T *t = &ts[nt++]; memset(t, 0, sizeof(*t));
    snprintf(t->name, sizeof(t->name), fmt, a, b);
    t->f32 = f32; t->nd = nd; t->shape[0] = s0; t->shape[1] = s1; t->shape[2] = s2;
    t->cnt = s0 * (nd > 1 ? s1 : 1) * (nd > 2 ? s2 : 1);
    t->data = (float *)malloc(t->cnt * 4);
    for (uint64_t i = 0; i < t->cnt; i++) t->data[i] = offset + scale * rnd();
}

int main(int argc, char **argv)
{
    if (argc != 2) { fprintf(stderr, "usage: sn_mktiny OUT_DIR\n"); return 2; }
    add("language_model.backbone.embeddings.weight%.0d%.0d", 0, 0, 0, 2, VOCAB, HID, 0, 0.5f, 0);
    add("language_model.lm_head.weight%.0d%.0d", 0, 0, 0, 2, VOCAB, HID, 0, 0.3f, 0);
    add("language_model.backbone.norm_f.weight%.0d", 0, 0, 0, 1, HID, 0, 0, 0.1f, 1.0f);
    for (int i = 0; i < NL; i++) {
        add("language_model.backbone.layers.%d.norm.weight%.0d", i, 0, 0, 1, HID, 0, 0, 0.1f, 1.0f);
        if (!strcmp(TYPES[i], "mamba")) {
            add("language_model.backbone.layers.%d.mixer.in_proj.weight%.0d", i, 0, 0, 2, INPROJ, HID, 0, 0.12f, 0);
            add("language_model.backbone.layers.%d.mixer.out_proj.weight%.0d", i, 0, 0, 2, HID, INNER, 0, 0.12f, 0);
            add("language_model.backbone.layers.%d.mixer.conv1d.weight%.0d", i, 0, 0, 3, CONV, 1, 4, 0.4f, 0);
            add("language_model.backbone.layers.%d.mixer.conv1d.bias%.0d", i, 0, 0, 1, CONV, 0, 0, 0.05f, 0);
            add("language_model.backbone.layers.%d.mixer.A_log%.0d", i, 0, 0, 1, HEADS, 0, 0, 0.5f, 1.5f);
            add("language_model.backbone.layers.%d.mixer.D%.0d", i, 0, 0, 1, HEADS, 0, 0, 0.2f, 1.0f);
            add("language_model.backbone.layers.%d.mixer.dt_bias%.0d", i, 0, 0, 1, HEADS, 0, 0, 0.5f, -2.0f);
            add("language_model.backbone.layers.%d.mixer.norm.weight%.0d", i, 0, 0, 1, INNER, 0, 0, 0.1f, 1.0f);
        } else if (!strcmp(TYPES[i], "attention")) {
            add("language_model.backbone.layers.%d.mixer.q_proj.weight%.0d", i, 0, 0, 2, NH * AHD, HID, 0, 0.12f, 0);
            add("language_model.backbone.layers.%d.mixer.k_proj.weight%.0d", i, 0, 0, 2, NKV * AHD, HID, 0, 0.12f, 0);
            add("language_model.backbone.layers.%d.mixer.v_proj.weight%.0d", i, 0, 0, 2, NKV * AHD, HID, 0, 0.12f, 0);
            add("language_model.backbone.layers.%d.mixer.o_proj.weight%.0d", i, 0, 0, 2, HID, NH * AHD, 0, 0.12f, 0);
        } else {
            add("language_model.backbone.layers.%d.mixer.gate.weight%.0d", i, 0, 0, 2, NE, HID, 0, 0.3f, 0);
            add("language_model.backbone.layers.%d.mixer.gate.e_score_correction_bias%.0d", i, 0, 1, 1, NE, 0, 0, 0.05f, 0);
            add("language_model.backbone.layers.%d.mixer.fc1_latent_proj.weight%.0d", i, 0, 0, 2, LAT, HID, 0, 0.12f, 0);
            add("language_model.backbone.layers.%d.mixer.fc2_latent_proj.weight%.0d", i, 0, 0, 2, HID, LAT, 0, 0.12f, 0);
            add("language_model.backbone.layers.%d.mixer.shared_experts.up_proj.weight%.0d", i, 0, 0, 2, SHARED, HID, 0, 0.12f, 0);
            add("language_model.backbone.layers.%d.mixer.shared_experts.down_proj.weight%.0d", i, 0, 0, 2, HID, SHARED, 0, 0.12f, 0);
            for (int e = 0; e < NE; e++) {
                add("language_model.backbone.layers.%d.mixer.experts.%d.up_proj.weight", i, e, 0, 2, INTER, LAT, 0, 0.12f, 0);
                add("language_model.backbone.layers.%d.mixer.experts.%d.down_proj.weight", i, e, 0, 2, LAT, INTER, 0, 0.12f, 0);
            }
        }
    }
    /* header */
    char path[1024]; snprintf(path, sizeof(path), "%s/model-00001-of-00001.safetensors", argv[1]);
    FILE *f = fopen(path, "wb"); if (!f) { fprintf(stderr, "cannot write %s\n", path); return 1; }
    size_t hc = 1 << 20; char *h = (char *)malloc(hc); size_t hl = 0; uint64_t off = 0;
    hl += (size_t)snprintf(h + hl, hc - hl, "{");
    for (int i = 0; i < nt; i++) {
        uint64_t bytes = ts[i].cnt * (ts[i].f32 ? 4 : 2);
        hl += (size_t)snprintf(h + hl, hc - hl, "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[", i ? "," : "", ts[i].name, ts[i].f32 ? "F32" : "BF16");
        for (int d = 0; d < ts[i].nd; d++) hl += (size_t)snprintf(h + hl, hc - hl, "%s%llu", d ? "," : "", (unsigned long long)ts[i].shape[d]);
        hl += (size_t)snprintf(h + hl, hc - hl, "],\"data_offsets\":[%llu,%llu]}", (unsigned long long)off, (unsigned long long)(off + bytes));
        off += bytes;
    }
    hl += (size_t)snprintf(h + hl, hc - hl, "}");
    uint8_t lb[8]; for (int i = 0; i < 8; i++) lb[i] = (uint8_t)((uint64_t)hl >> (8 * i));
    fwrite(lb, 1, 8, f); fwrite(h, 1, hl, f);
    for (int i = 0; i < nt; i++) {
        if (ts[i].f32) fwrite(ts[i].data, 4, ts[i].cnt, f);
        else { for (uint64_t k = 0; k < ts[i].cnt; k++) { uint16_t b = bf16(ts[i].data[k]); fwrite(&b, 2, 1, f); } }
    }
    fclose(f);
    /* index json + shapes sidecar */
    snprintf(path, sizeof(path), "%s/model.safetensors.index.json", argv[1]);
    f = fopen(path, "wb"); fprintf(f, "{\"weight_map\":{");
    for (int i = 0; i < nt; i++) fprintf(f, "%s\"%s\":\"model-00001-of-00001.safetensors\"", i ? "," : "", ts[i].name);
    fprintf(f, "}}"); fclose(f);
    snprintf(path, sizeof(path), "%s/shapes.tsv", argv[1]);
    f = fopen(path, "wb"); fprintf(f, "name\tdtype\tshape\tcount\n");
    for (int i = 0; i < nt; i++) {
        fprintf(f, "%s\t%s\t", ts[i].name, ts[i].f32 ? "F32" : "BF16");
        for (int d = 0; d < ts[i].nd; d++) fprintf(f, "%s%llu", d ? "x" : "", (unsigned long long)ts[i].shape[d]);
        fprintf(f, "\t%llu\n", (unsigned long long)ts[i].cnt);
    }
    fclose(f);
    snprintf(path, sizeof(path), "%s/config.json", argv[1]);
    f = fopen(path, "wb");
    fprintf(f, "{\"llm_config\":{\"hidden_size\":%d,\"vocab_size\":%d,\"mamba_num_heads\":%d,\"mamba_head_dim\":%d,\"ssm_state_size\":%d,\"n_groups\":%d,"
               "\"num_attention_heads\":%d,\"num_key_value_heads\":%d,\"head_dim\":%d,\"n_routed_experts\":%d,\"num_experts_per_tok\":%d,"
               "\"moe_intermediate_size\":%d,\"moe_latent_size\":%d,\"moe_shared_expert_intermediate_size\":%d,\"routed_scaling_factor\":2.5,"
               "\"norm_topk_prob\":true,\"layer_norm_epsilon\":1e-05,\"layers_block_type\":[", HID, VOCAB, HEADS, HDIM, STATE, GROUPS, NH, NKV, AHD, NE, TOPK, INTER, LAT, SHARED);
    for (int i = 0; i < NL; i++) fprintf(f, "%s\"%s\"", i ? "," : "", TYPES[i]);
    fprintf(f, "]}}\n"); fclose(f);
    printf("tiny checkpoint: %d tensors\n", nt);
    return 0;
}
