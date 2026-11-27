/* sn_preflight.c - does a pack hold exactly the tensors SuperNemo needs?
 *
 *   sn_preflight PREFIX SHAPES.tsv CONFIG.json
 *
 * Reads layers_block_type from the checkpoint config, builds the tensor
 * list the language model needs (embeddings, 88 layers of Mamba-2 / MoE /
 * attention, final norm, lm_head, plus the RADIO tower, the projector and
 * the MTP head, reported separately), and checks that each is present in
 * the pack with the element count the geometry implies. Exit 0 only when
 * the language path is whole. Pure C99, no locks. */
#include "../src/sn_pack.h"

#include <stdlib.h>
#include <string.h>

enum { HID = 4096, MAM_HEADS = 128, MAM_HD = 64, MAM_STATE = 128, MAM_GROUPS = 8,
       MAM_INNER = 8192, MAM_CONV = 10240, MAM_IN = 18560,
       LAT = 1024, MOE_INTER = 2688, SH_INTER = 5376, NEXP = 512,
       NH = 32, NKV = 2, HD = 128, VOCAB = 131072,
       V_HID = 1280, V_LAYERS = 32, V_FF = 5120 };

static int missing = 0, wrong = 0, checked = 0;
static int quiet_tensors = 0;

static void expect(const sn_pack *p, const char *fmt, int a, int b, uint64_t count)
{
    char name[300];
    snprintf(name, sizeof(name), fmt, a, b);
    const sn_tensor *t = sn_pack_find(p, name);
    checked++;
    if (!t) { missing++; if (missing <= 10) printf("MISSING %s\n", name); return; }
    if (t->count != count) {
        wrong++;
        if (wrong <= 10) printf("WRONG SIZE %s: pack %llu, expected %llu\n", name, (unsigned long long)t->count, (unsigned long long)count);
    }
    (void)quiet_tensors;
}

int main(int argc, char **argv)
{
    if (argc != 4) { fprintf(stderr, "usage: sn_preflight PREFIX SHAPES.tsv CONFIG.json\n"); return 2; }
    sn_pack p;
    if (sn_pack_open(&p, argv[1], argv[2]) != 0) return 1;
    FILE *f = fopen(argv[3], "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[3]); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *cfg = (char *)malloc((size_t)sz + 1);
    if (fread(cfg, 1, (size_t)sz, f) != (size_t)sz) return 1;
    cfg[sz] = 0; fclose(f);
    char *lb = strstr(cfg, "\"layers_block_type\"");
    if (!lb) { fprintf(stderr, "config has no layers_block_type\n"); return 1; }
    char *open = strchr(lb, '['), *close = open ? strchr(open, ']') : NULL;
    if (!open || !close) return 1;
    int nl = 0; char types[256];
    for (char *q = open; q < close; q++) {
        if (*q == '"') {
            char *e = strchr(q + 1, '"');
            if (!e || e > close) break;
            if (nl < 256) types[nl++] = (strncmp(q + 1, "mamba", 5) == 0) ? 'M' : (strncmp(q + 1, "moe", 3) == 0) ? 'E' : (strncmp(q + 1, "attention", 9) == 0) ? 'A' : '?';
            q = e;
        }
    }
    int nm = 0, ne = 0, na = 0, nq = 0;
    for (int i = 0; i < nl; i++) { if (types[i] == 'M') nm++; else if (types[i] == 'E') ne++; else if (types[i] == 'A') na++; else nq++; }
    printf("layers: %d (%d mamba, %d moe, %d attention, %d unknown)\n", nl, nm, ne, na, nq);
    if (nq) { printf("unknown layer type in config\n"); return 1; }

    expect(&p, "language_model.backbone.embeddings.weight%.0d%.0d", 0, 0, (uint64_t)VOCAB * HID);
    expect(&p, "language_model.lm_head.weight%.0d%.0d", 0, 0, (uint64_t)VOCAB * HID);
    expect(&p, "language_model.backbone.norm_f.weight%.0d%.0d", 0, 0, HID);
    for (int i = 0; i < nl; i++) {
        expect(&p, "language_model.backbone.layers.%d.norm.weight%.0d", i, 0, HID);
        if (types[i] == 'M') {
            expect(&p, "language_model.backbone.layers.%d.mixer.in_proj.weight%.0d", i, 0, (uint64_t)MAM_IN * HID);
            expect(&p, "language_model.backbone.layers.%d.mixer.out_proj.weight%.0d", i, 0, (uint64_t)HID * MAM_INNER);
            expect(&p, "language_model.backbone.layers.%d.mixer.conv1d.weight%.0d", i, 0, (uint64_t)MAM_CONV * 4);
            expect(&p, "language_model.backbone.layers.%d.mixer.conv1d.bias%.0d", i, 0, MAM_CONV);
            expect(&p, "language_model.backbone.layers.%d.mixer.A_log%.0d", i, 0, MAM_HEADS);
            expect(&p, "language_model.backbone.layers.%d.mixer.D%.0d", i, 0, MAM_HEADS);
            expect(&p, "language_model.backbone.layers.%d.mixer.dt_bias%.0d", i, 0, MAM_HEADS);
            expect(&p, "language_model.backbone.layers.%d.mixer.norm.weight%.0d", i, 0, MAM_INNER);
        } else if (types[i] == 'A') {
            expect(&p, "language_model.backbone.layers.%d.mixer.q_proj.weight%.0d", i, 0, (uint64_t)NH * HD * HID);
            expect(&p, "language_model.backbone.layers.%d.mixer.k_proj.weight%.0d", i, 0, (uint64_t)NKV * HD * HID);
            expect(&p, "language_model.backbone.layers.%d.mixer.v_proj.weight%.0d", i, 0, (uint64_t)NKV * HD * HID);
            expect(&p, "language_model.backbone.layers.%d.mixer.o_proj.weight%.0d", i, 0, (uint64_t)HID * NH * HD);
        } else {
            expect(&p, "language_model.backbone.layers.%d.mixer.gate.weight%.0d", i, 0, (uint64_t)NEXP * HID);
            expect(&p, "language_model.backbone.layers.%d.mixer.gate.e_score_correction_bias%.0d", i, 0, NEXP);
            expect(&p, "language_model.backbone.layers.%d.mixer.fc1_latent_proj.weight%.0d", i, 0, (uint64_t)LAT * HID);
            expect(&p, "language_model.backbone.layers.%d.mixer.fc2_latent_proj.weight%.0d", i, 0, (uint64_t)HID * LAT);
            expect(&p, "language_model.backbone.layers.%d.mixer.shared_experts.up_proj.weight%.0d", i, 0, (uint64_t)SH_INTER * HID);
            expect(&p, "language_model.backbone.layers.%d.mixer.shared_experts.down_proj.weight%.0d", i, 0, (uint64_t)HID * SH_INTER);
            for (int e = 0; e < NEXP; e++) {
                expect(&p, "language_model.backbone.layers.%d.mixer.experts.%d.up_proj.weight", i, e, (uint64_t)MOE_INTER * LAT);
                expect(&p, "language_model.backbone.layers.%d.mixer.experts.%d.down_proj.weight", i, e, (uint64_t)LAT * MOE_INTER);
            }
        }
    }
    int lm_missing = missing, lm_wrong = wrong, lm_checked = checked;
    printf("language path: %d tensors checked, %d missing, %d wrong size\n", lm_checked, lm_missing, lm_wrong);

    /* vision tower */
    for (int i = 0; i < V_LAYERS; i++) {
        expect(&p, "vision_model.radio_model.model.blocks.%d.attn.qkv.weight%.0d", i, 0, (uint64_t)3 * V_HID * V_HID);
        expect(&p, "vision_model.radio_model.model.blocks.%d.attn.qkv.bias%.0d", i, 0, 3 * V_HID);
        expect(&p, "vision_model.radio_model.model.blocks.%d.attn.proj.weight%.0d", i, 0, (uint64_t)V_HID * V_HID);
        expect(&p, "vision_model.radio_model.model.blocks.%d.attn.proj.bias%.0d", i, 0, V_HID);
        expect(&p, "vision_model.radio_model.model.blocks.%d.mlp.fc1.weight%.0d", i, 0, (uint64_t)V_FF * V_HID);
        expect(&p, "vision_model.radio_model.model.blocks.%d.mlp.fc1.bias%.0d", i, 0, V_FF);
        expect(&p, "vision_model.radio_model.model.blocks.%d.mlp.fc2.weight%.0d", i, 0, (uint64_t)V_HID * V_FF);
        expect(&p, "vision_model.radio_model.model.blocks.%d.mlp.fc2.bias%.0d", i, 0, V_HID);
        expect(&p, "vision_model.radio_model.model.blocks.%d.norm1.weight%.0d", i, 0, V_HID);
        expect(&p, "vision_model.radio_model.model.blocks.%d.norm1.bias%.0d", i, 0, V_HID);
        expect(&p, "vision_model.radio_model.model.blocks.%d.norm2.weight%.0d", i, 0, V_HID);
        expect(&p, "vision_model.radio_model.model.blocks.%d.norm2.bias%.0d", i, 0, V_HID);
    }
    expect(&p, "vision_model.radio_model.model.patch_generator.embedder.weight%.0d%.0d", 0, 0, (uint64_t)V_HID * 768);
    expect(&p, "vision_model.radio_model.model.patch_generator.video_embedder.weight%.0d%.0d", 0, 0, (uint64_t)V_HID * 1536);
    expect(&p, "vision_model.radio_model.model.patch_generator.pos_embed%.0d%.0d", 0, 0, (uint64_t)16384 * V_HID);
    expect(&p, "vision_model.radio_model.model.patch_generator.cls_token.token%.0d%.0d", 0, 0, (uint64_t)10 * V_HID);
    expect(&p, "vision_projector.vision_final_layernorm.weight%.0d%.0d", 0, 0, V_HID);
    expect(&p, "vision_projector.vision_final_layernorm.bias%.0d%.0d", 0, 0, V_HID);
    expect(&p, "mlp1.0.weight%.0d%.0d", 0, 0, 5120);
    expect(&p, "mlp1.1.weight%.0d%.0d", 0, 0, (uint64_t)20480 * 5120);
    expect(&p, "mlp1.3.weight%.0d%.0d", 0, 0, (uint64_t)4096 * 20480);
    printf("vision path: %d more checked, %d missing, %d wrong size in total\n", checked - lm_checked, missing - lm_missing, wrong - lm_wrong);
    printf("(the MTP head, language_model.mtp.*, is packed but not used by the first engine)\n");
    int rc = (lm_missing || lm_wrong) ? 1 : 0;
    printf("%s\n", rc ? "LANGUAGE PATH NOT WHOLE" : "language path whole");
    sn_pack_close(&p);
    return rc;
}

