/* sn_model.h - SuperNemo language model: geometry-driven Nemotron-H hybrid
 * (Mamba-2 / latent MoE / attention) decoded from an NQ2 pack. Pure C99.
 *
 * Nothing is hard-coded to 4096 or 88: the geometry comes from the
 * checkpoint's config.json (llm_config), and every tensor the stack needs
 * is validated against it at open. Weights are never rebuilt into matrices:
 * the fused kernels in sn_pack.h compute the matrix-vector products straight
 * from the packed form, on the Little Nemo zero-mutex pool (rows per lane for
 * dense mixers, experts per lane for the MoE). Results are bit-identical for
 * any lane count.
 *
 * Direction by JD. Converter by Elfie. Co-implementation by Cortana, built
 * with Grok Bot by SpaceX.
 */
#ifndef SN_MODEL_H
#define SN_MODEL_H

#include <stdint.h>
#include <stdio.h>
#include "sn_pack.h"
#include "sn_pqk.h"
#include "sn_cuda.h"
#include "../../littlenemo/src/pool.h"

#define SN_MAX_LAYERS 256

typedef struct {
    int hidden, vocab, n_layers;
    char types[SN_MAX_LAYERS];       /* 'M' mamba, 'E' moe, 'A' attention */
    int m_heads, m_head_dim, m_state, m_groups, m_inner, m_conv_dim, m_in_proj;
    int a_heads, a_kv, a_hd;
    int moe_experts, moe_topk, moe_latent, moe_inter, moe_shared;
    float moe_scale; int moe_norm_topk;
    float eps;
    int mtp;                         /* the checkpoint carries one MTP module: an attention block then a MoE block */
} sn_geom;

/* Fill g from config.json (llm_config). Returns 0, or -1 with a reason on stderr. */
int sn_geom_load(sn_geom *g, const char *config_json);

typedef struct sn_model sn_model;

/* Open a pack and validate every tensor of the language path. nthreads 0 = all
 * logical cores. Returns NULL on failure (reason on stderr). */
sn_model *sn_model_open(const char *pack_prefix, const char *shapes_tsv,
                        const char *config_json, int max_pos, int nthreads);
void sn_model_close(sn_model *m);
const sn_geom *sn_model_geom(const sn_model *m);
int sn_model_threads(const sn_model *m);

/* New conversation: zero all recurrent state and KV caches. */
void sn_model_reset(sn_model *m);

/* One token through the stack, then final norm and lm_head. logits has
 * vocab floats; may be NULL to advance state only (prompt prefill).
 * Returns 0; -1 on failure (state then invalid until reset); -2 if the
 * state is already invalid; -3 if the context is full. */
int sn_model_step(sn_model *m, int token, float *logits);
int sn_model_pos(const sn_model *m);
/* Batched prefill of n tokens (layer-major, 128 per chunk): every weight matrix is expanded once per chunk and each
 * routed expert once for all the tokens that chose it, instead of once per token. embed_row (may be NULL) maps a position
 * to a row of embeds (vision tokens), -1 for a plain token. logits (may be NULL) is filled for the LAST token.
 * Numerically close to n sn_model_step calls, not bit-identical (dense float32 products instead of the fused kernel). */
int sn_model_prefill(sn_model *m, const int *tokens, const int *embed_row, const float *embeds, int n, float *logits);
/* Same as sn_model_step, but the input is an embedding vector of hidden floats (vision tokens). */
int sn_model_step_embed(sn_model *m, const float *embed, float *logits);
/* For the vision tower: it reads the same pack and runs on the same zero-mutex pool. */
const sn_pack *sn_model_pack(const sn_model *m);
/* Optional CUDA lane: returns the number of tensors made resident (0 is fine), or -1 with a reason in why (CPU only then). */
int sn_model_cuda(sn_model *m, long long budget_bytes, char *why, int whylen);
const char *sn_model_accel(const sn_model *m);    /* "cuda" or "cpu" */
long long sn_model_gpu_bytes(const sn_model *m);
int sn_model_gpu_count(const sn_model *m);
const char *sn_model_gpu_name(const sn_model *m);
nemo_pool *sn_model_pool(sn_model *m);


/* Dense Tensor Compute: 0 = off (direct fused kernels), 1 = auto (default: DTC where it pays), 2 = force where the
 * geometry allows. DTC equals direct to float noise (~1e-6 relative), not bit for bit. */
void sn_model_set_dtc(sn_model *m, int mode);
int sn_model_dtc(const sn_model *m);

/* n (<= SN_NV) tokens in one fused pass over the packed data: every token's logits (n * vocab floats) are bit-identical to
 * n separate sn_model_step calls. snap = keep Mamba snapshots so the tail can be dropped with sn_model_rollback
 * (base = the position before the call, keep = how many of the n tokens stay). */
int sn_model_step_n(sn_model *m, const int *tokens, int n, float *logits_all, int snap);
int sn_model_rollback(sn_model *m, int base, int keep, int n);

/* Speculative decoding with the checkpoint's MTP head (greedy only). sn_model_mtp_enable(m, 1) fails (-1) when the pack has
 * no complete head. One cycle consumes cur (the token already chosen from the last logits) plus the accepted drafts,
 * returns how many drafts were accepted (they are in acc[]), and leaves logits_out exactly as plain stepping would. The
 * output is token-identical to plain greedy stepping; only the speed changes. */
int sn_model_has_mtp(const sn_model *m);
int sn_model_mtp_on(const sn_model *m);
int sn_model_mtp_enable(sn_model *m, int on);
int sn_model_spec_cycle(sn_model *m, int cur, int kmax, int stop_a, int stop_b, int *acc, float *logits_out);
int sn_model_spec_verify(sn_model *m, int cur, const int *d, int kmax, int stop_a, int stop_b, int *acc, float *logits_out);
void sn_model_set_probe(sn_model *m, float *buf);   /* tests: the last token's hidden after each layer lands in buf[layer * hidden] */
void sn_model_spec_stats(const sn_model *m, long *cycles, long *drafted, long *accepted);
/* Optional profile, on when the environment has SN_PROF=1 (measurement only, results unchanged): time per kind of work,
 * router predictability per MoE layer, GPU expert stream stage times, MTP draft hits by position. tokname may be NULL. */
void sn_model_prof_report(sn_model *m, FILE *f, void (*tokname)(void *u, int tok, char *buf, int len), void *u);
/* Hot-expert cache (see sn_model.c). ram_bytes / vram_bytes of the most-used experts stay resident; 0 = off. The VRAM tier
 * needs the CUDA lane (sn_model_cuda first) and DTC. why gets a reason when a tier is left off. Returns 0, or -1 on failure. */
typedef struct {
    uint64_t hits_ram, hits_gpu, misses, admits, evicts, gpu_puts, gpu_drops;
    uint64_t ram_bytes, ram_budget, gpu_bytes;
    int ram_experts, gpu_experts;
} sn_cache_stats;
int sn_model_cache(sn_model *m, long long ram_bytes, long long vram_bytes, char *why, int whylen);
void sn_model_cache_stats(const sn_model *m, sn_cache_stats *s);

/* GPU expert stream (JD's standing CUDA design, 2026-11-27; needs sn_model_cuda first and DTC on). Every routed expert goes
 * to the card compressed, is expanded and multiplied there, and its VRAM slot is reused for the next token; the RAM expert
 * cache keeps feeding it packed copies. Slots are sized from the hardware calibration. Bit-identical to the CPU lanes, so
 * the output does not change. Replaces the VRAM tier of sn_model_cache when on. 0, or -1 with a reason (CPU lanes then). */
int sn_model_gpu_experts(sn_model *m, int on, char *why, int whylen);
int sn_model_gpu_experts_on(const sn_model *m);
void sn_model_gpu_experts_info(const sn_model *m, sn_xs_info *out);

/* Optional telemetry: called per layer entered and per MoE routing. */
typedef void (*sn_layer_cb)(void *user, int layer, char type);
typedef void (*sn_moe_cb)(void *user, int layer, const int *experts, int k);
void sn_model_set_hooks(sn_model *m, sn_layer_cb on_layer, sn_moe_cb on_moe, void *user);

#endif

