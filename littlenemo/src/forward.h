/* forward.h - Little Nemo full forward pass (Nemotron-Nano-9B-v2), pure C.
 * Assembles the verified kernels (mamba.c, attn.c, nn.c) over the loader:
 * embed -> 56 blocks of RMSNorm -> mixer -> residual -> final norm ->
 * lm_head. Block pattern from the checkpoint config:
 * M = Mamba-2, - = MLP, * = attention.
 *
 * Weight path: tensors are fetched per use as raw bf16 (nemo_load_bf16)
 * and freed after the layer runs, so the raw model's resident footprint
 * is one layer's weights plus state, not 17.8 GB. Fetching is the simple
 * correct path; caching and the packed (NanoQuant) weight source plug
 * into the same fetch points later.
 *
 * Documented precision supersets vs the NVIDIA reference: the residual
 * stream is carried in fp32 (the reference carries bf16), and attention
 * accumulates in double. The PCHIE battery measures what, if anything,
 * those change; they are deliberate and recorded, not accidents.
 */
#ifndef NEMO_FORWARD_H
#define NEMO_FORWARD_H

#include <stddef.h>

#include "attn.h"
#include "loader.h"
#include "mamba.h"

#define NEMO_HIDDEN 4480
#define NEMO_LAYERS 56
#define NEMO_VOCAB 131072

typedef struct {
    const nemo_checkpoint *cp;
    int layer_limit; /* 0 = all 56; a smaller value runs a truncated
                        stack (plumbing tests only, not a model output) */
    int max_pos;
    nemo_mamba_state mamba[NEMO_LAYERS]; /* live only on M layers */
    nemo_attn_cache attn[NEMO_LAYERS];   /* live only on * layers */
    float *x, *h, *y;                    /* [hidden] streams */
    float *proj;                         /* [22656] in_proj output */
    float *inner;                        /* [10240] mamba output */
    float *mid;                          /* [15680] mlp hidden */
    float *qkv;                          /* [5120] attention q */
    /* Optional progress hook, called as each layer is entered during
     * a step (the engine's LAYER telemetry rides on it). NULL in tests. */
    void (*on_layer)(void *user, int layer, char type);
    void *on_layer_user;
    /* State coherence (Gizmo review, 2026-11-24): 1 while no partially
     * committed step is outstanding. A step that fails after its
     * preflight clears this; step entry then refuses with -2 until
     * nemo_model_reset restores it. Preparation can be concurrent;
     * state commitment must be coherent. */
    int state_valid;
} nemo_model;

/* layer_limit 0 means the full stack. Returns 0 on success. */
int nemo_model_init(nemo_model *m, const nemo_checkpoint *cp,
                    int layer_limit, int max_pos);
void nemo_model_free(nemo_model *m);
/* Zero all recurrent state and KV caches (new conversation). */
void nemo_model_reset(nemo_model *m);

/* Run one token through the stack (up to layer_limit), leaving the
 * pre-final-norm hidden state in hidden_out [hidden].
 * Returns 0 on success; -1 on a preparation failure (the step's
 * tensors were preflighted before any state moved, so nothing was
 * committed) or on a mid-step failure (state_valid is then 0);
 * -2 when the state is invalid from an earlier failure and a reset
 * is required before any further step. */
int nemo_model_step_hidden(nemo_model *m, int token, float *hidden_out);

/* Full step: hidden state -> final norm -> lm_head logits [vocab].
 * The head tensors are preflighted before the stack runs, so a
 * missing lm_head refuses the step before the token commits.
 * Same return conventions as nemo_model_step_hidden. */
int nemo_model_step(nemo_model *m, int token, float *logits_out);

/* Validate the whole model geometry against the checkpoint index
 * before inference begins (Gizmo review, 2026-11-24): every tensor
 * the steps will fetch, across layers 0..layer_limit-1 (0 = all 56)
 * plus embeddings, and plus the final norm and lm_head when
 * need_head is set, checked for presence, shape, and dtype class.
 * Returns 0 when the geometry is whole; on failure returns -1 and,
 * when bad/code are given, names the first offending tensor and the
 * failure class: 1 missing, 2 wrong shape, 3 wrong dtype. */
int nemo_validate_model(const nemo_checkpoint *cp, int layer_limit,
                        int need_head, char *bad, size_t badn, int *code);

/* Block type of layer i: 'M', '-', or '*', from the config pattern. */
char nemo_layer_type(int i);

#endif
