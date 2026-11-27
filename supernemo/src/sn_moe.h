/* sn_moe.h - SuperNemo latent MoE block math (Nemotron-H MoE), pure C99.
 *
 * One MoE mixer for one token, as the Nemotron 3.5 Super language model
 * defines it (config: 512 routed experts, top 22, 1 shared expert, latent
 * width 1024, relu2 experts, sigmoid routing with a selection bias,
 * normalised top-k weights, routed scaling 5.0):
 *
 *   s[e]   = sigmoid(gate_w[e] . x)                 router scores
 *   c[e]   = s[e] + e_score_correction_bias[e]      used to CHOOSE only
 *   top    = the k largest c[e] (ties: lower index first)
 *   w[i]   = s[top[i]]  (the unbiased score), / (sum + 1e-20) if norm_topk,
 *            then * routed_scaling_factor
 *   z      = fc1_latent_proj x                      hidden -> latent
 *   acc    = sum_i w[i] * down_i( relu2( up_i z ) ) experts live in latent space
 *   routed = fc2_latent_proj acc                    latent -> hidden
 *   shared = shared_down( relu2( shared_up x ) )    full hidden width
 *   y      = routed + shared
 *
 * STATUS: written from the published Nemotron-H MoE structure and the
 * checkpoint's tensor shapes (all consistent: gate 512x4096, up 2688x1024,
 * down 1024x2688, fc1/fc2 latent 1024x4096, shared 5376x4096). The upstream
 * reference module is not on this machine, so the router tie rules and the
 * 1e-20 epsilon are the part to confirm against a reference run.
 *
 * Weights arrive as dense row-major float32 [out][in]. Experts are fetched
 * through a callback so the caller can expand them from a pack on demand.
 * No locks, no globals.
 */
#ifndef SN_MOE_H
#define SN_MOE_H

typedef struct {
    int hidden;          /* 4096 */
    int latent;          /* 1024 */
    int inter;           /* 2688 expert intermediate width */
    int shared_inter;    /* 5376 */
    int n_experts;       /* 512 */
    int topk;            /* 22 */
    float scale;         /* 5.0 */
    int norm_topk;       /* 1 */
} sn_moe_cfg;

float sn_relu2(float v);
float sn_sigmoid(float v);

/* y[r] = sum_c W[r*cols + c] * x[c], fp32 accumulate. */
void sn_matvec(const float *W, int rows, int cols, const float *x, float *y);

/* Router: gate_w [E][hidden], bias [E]. Writes the chosen expert ids (best
 * first) and their final weights. Returns 0, or -1 on bad geometry. */
int sn_moe_route(const sn_moe_cfg *c, const float *gate_w, const float *bias,
                 const float *x, int *top_idx, float *top_w);

/* Expert fetch: set *up ([inter][latent]) and *down ([latent][inter]) for
 * expert e. The pointers need only stay valid until the next fetch call
 * from the same thread. Return 0, or nonzero to fail the step. */
typedef int (*sn_expert_fetch)(void *user, int e, const float **up,
                               const float **down);

/* Full block for one token. scratch layout is internal: the function
 * allocates and frees its own buffers. Returns 0 or a negative code. */
int sn_moe_forward(const sn_moe_cfg *c,
                   const float *gate_w, const float *bias,
                   const float *fc1_latent, const float *fc2_latent,
                   const float *shared_up, const float *shared_down,
                   sn_expert_fetch fetch, void *fetch_user,
                   const float *x, float *y,
                   int *top_idx_out /* optional, [topk] */);

#endif
