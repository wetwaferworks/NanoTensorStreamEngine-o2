/* sn_vision.h - SuperNemo vision: the RADIO ViT-H tower and the mlp1 projector, from the NQ2 pack.
 * Pure C99, zero mutexes (Little Nemo pool), AVX2/FMA kernels picked at run time with a scalar fallback.
 *
 * Follows the checkpoint's own modeling_radio.py and modeling_nemotron_h_omni.py:
 *   pixels -> 16x16 patches -> linear embedder + bilinear-resampled position embedding (CPE)
 *   -> [3 CLS + 7 register tokens | patches] -> 32 x (LN, fused-qkv attention, LN, GELU MLP)
 *   -> drop the 10 prefix tokens -> final LayerNorm -> 2x2 pixel shuffle (ps v2)
 *   -> mlp1: RMSNorm(5120) -> linear 20480 -> ReLU^2 -> linear 4096
 * The result is one LLM-space embedding per 2x2 patch group, to be fed in place of <image> tokens.
 * Weights stay packed on disk; each weight matrix is expanded one slab of rows at a time per lane.
 *
 * Direction by JD. Co-implementation by Cortana, built with Grok Bot by SpaceX. */
#ifndef SN_VISION_H
#define SN_VISION_H

#include "sn_pack.h"
#include "../../littlenemo/src/pool.h"

typedef struct {
    int hidden, layers, heads, mlp, ncls, nreg, patch, max_img;
    float eps;
    int llm_hidden, proj_hidden, shuffle;     /* shuffle = 1 / downsample_ratio */
} sn_vgeom;

/* Read vision_config (and the projector keys) from the checkpoint's config.json. 0 or -1 with a reason. */
int sn_vgeom_load(sn_vgeom *g, const char *config_json, int llm_hidden);

typedef struct sn_vision sn_vision;

/* Validates every vision tensor by name and size. pool is borrowed (the model's); nthreads = its lanes. */
sn_vision *sn_vision_open(const sn_pack *pack, nemo_pool *pool, int nthreads, const sn_vgeom *g);
void sn_vision_close(sn_vision *v);
const sn_vgeom *sn_vision_geom(const sn_vision *v);

/* pix: planar float32 [3][H][W], already /255 and normalised with the checkpoint mean/std.
 * H and W must be multiples of patch*shuffle. out holds (H/patch/shuffle)*(W/patch/shuffle)*llm_hidden floats.
 * Returns 0; -1 on a read failure; -2 on bad dimensions. */
int sn_vision_embed(sn_vision *v, const float *pix, int H, int W, float *out);

/* Exposed for tests: out[n][r] = act(bias[r] + sum_k X[n][k] * W[r][k]) with W read from the pack.
 * act: 0 none, 1 GELU (erf), 2 ReLU squared. bias may be NULL. */
int sn_vision_linear(sn_vision *v, const sn_tensor *t, const float *bias, const float *X, int N, int K, int R, float *Y, int act);

#endif
