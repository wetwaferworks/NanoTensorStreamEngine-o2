/* nemo_gpu.h - Little Nemo end-to-end GPU runtime (CUDA lane), pure C API.
 *
 * Weights: the NTSE .nq2 format-v3 PQ pack (d=4, K=256, block 256,
 * per-tensor books) for every 2-D layer projection, resident on the
 * GPU and dequantized on the fly inside the GEMV (no W is ever
 * materialized). Norms, conv, SSM parameters, embeddings and lm_head
 * come from the checkpoint directory (bf16): lm_head stays bf16 on the
 * GPU, the embedding row is read on the host per token.
 *
 * The CUDA driver API and NVRTC are loaded at run time (no CUDA SDK at
 * build time). One host thread drives everything: no mutexes, no
 * worker threads. */
#ifndef NEMO_GPU_H
#define NEMO_GPU_H

#include <stddef.h>
#include <stdint.h>

typedef struct nemo_gpu nemo_gpu;

typedef struct {
    char device[128];
    int cc_major, cc_minor;
    uint64_t vram_total, vram_free_at_open, vram_used_by_us;
    uint64_t bytes_per_token;  /* device weight bytes read per token */
    uint64_t head_bytes;       /* of which lm_head */
    uint64_t pack_bytes;       /* PQ index + frames + books on device */
    int layers, max_pos;
    int half_book;             /* 1: books held fp16 on the device (speed path default) */
    int exact;                 /* 1: NEMO_GPU_EXACT=1, bf16-faithful decode kernel (verification) */
    int head_int8;             /* 1: lm_head stored int8 per-row scaled (bf16 did not fit) */
    int full_ok;               /* 1: the full (bf16-faithful decode) kernel can run: EXPAND possible */
    int snap_ok;               /* 1: a step can be snapshotted and redone at full precision */
} nemo_gpu_info;

/* Open the CUDA lane and upload everything. ckpt_dir must hold the
 * tensors not in the pack (norms, conv, SSM params, embeddings,
 * lm_head). Returns 0, or -1 with a reason in why. */
int nemo_gpu_open(nemo_gpu **out, const char *ckpt_dir, const char *pack_path,
                  int layer_limit, int max_pos, char *why, size_t wl);
void nemo_gpu_close(nemo_gpu *g);
void nemo_gpu_info_get(const nemo_gpu *g, nemo_gpu_info *info);

/* One token through the stack. If logits is non-NULL the final norm and
 * lm_head run and [131072] logits are copied back. Returns 0, -1 on a
 * device error (the state is then invalid until nemo_gpu_reset). */
int nemo_gpu_step(nemo_gpu *g, int token, float *logits);
/* Final norm + lm_head on the current residual (after a step without
 * logits). */
int nemo_gpu_logits(nemo_gpu *g, float *logits);
/* Batched prefill (tokens in): run n prompt tokens through the stack
 * without logits, `chunk` tokens per layer pass (NEMO_GPU_PREFILL_CHUNK,
 * default 64; 0 or 1 = one token at a time). The resulting state is
 * bit-identical to n calls of nemo_gpu_step(.., NULL). Returns 0, -1 on
 * error (state invalid until nemo_gpu_reset), -2 if already invalid. */
int nemo_gpu_prefill(nemo_gpu *g, const int *ids, int n);
int nemo_gpu_prefill_chunk(const nemo_gpu *g);   /* 0: batched prefill off */
void nemo_gpu_reset(nemo_gpu *g);
int nemo_gpu_pos(const nemo_gpu *g);

/* Verification hook: when set, the residual stream entering each layer
 * (and the final one after the last layer, layer == layer_limit) is
 * copied to the host and handed to the hook. Synchronizing; tests only. */
void nemo_gpu_set_layer_hook(nemo_gpu *g,
                             void (*hook)(void *user, int layer, const float *x),
                             void *user);

/* ---- DTC / EXPAND ----
 * DTC (default): the fast kernel, books fp16 in shared memory, weights
 * dequantized in fp32 registers (dense tensor compute on the pack).
 * FULL (EXPAND): the bf16-faithful kernel, fp32 books, every weight
 * decoded and rounded to bf16 exactly like the unpacked reference
 * checkpoint. Both read the same pack; FULL is about 1.7x slower. */
int nemo_gpu_full_available(const nemo_gpu *g, int *per_step, char *why, size_t wl);
int nemo_gpu_set_full(nemo_gpu *g, int on);   /* -1 when FULL is unavailable */
int nemo_gpu_get_full(const nemo_gpu *g);
/* Copy the recurrent state (SSM + conv windows) and position aside, and
 * put them back: the step after a snapshot can be redone. */
int nemo_gpu_snapshot(nemo_gpu *g);
int nemo_gpu_restore(nemo_gpu *g);

/* ---- calibration probe ----
 * Opens the CUDA driver + NVRTC, compiles the kernels, runs the DTC and
 * FULL GEMV kernels on small synthetic PQ tensors and compares them
 * bit for bit with a C emulation of the same arithmetic, then times a
 * GEMV of in_proj size and a plain read of device memory. Releases the
 * device afterwards. Returns 0 if the lane is usable (r->pass), -1 with
 * why otherwise. */
typedef struct {
    char device[128];
    int cc_major, cc_minor, driver, nvrtc_major, nvrtc_minor;
    uint64_t vram_total, vram_free;
    int dtc_exact, full_exact;      /* 1: bit-identical to the C emulation */
    long dtc_mismatch, full_mismatch;
    double dtc_gwps, full_gwps;     /* billions of weights / s, in_proj-sized GEMV */
    double read_gbs;                /* device memory read GB/s */
    double est_tps;                 /* DTC tokens/s estimate for the 9B stack */
    int pass;
} nemo_gpu_probe;
int nemo_gpu_selftest(nemo_gpu_probe *r, char *why, size_t wl);
/* Driver version and device name only (no context): the calibration key. */
int nemo_gpu_identify(char *device, size_t dl, int *driver, int *nvrtc_major, int *nvrtc_minor);
/* Identity-check a format-v3 PQ pack (stream + book + manifest
 * digests, kind, trailing end line). No device. Returns 0, or -1
 * with a reason in why. */
int nemo_gpu_pack_check(const char *pack_path, char *why, size_t wl);

#endif
