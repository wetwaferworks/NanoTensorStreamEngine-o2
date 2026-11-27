/* sn_cuda.h - CUDA lane for the SuperNemo dense matvecs and the lm_head (pure C host code, no toolkit headers).
 *
 * Loads nvcuda.dll (the driver) and NVRTC at run time, compiles two small kernels for the installed GPU, and keeps
 * as many tensors as the VRAM budget allows resident in their PACKED form: a PQ4 tensor is its index bytes, its
 * fp16 minimum/scale per block and its 256 KB book; a raw tensor is its fp16 weights. The kernels read straight
 * from that form (the same arithmetic as sn_pq_rows, in float), so a resident weight is never rebuilt either.
 * If the driver, NVRTC or a GPU is missing, sn_cuda_open returns NULL with a reason and the CPU lane runs alone.
 * No toolkit is needed to BUILD this file; NVRTC (CUDA toolkit bin) is needed to RUN it.
 *
 * Direction by JD. Co-implementation by Cortana, built with Grok Bot by SpaceX. */
#ifndef SN_CUDA_H
#define SN_CUDA_H

#include "sn_pack.h"
#include "sn_pqk.h"

typedef struct sn_cuda sn_cuda;

/* budget_bytes: VRAM the lane may fill (it also stops if the device reports less free). */
sn_cuda *sn_cuda_open(long long budget_bytes, int pack_tensors, char *why, int whylen);
void sn_cuda_close(sn_cuda *c);
const char *sn_cuda_name(const sn_cuda *c);
long long sn_cuda_used(const sn_cuda *c);
int sn_cuda_count(const sn_cuda *c);

/* Make t resident. 0 = resident now (or already); -1 = no room or a read/copy failure (CPU keeps it). */
int sn_cuda_load(sn_cuda *c, const sn_pack *p, sn_rd *rd, const sn_tensor *t);
int sn_cuda_resident(const sn_cuda *c, const sn_pack *p, const sn_tensor *t);

/* y[0..rows) = W x for a resident [rows][cols] tensor. 0, or -1 (caller falls back to the CPU lane). */
int sn_cuda_matvec(sn_cuda *c, const sn_pack *p, const sn_tensor *t, int cols, int rows, const float *x, float *y);

/* Dense Tensor Compute on a resident PQ tensor: builds the 256-entry tables on the device, then one thread per row looks
 * them up. Compiled without fused multiply-add and written operation for operation like the CPU DTC kernel in sn_pqk.c, so the
 * result is BIT-IDENTICAL to sn_pqk_rows with tables (that is what makes the hot-expert GPU tier invisible in the output).
 * P is the double prefix sum of x (cols + 1). 0, or -1 (caller falls back to the CPU lane). */
int sn_cuda_dtc_rows(sn_cuda *c, const sn_pack *p, const sn_tensor *t, int cols, int rows, const float *x, const double *P, float *y);

/* Hot-expert tier: packed tensors copied up from a RAM-resident copy, under their own VRAM budget (separate from the dense
 * tensors' budget given to sn_cuda_open). Resident entries are used through sn_cuda_dtc_rows. */
void sn_cuda_set_cache_budget(sn_cuda *c, long long bytes);
long long sn_cuda_cache_used(const sn_cuda *c);
int sn_cuda_cache_count(const sn_cuda *c);
int sn_cuda_cache_put(sn_cuda *c, const sn_pack *p, const sn_tensor *t, const sn_pq_res *res);   /* 0 = resident, -1 = no room */
void sn_cuda_cache_drop(sn_cuda *c, const sn_pack *p, const sn_tensor *t);

/* ---- Expert stream (JD's standing CUDA design, 2026-11-27) ----
 * Compressed experts are copied to the card, expanded there with CUDA, multiplied there, and their VRAM slots are reused
 * for the next token. The warm expert cache stays in system RAM holding the COMPRESSED experts, so a load is a memcpy into
 * pinned staging plus one DMA. Decode (1..8 tokens) runs the DTC kernels straight from a slot (the expansion happens in
 * registers); prefill expands each matrix into a float scratch on the card and runs a GEMM that mirrors the CPU's AVX2
 * (or scalar) kernel. Both are bit-identical to the CPU, so the output cannot tell which side computed an expert.
 * Sizing: the slot count is "as many as fit" in the VRAM the hardware calibration measured free (nemo-calib.txt), never more
 * than the device reports free right now, after the dense tensors and a 64 MB headroom; staging is two halves of one
 * routed set (top-k experts). */
typedef struct {
    int Lt, I, K, nmax;              /* latent width, expert width, top-k, most tokens per layer call */
    uint64_t up_blocks, dn_blocks;   /* 256-weight blocks of one expert's up / down matrix */
    int nkeys;                       /* layers x experts (slot lookup table size) */
    int simd;                        /* 1: the CPU GEMM is the AVX2/FMA kernel, 0: the scalar one (sn_gemm_simd) */
} sn_xs_geom;

typedef struct {
    int on, slots, stage;
    long long slot_bytes, vram_bytes, pinned_bytes, work_bytes;
    char sized_by[400];
    uint64_t hits, loads, bytes_up, experts_dtc, experts_gemm, dense_gemm;
} sn_xs_info;

int sn_cuda_xs_open(sn_cuda *c, const sn_pack *p, const sn_xs_geom *g, char *why, int whylen);   /* 0, or -1 with a reason */
int sn_cuda_xs_on(const sn_cuda *c);
void sn_cuda_xs_info(const sn_cuda *c, sn_xs_info *out);
int sn_cuda_xs_chunk(const sn_cuda *c);       /* experts staged per half */
int sn_cuda_xs_nstage(const sn_cuda *c);
/* staging buffer j (pinned RAM): where the packed bytes of the up (which 0) or down (which 1) matrix go */
void sn_cuda_xs_stage_ptrs(sn_cuda *c, int j, int which, uint8_t **idx, uint16_t **mn, uint16_t **sc, float **book);
int sn_cuda_xs_stage_wait(sn_cuda *c, int j0, int n);     /* block until staging buffers j0.. are free to refill */
void sn_cuda_xs_chunk_begin(sn_cuda *c);                  /* slots claimed after this are protected until the next call */
int sn_cuda_xs_find(sn_cuda *c, int key);                 /* slot already holding expert `key`, or -1 */
int sn_cuda_xs_claim(sn_cuda *c, int key);                /* a slot to load `key` into (least recently used), or -1 */
int sn_cuda_xs_upload(sn_cuda *c, int j, int slot, int key);   /* async DMA staging j -> slot */
/* per MoE layer: n token latents (and their double prefix sums for decode), the expert-major token / output-row lists */
int sn_cuda_xs_layer_begin(sn_cuda *c, const float *lat, const double *plat, int n, const int *tok, const int *oidx, int nidx);
int sn_cuda_xs_dtc(sn_cuda *c, int slot, int tok, int orow);    /* decode: one token through one expert */
int sn_cuda_xs_gemm(sn_cuda *c, int slot, int start, int cnt);  /* prefill: list entries [start, start + cnt) through one expert */
int sn_cuda_xs_layer_end(sn_cuda *c, float *eo, int nrows);     /* wait; copy nrows x Lt outputs back */
void sn_cuda_xs_abort(sn_cuda *c);                              /* after a failure: drain the card, forget slot contents */
/* prefill product on a dense tensor resident on the card: expand on the card + the CPU-mirroring GEMM (act 0 or 2) */
int sn_cuda_xs_dense_gemm(sn_cuda *c, const sn_pack *p, const sn_tensor *t, const float *X, int N, int K, int R, float *Y, int act);
/* test hook: expand [first, first + n) of a resident tensor t, or of slot `slot`'s matrix `which` when t is NULL */
int sn_cuda_expand(sn_cuda *c, const sn_pack *p, const sn_tensor *t, int slot, int which, uint64_t first, uint64_t n, float *out);

#endif
