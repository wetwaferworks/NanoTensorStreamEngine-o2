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

#endif
