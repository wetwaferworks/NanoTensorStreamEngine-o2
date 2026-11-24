/* loader.h - Little Nemo checkpoint loader, pure C.
 * Walks the safetensors shard headers in a checkpoint directory and builds
 * a tensor index; tensors are fetched on demand and widened to fp32.
 * Headers are untrusted input (hardened length guard and span checks, the
 * 2026-11-24 pattern). The header-walk code is shared in spirit with the
 * P0 inspector in nemo.c; the two will be unified once both are stable.
 */
#ifndef NEMO_LOADER_H
#define NEMO_LOADER_H

#include <stdint.h>

typedef struct {
    char *name;
    char *shard_path;
    uint64_t data_off; /* absolute file offset of the tensor's first byte */
    uint64_t count;    /* element count */
    int ndim;
    uint64_t shape[8];
    int elem_size;     /* bytes per element in the file */
    int is_bf16;
    int is_f32;        /* loads dispatch on dtype, never on size alone */
} nemo_tensor_info;

typedef struct {
    nemo_tensor_info *items;
    int n, cap;
} nemo_checkpoint;

/* Index every *.safetensors shard in dir. Returns 0 on success. */
int nemo_checkpoint_open(nemo_checkpoint *cp, const char *dir);
void nemo_checkpoint_close(nemo_checkpoint *cp);
const nemo_tensor_info *nemo_find(const nemo_checkpoint *cp, const char *name);

/* Fetch a tensor widened to fp32 (malloc'd; caller frees).
 * BF16 widens exactly; F32 passes through; anything else is refused. */
float *nemo_load_f32(const nemo_checkpoint *cp, const nemo_tensor_info *t);

/* FNV-1a 64 over the tensor's raw bytes in the file (fingerprinting). */
uint64_t nemo_tensor_fnv(const nemo_checkpoint *cp, const nemo_tensor_info *t);

#endif

/* Fetch a tensor's raw bf16 words (malloc'd; caller frees). Refuses
 * non-bf16 tensors. This is the runner's weight path: weights stay bf16
 * in memory and widen inside the kernels, never a full fp32 copy. */
uint16_t *nemo_load_bf16(const nemo_checkpoint *cp, const nemo_tensor_info *t);
