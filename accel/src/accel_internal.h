/* accel_internal.h - the backend interface every GPU lane implements.
 *
 * Co-implemented by Cortana for JD's acceleration lanes; NTSE and
 * the nq2 format are Elfie's. Internal to accel/: nothing in the
 * pure C engine includes this.
 *
 * One pipeline, several backends. The core (accel_core.c) owns
 * validation, the memory budget, the device layout of a tensor,
 * fallback policy and the streaming handshake; a backend only
 * moves bytes, records and waits on events, and launches the
 * seven kernels. Every backend call returns NTSE_OK or a status;
 * a backend never prints on the hot path and never aborts.
 */
#ifndef NTSE_ACCEL_INTERNAL_H
#define NTSE_ACCEL_INTERNAL_H

#include <stdint.h>
#include <stddef.h>
#include "../include/ntse_accel.h"

#if UINTPTR_MAX != 0xffffffffffffffffull
#error "the accel lanes are 64-bit only (vendor calling conventions and CUdeviceptr)"
#endif

/* Device buffer: CUDA uses ptr, OpenCL uses obj (a cl_mem). */
typedef struct {
    uint64_t ptr;
    void *obj;
    uint64_t bytes;
} ntse_dmem;

/* A typed reference into a device buffer: byte offset, always a
 * multiple of the element size (the layout aligns every section to
 * NTSE_ALIGN bytes). */
typedef struct {
    const ntse_dmem *m;
    uint64_t off;
} ntse_dref;

#define NTSE_ALIGN 256u

/* Queues: work on one queue runs in submission order. */
#define NTSE_Q_COMPUTE 0
#define NTSE_Q_COPY    1
#define NTSE_NQUEUES   2

/* Event slots owned by a backend. */
#define NTSE_MAX_EVENTS 16

/* Kernels. */
enum {
    NTSE_K_SEGSUM = 0,      /* A[b][j]                                 */
    NTSE_K_BUILD_EAGER,     /* T[b][j][k] for all k                    */
    NTSE_K_BUILD_ACTIVE,    /* T[b][j][k] for (j,k) in the active list */
    NTSE_K_APPLY,           /* y from T, A, idx, frames                */
    NTSE_K_APPLY_DIRECT,    /* y from book, x, A, idx, frames          */
    NTSE_K_EXPAND,          /* W from book, idx, frames                */
    NTSE_K_GEMV,            /* y = W x, fixed-tree reduction           */
    NTSE_K_LINEAR_BF16,     /* Little Nemo projection                  */
    NTSE_K_MARK,            /* lazy: flag (j, idx) pairs used          */
    NTSE_K_BUILD_LAZY,      /* lazy: T[b][j][k] for flagged (j,k) only */
    NTSE_K_COUNT
};

typedef struct {
    /* tensor (device layout, see accel_core.c) */
    ntse_dref bookT, book, idxT, mnT, scT, act;
    /* activations and workspace */
    ntse_dref x, A, T, y, W;
    ntse_dref F;            /* lazy flags, uint32 [nseg][entries]      */
    uint64_t rows, cols, nseg, nact;
    uint32_t block, entries, batch;
} ntse_kargs;

typedef struct ntse_be ntse_be;

typedef struct {
    const char *name;
    void (*destroy)(ntse_be *be);
    int (*dalloc)(ntse_be *be, uint64_t bytes, ntse_dmem *out);
    void (*dfree)(ntse_be *be, ntse_dmem *m);
    /* pinned (page-locked) host memory for async transfers */
    int (*halloc)(ntse_be *be, uint64_t bytes, void **host, void **tok);
    void (*hfree)(ntse_be *be, void *host, void *tok);
    /* Host to device. tok == NULL: src is pageable and the call is
     * complete (src reusable) when it returns. tok != NULL: src is
     * the pinned buffer from halloc and the copy is asynchronous on
     * queue q; src is owned by the copy until an event recorded
     * after it on q completes. */
    int (*put)(ntse_be *be, int q, const ntse_dmem *dst, uint64_t off,
               const void *src, uint64_t n, void *tok);
    int (*get)(ntse_be *be, int q, void *dst, const ntse_dmem *src,
               uint64_t off, uint64_t n, void *tok);
    int (*zero)(ntse_be *be, int q, const ntse_dmem *m);
    int (*ev_record)(ntse_be *be, int q, int ev);
    int (*ev_wait)(ntse_be *be, int q, int ev);   /* device-side wait */
    int (*ev_sync)(ntse_be *be, int ev);          /* host waits       */
    int (*ev_query)(ntse_be *be, int ev);         /* 1 done, 0 not    */
    int (*qsync)(ntse_be *be, int q);
    int (*launch)(ntse_be *be, int q, int kernel, const ntse_kargs *k);
    int (*linear_bf16)(ntse_be *be, int q, ntse_dref W, ntse_dref x,
                       ntse_dref y, uint32_t out, uint32_t in);
} ntse_be_ops;

typedef struct {
    char device[256];      /* human-readable device line           */
    uint64_t total_mem;    /* device memory                        */
    uint64_t free_mem;     /* free at open (0 if unknown)          */
    uint64_t max_alloc;    /* largest single allocation            */
    int integrated;        /* shares host memory                   */
    int has_fp64;
    int denorms;           /* fp32 denormals honored (1) or flushed */
} ntse_be_info;

/* Backend constructors. want: NTSE_LANE_CUDA, NTSE_LANE_CL_IGPU or
 * NTSE_LANE_CL_GPU. They return NTSE_E_NODEV when the lane is not
 * present (library missing, no device), with a reason in why. */
int ntse_cuda_create(ntse_be **out, const ntse_be_ops **ops,
                     ntse_be_info *info, char *why, size_t whylen);
int ntse_cl_create(ntse_lane want, ntse_be **out, const ntse_be_ops **ops,
                   ntse_be_info *info, char *why, size_t whylen);

/* Exact fp16 -> fp32 (the same conversion as nq2-infer / nq2-dc). */
float ntse_half_to_float(uint16_t h);

/* Dynamic library loading (no vendor SDK at build time). */
void *ntse_dl_open(const char *name);
void *ntse_dl_sym(void *lib, const char *sym);
void ntse_dl_close(void *lib);

#endif
