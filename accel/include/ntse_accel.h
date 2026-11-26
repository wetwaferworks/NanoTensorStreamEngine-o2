/* ntse_accel.h - NTSE-o2 GPU acceleration lanes (research, scratch).
 *
 * Idea and direction: JD. Lane proposals: Gizmo (NTSE-O2 research
 * lanes, 2026-11-26). NTSE, the nq2 format and the converter:
 * Elfie (E.L.F.I.E.). Co-implementer of this layer: Cortana.
 *
 * This is an OPTIONAL layer that lives entirely under accel/. The
 * pure C engine never includes it and builds and runs with zero
 * GPU code. The CPU path stays the reference and the official
 * path; every accelerated operator here is checked against it.
 *
 * Lanes:
 *   NTSE_LANE_CPU          the reference (tools/nq2-dc-kernel.h).
 *   NTSE_LANE_CUDA         NVIDIA CUDA: driver API + NVRTC, both
 *                          loaded at run time (no CUDA link, no
 *                          MSVC needed, binary runs on machines
 *                          without CUDA).
 *   NTSE_LANE_CL_IGPU      OpenCL on an integrated GPU (the CPU's
 *                          own graphics, e.g. Intel UHD).
 *   NTSE_LANE_CL_GPU       OpenCL on a discrete GPU: the default
 *                          lane for GPUs where CUDA is unavailable.
 *
 * Safety contract (safety-critical bar):
 *   - every size and input is validated before any allocation;
 *   - every vendor API return is checked;
 *   - a device failure is sticky: the context drops to the CPU
 *     reference for the rest of its life and says so;
 *   - device memory is budgeted (hard cap, never above the
 *     device's own free memory), every allocation is charged;
 *   - bad INPUT is refused on every lane (no fallback hides it).
 *
 * Determinism contract: no kernel here uses atomics in any
 * accumulation; every reduction has a fixed order, so repeated
 * runs are bit-identical. The DTC kernels (eager, active, direct)
 * use exactly the accumulation order pinned by nq2-dc-kernel.h
 * with contraction (FMA) disabled, so they are designed to be
 * bit-identical to the CPU kernel, not just close; the tests
 * check that. The expanded-weights GEMV uses a fixed tree and is
 * equal to the CPU only to float noise (disclosed, measured).
 *
 * Threading: a context is used by one thread at a time (the
 * caller's). The streaming pipeline starts its own stager thread
 * internally and coordinates it with release/acquire atomics and
 * the Tensor Handshake door (tools/nq2-handshake.h): zero mutexes
 * in this code. Vendor drivers have their own internal locking,
 * which is outside this code's control and is disclosed as such.
 */
#ifndef NTSE_ACCEL_H
#define NTSE_ACCEL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- status codes ---- */
#define NTSE_OK             0
#define NTSE_E_INVAL       -1  /* bad argument or malformed tensor */
#define NTSE_E_UNSUPPORTED -2  /* valid, but this lane cannot do it */
#define NTSE_E_NOMEM       -3  /* host allocation failed */
#define NTSE_E_BUDGET      -4  /* would exceed the device budget */
#define NTSE_E_DEVICE      -5  /* a vendor API call failed */
#define NTSE_E_NODEV       -6  /* lane not present on this machine */
#define NTSE_E_STATE       -7  /* buffer ownership violation refused */
#define NTSE_E_NONFINITE   -8  /* NaN/inf in book or frames */

const char *ntse_accel_strerror(int code);

typedef enum {
    NTSE_LANE_AUTO = -1,
    NTSE_LANE_CPU = 0,
    NTSE_LANE_CUDA = 1,
    NTSE_LANE_CL_IGPU = 2,
    NTSE_LANE_CL_GPU = 3
} ntse_lane;

const char *ntse_lane_name(ntse_lane l);

/* Operators for y = W x on a dictionary-encoded tensor. */
typedef enum {
    NTSE_OP_DTC_EAGER = 0,  /* full table T[j][k] for every k        */
    NTSE_OP_DTC_ACTIVE = 1, /* table only for indices the tensor uses
                             * (the active set is fixed by idx, so it
                             * is computed once, at upload)            */
    NTSE_OP_DTC_DIRECT = 2, /* no table: each row dots its own
                             * prototype with the segment (the GPU
                             * form of "lazy"; see README)             */
    NTSE_OP_EXPANDED = 3,    /* reference: materialize W in device
                             * memory, then dense GEMV                 */
    NTSE_OP_DTC_LAZY = 4    /* tables built on demand: a mark pass
                             * flags the (segment, code) pairs the
                             * index touches this call, the build
                             * pass computes only flagged entries;
                             * same bits as eager                  */
} ntse_op;

const char *ntse_op_name(ntse_op op);

/* One dictionary-encoded tensor in the nq2 (non-PQ) layout:
 *   w[r][j*block + i] = mn[r][j] + sc[r][j] * book[idx[r][j]][i]
 * rows x cols weights, cols % block == 0, nseg = cols / block.
 * idx, mn, sc are row-major [rows][nseg] exactly as the stream
 * stores one tensor's blocks; mn and sc are the raw fp16 frame
 * bits (the stream's encoding), book is entries x block fp32.
 * The arrays are BORROWED: they must stay valid and unchanged
 * while any device handle made from them is alive, because the
 * CPU fallback reads them. */
typedef struct {
    uint64_t rows, cols;
    uint32_t block, entries;
    const float *book;
    const uint8_t *idx;
    const uint16_t *mn_h;
    const uint16_t *sc_h;
} ntse_dtc_tensor;

/* Limits (validated): */
#define NTSE_MAX_BLOCK    4096u
#define NTSE_MAX_ENTRIES  256u   /* 8-bit index */
#define NTSE_MAX_BATCH    64u
#define NTSE_MAX_DIM      (1ull << 31)

typedef struct {
    ntse_lane lane;            /* AUTO picks CUDA > CL_GPU > CL_IGPU > CPU */
    uint64_t vram_budget;      /* bytes; 0 = derive from the device       */
    int allow_fallback;        /* 1: device failure -> CPU reference      */
    int verbose;               /* 1: print the startup / fault lines      */
} ntse_accel_config;

void ntse_accel_config_default(ntse_accel_config *c);

typedef struct ntse_accel ntse_accel;
typedef struct ntse_dtensor ntse_dtensor;

/* Opening never fails for lack of a GPU when allow_fallback is
 * set: the context comes up on the CPU lane and says why. */
int ntse_accel_open(ntse_accel **out, const ntse_accel_config *cfg);
void ntse_accel_close(ntse_accel *a);

ntse_lane ntse_accel_lane(const ntse_accel *a);      /* lane in use now */
const char *ntse_accel_device(const ntse_accel *a);  /* device string   */
int ntse_accel_failed(const ntse_accel *a);          /* sticky fault?   */
uint64_t ntse_accel_budget(const ntse_accel *a);
uint64_t ntse_accel_charged(const ntse_accel *a);    /* bytes in use    */
uint64_t ntse_accel_peak(const ntse_accel *a);       /* peak charged    */
uint64_t ntse_accel_h2d_bytes(const ntse_accel *a);  /* bytes uploaded  */

/* Validate a tensor without touching any device (also used by
 * every upload): geometry, overflow, idx < entries, finite book
 * and frames. Returns NTSE_OK or a refusal code with a message. */
int ntse_dtc_validate(const ntse_dtc_tensor *t, char *why, size_t whylen);

int ntse_accel_upload(ntse_accel *a, const ntse_dtc_tensor *t,
                      ntse_dtensor **out);
void ntse_accel_release(ntse_accel *a, ntse_dtensor *d);

/* y[b][r] = sum_c W[r][c] * x[b][c] for b < batch. x is batch x
 * cols, y is batch x rows, both row-major host fp32. *used (may
 * be NULL) reports which lane produced y. */
int ntse_accel_matvec(ntse_accel *a, ntse_dtensor *d, ntse_op op,
                      const float *x, uint32_t batch, float *y,
                      ntse_lane *used);

/* The CPU reference, callable directly (no context): the shared
 * kernel nqdc_build + nqdc_apply per batch row for the DTC ops,
 * and expand + sequential fp32 dot for NTSE_OP_EXPANDED. */
int ntse_cpu_matvec(const ntse_dtc_tensor *t, ntse_op op,
                    const float *x, uint32_t batch, float *y);

/* fp64 oracle over the same packed function (for error metrics). */
int ntse_cpu_oracle(const ntse_dtc_tensor *t, const float *x,
                    uint32_t batch, double *y);

/* Little Nemo projection: y[o] = sum_i bf16(W[o][i]) * x[i] with
 * the house double accumulation, fp32 result. W is uploaded per
 * call (Little Nemo reuses its weight buffers across layers, so a
 * pointer-keyed cache would be unsound). */
int ntse_accel_linear_bf16(ntse_accel *a, const uint16_t *W,
                           const float *x, float *y, int out, int in,
                           ntse_lane *used);

/* ---- Tensor Handshake, GPU side: streamed DTC ----
 * Items are consumed in ordinal order; preparation (host staging,
 * transposition into the device layout, H2D) runs ahead into
 * `slots` pinned staging buffers. Slot ownership is explicit:
 *   FREE -> STAGING (stager) -> STAGED -> INFLIGHT (device queues,
 *   bounded by events) -> FREE only after the slot's completion
 *   event has been observed complete on the host.
 * No buffer is ever recycled while another stage owns it; an
 * attempted early recycle is refused (NTSE_E_STATE) and counted. */
typedef struct {
    const ntse_dtc_tensor *t;
    const float *x;      /* batch x t->cols */
    float *y;            /* batch x t->rows */
} ntse_stream_item;

typedef struct {
    uint32_t slots;      /* 1 single, 2 double, 3 triple (max 4) */
    ntse_op op;          /* a DTC op (EXPANDED is refused here)    */
    int stager_thread;   /* 1: stage on a second thread             */
} ntse_stream_config;

typedef struct {
    uint64_t items, h2d_bytes, d2h_bytes;
    uint64_t slot_bytes;           /* pinned bytes per slot          */
    uint64_t stager_waits;         /* stager found its slot owned    */
    uint64_t driver_waits;         /* driver found its slot unstaged */
    uint64_t ownership_violations; /* must be 0                      */
    double wall_ms;
    ntse_lane lane;                /* lane that produced the outputs */
} ntse_stream_stats;

int ntse_accel_stream(ntse_accel *a, const ntse_stream_item *items,
                      uint32_t n, uint32_t batch,
                      const ntse_stream_config *cfg,
                      ntse_stream_stats *st);

/* ---- Prepared stream plan (Lane 2: Tensor Handshake, GPU side) ----
 *
 * For models larger than device memory: every tensor is laid out
 * ONCE into a host image of its device layout (transposed idx and
 * frames, active list), each distinct dictionary book is uploaded
 * ONCE and stays resident, and a run moves only the per-tensor
 * image plus the activations over PCIe, through `slots` device
 * buffers (1 single, 2 double, 3 triple buffering, max 4) with the
 * copy queue running ahead of the compute queue.
 *
 * The host image lives in one pinned arena when it fits the pinned
 * budget (direct asynchronous DMA, no per-token CPU copy), else in
 * pageable memory staged through pinned slot buffers.
 *
 * Ownership per slot: FREE -> STAGING -> STAGED -> INFLIGHT -> FREE,
 * claimed by CAS; a slot (device buffer and pinned buffer) returns
 * to FREE only after its completion event has been observed on the
 * host. Zero mutexes.
 *
 * The plan snapshots the tensors at create time (the GPU path never
 * reads the caller's arrays again); the CPU fallback reads the
 * caller's arrays, so they must stay unchanged while the plan lives.
 * Outputs are bit-identical to ntse_accel_matvec on the same lane.
 * Pinned budget: cfg.pin_budget, else env NTSE_ACCEL_PIN_MB, else
 * 1 GiB; clamped to 4 GiB. */
typedef struct ntse_plan ntse_plan;

typedef struct {
    uint32_t slots;      /* 1..4                                     */
    ntse_op op;          /* DTC_EAGER, DTC_ACTIVE or DTC_DIRECT      */
    uint32_t max_batch;  /* largest batch a run may use (1..64)      */
    int pin_image;       /* 1: pinned image if it fits (default), 0: never */
    uint64_t pin_budget; /* bytes, 0 = default                       */
} ntse_plan_config;

typedef struct {
    uint32_t tensors, books;
    int on_device;               /* 0: host-only plan (CPU reference)  */
    int pinned_image;            /* 1: direct DMA from a pinned arena  */
    uint64_t image_bytes;        /* host image, all tensors            */
    uint64_t book_bytes;         /* resident on the device, once       */
    uint64_t slot_dev_bytes;     /* per slot                           */
    uint64_t slot_pin_bytes;     /* per slot                           */
    uint64_t pinned_bytes;       /* total page-locked host memory      */
    double create_ms;
} ntse_plan_info;

void ntse_plan_config_default(ntse_plan_config *c);
int ntse_accel_plan_create(ntse_accel *a, const ntse_dtc_tensor *const *ts,
                           uint32_t n, const ntse_plan_config *cfg,
                           ntse_plan **out);
/* x[i]: batch x cols of tensor i; y[i]: batch x rows of tensor i. */
int ntse_accel_plan_run(ntse_accel *a, ntse_plan *p, const float *const *x,
                        float *const *y, uint32_t batch,
                        ntse_stream_stats *st);
void ntse_accel_plan_info_get(const ntse_plan *p, ntse_plan_info *info);
void ntse_accel_plan_free(ntse_accel *a, ntse_plan *p);
/* PCIe probe (GB/s, median of 5): pinned H2D, pageable H2D, pinned D2H. */
int ntse_accel_probe_bandwidth(ntse_accel *a, uint64_t bytes, double *h2d_pinned_gbs,
                               double *h2d_pageable_gbs, double *d2h_pinned_gbs);

/* Test hook: ask the pipeline to try to recycle slot 0 before its
 * completion event is observed. The guard must refuse it; the
 * stream then reports one ownership violation refused, and still
 * produces correct output (the refused recycle never happens). */
void ntse_accel_test_force_early_recycle(ntse_accel *a, int on);

#ifdef __cplusplus
}
#endif
#endif /* NTSE_ACCEL_H */
