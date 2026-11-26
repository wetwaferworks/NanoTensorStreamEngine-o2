/* nemo_calib.h - pick the compute lane at engine startup. Pure C.
 *
 * Probes every lane this machine might offer, checks each one for
 * correctness and speed, and picks the fastest correct one, preferring
 * CUDA; CPU only when no GPU lane passes. The verdict is cached in a
 * small text file keyed by GPU name + driver + NVRTC version + calib
 * version, so later launches skip the tests.
 *   CUDA        driver + NVRTC load, VRAM free, the DTC and FULL GEMV
 *               kernels bit-exact against a C emulation on synthetic PQ
 *               tensors, then an in_proj-sized GEMV and a memory read
 *               timed (nemo_gpu_selftest).
 *   OpenCL-NV   OpenCL platform/device probe (NVIDIA vendor)
 *   OpenCL-iGPU OpenCL probe (Intel/AMD or unified-memory GPU)
 *   CPU         always correct (it is the reference); speed from a
 *               memory-read micro-bench over the raw bf16 bytes/token. */
#ifndef NEMO_CALIB_H
#define NEMO_CALIB_H
#include "nemo_gpu.h"

typedef struct {
    char lane[16];               /* "cuda" or "cpu" */
    int cached;                  /* 1: verdict read from the cache file */
    char path[512];              /* cache file */
    char key[384];
    /* per lane: status pass/rejected/skipped and why */
    char cuda_status[16], cuda_why[256];
    char oclnv_status[16], oclnv_why[256], oclnv_dev[128];
    char igpu_status[16], igpu_why[256], igpu_dev[128];
    char cpu_status[16], cpu_why[256];
    double est_cuda_tps, est_cpu_tps;
    nemo_gpu_probe probe;
    double ms;                   /* time spent calibrating */
} nemo_calib;

/* override: NULL/"auto" = choose, "cuda" or "cpu" = forced (the cache
 * is still read and written for the record). recal = ignore the cache.
 * have_pack = 0 rejects CUDA (no .nq2 pack to run from). Never fails:
 * at worst the lane is cpu. */
void nemo_calib_run(nemo_calib *c, const char *override, int recal, int have_pack);
/* Log every lane verdict to stderr. */
void nemo_calib_log(const nemo_calib *c);
#endif
