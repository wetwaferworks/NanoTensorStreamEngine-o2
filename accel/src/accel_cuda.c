/* accel_cuda.c - the CUDA lane (research, scratch).
 *
 * Direction: JD. Lane 1/2 proposals: Gizmo. Co-implemented by
 * Cortana on Elfie's NTSE.
 *
 * Design: the CUDA DRIVER API (nvcuda.dll / libcuda.so.1) and
 * NVRTC (nvrtc64_*.dll / libnvrtc.so) are both loaded at run time.
 * Nothing links against CUDA, no CUDA header is needed, and the
 * build needs only MinGW gcc (nvcc on Windows would need MSVC,
 * which this workstation does not have). A machine without CUDA
 * gets NTSE_E_NODEV here and the context falls back cleanly.
 *
 * The kernels are compiled once per context for the device's own
 * architecture (cubin, sm_XY); if NVRTC does not know the
 * architecture, PTX for compute_XY is JIT-loaded instead.
 *
 * Every driver call is checked. Calls are made only from the
 * thread that drives the context; the streaming stager thread
 * never calls into CUDA.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "accel_internal.h"
#include "accel_kernels_cuda.h"

typedef int CUresult;
typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;
typedef struct CUevent_st *CUevent;
typedef int nvrtcResult;
typedef struct _nvrtcProgram *nvrtcProgram;

#define CUDA_SUCCESS 0
#define CUDA_ERROR_NOT_READY 600
#define CU_ATTR_MP_COUNT 16
#define CU_ATTR_INTEGRATED 18
#define CU_ATTR_CC_MAJOR 75
#define CU_ATTR_CC_MINOR 76
#define CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK 0
#define CU_EVENT_DISABLE_TIMING 2
#define CU_STREAM_NON_BLOCKING 1

typedef struct {
    CUresult (*cuInit)(unsigned);
    CUresult (*cuDriverGetVersion)(int *);
    CUresult (*cuDeviceGetCount)(int *);
    CUresult (*cuDeviceGet)(CUdevice *, int);
    CUresult (*cuDeviceGetName)(char *, int, CUdevice);
    CUresult (*cuDeviceGetAttribute)(int *, int, CUdevice);
    CUresult (*cuDeviceTotalMem)(size_t *, CUdevice);
    CUresult (*cuDevicePrimaryCtxRetain)(CUcontext *, CUdevice);
    CUresult (*cuDevicePrimaryCtxRelease)(CUdevice);
    CUresult (*cuCtxSetCurrent)(CUcontext);
    CUresult (*cuMemGetInfo)(size_t *, size_t *);
    CUresult (*cuMemAlloc)(CUdeviceptr *, size_t);
    CUresult (*cuMemFree)(CUdeviceptr);
    CUresult (*cuMemAllocHost)(void **, size_t);
    CUresult (*cuMemFreeHost)(void *);
    CUresult (*cuMemcpyHtoDAsync)(CUdeviceptr, const void *, size_t, CUstream);
    CUresult (*cuMemcpyDtoHAsync)(void *, CUdeviceptr, size_t, CUstream);
    CUresult (*cuMemsetD8Async)(CUdeviceptr, unsigned char, size_t, CUstream);
    CUresult (*cuStreamCreate)(CUstream *, unsigned);
    CUresult (*cuStreamDestroy)(CUstream);
    CUresult (*cuStreamSynchronize)(CUstream);
    CUresult (*cuStreamWaitEvent)(CUstream, CUevent, unsigned);
    CUresult (*cuEventCreate)(CUevent *, unsigned);
    CUresult (*cuEventDestroy)(CUevent);
    CUresult (*cuEventRecord)(CUevent, CUstream);
    CUresult (*cuEventQuery)(CUevent);
    CUresult (*cuEventSynchronize)(CUevent);
    CUresult (*cuModuleLoadData)(CUmodule *, const void *);
    CUresult (*cuModuleUnload)(CUmodule);
    CUresult (*cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
    CUresult (*cuFuncGetAttribute)(int *, int, CUfunction);
    CUresult (*cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned,
                               unsigned, unsigned, unsigned, unsigned,
                               CUstream, void **, void **);
    CUresult (*cuGetErrorString)(CUresult, const char **);
} cu_api;

typedef struct {
    nvrtcResult (*nvrtcVersion)(int *, int *);
    nvrtcResult (*nvrtcCreateProgram)(nvrtcProgram *, const char *,
                                      const char *, int, const char *const *,
                                      const char *const *);
    nvrtcResult (*nvrtcCompileProgram)(nvrtcProgram, int, const char *const *);
    nvrtcResult (*nvrtcGetProgramLogSize)(nvrtcProgram, size_t *);
    nvrtcResult (*nvrtcGetProgramLog)(nvrtcProgram, char *);
    nvrtcResult (*nvrtcGetPTXSize)(nvrtcProgram, size_t *);
    nvrtcResult (*nvrtcGetPTX)(nvrtcProgram, char *);
    nvrtcResult (*nvrtcGetCUBINSize)(nvrtcProgram, size_t *);
    nvrtcResult (*nvrtcGetCUBIN)(nvrtcProgram, char *);
    nvrtcResult (*nvrtcDestroyProgram)(nvrtcProgram *);
} rtc_api;

static const char *const kernel_names[NTSE_K_COUNT] = {
    "k_segsum", "k_build_eager", "k_build_active", "k_apply",
    "k_apply_direct", "k_expand", "k_gemv", "k_linear_bf16",
    "k_mark", "k_build_lazy"
};

struct ntse_be {
    void *lib_cu, *lib_rtc;
    cu_api cu;
    rtc_api rtc;
    CUdevice dev;
    CUcontext ctx;
    int ctx_retained;
    CUstream q[NTSE_NQUEUES];
    CUevent ev[NTSE_MAX_EVENTS];
    CUmodule mod;
    CUfunction fn[NTSE_K_COUNT];
    int mp_count;
    CUresult last;
};

/* ------------------------------------------------------------ */

static int load_sym(void *lib, const char *name, void *slot)
{
    void *p = ntse_dl_sym(lib, name);
    if (!p) return -1;
    memcpy(slot, &p, sizeof p);
    return 0;
}

static int load_cu(struct ntse_be *b, char *why, size_t wl)
{
#ifdef _WIN32
    b->lib_cu = ntse_dl_open("nvcuda.dll");
#else
    b->lib_cu = ntse_dl_open("libcuda.so.1");
#endif
    if (!b->lib_cu) { snprintf(why, wl, "CUDA driver library not found"); return -1; }
    struct { const char *n; void *slot; } t[] = {
        { "cuInit", &b->cu.cuInit },
        { "cuDriverGetVersion", &b->cu.cuDriverGetVersion },
        { "cuDeviceGetCount", &b->cu.cuDeviceGetCount },
        { "cuDeviceGet", &b->cu.cuDeviceGet },
        { "cuDeviceGetName", &b->cu.cuDeviceGetName },
        { "cuDeviceGetAttribute", &b->cu.cuDeviceGetAttribute },
        { "cuDeviceTotalMem_v2", &b->cu.cuDeviceTotalMem },
        { "cuDevicePrimaryCtxRetain", &b->cu.cuDevicePrimaryCtxRetain },
        { "cuDevicePrimaryCtxRelease_v2", &b->cu.cuDevicePrimaryCtxRelease },
        { "cuCtxSetCurrent", &b->cu.cuCtxSetCurrent },
        { "cuMemGetInfo_v2", &b->cu.cuMemGetInfo },
        { "cuMemAlloc_v2", &b->cu.cuMemAlloc },
        { "cuMemFree_v2", &b->cu.cuMemFree },
        { "cuMemAllocHost_v2", &b->cu.cuMemAllocHost },
        { "cuMemFreeHost", &b->cu.cuMemFreeHost },
        { "cuMemcpyHtoDAsync_v2", &b->cu.cuMemcpyHtoDAsync },
        { "cuMemcpyDtoHAsync_v2", &b->cu.cuMemcpyDtoHAsync },
        { "cuMemsetD8Async", &b->cu.cuMemsetD8Async },
        { "cuStreamCreate", &b->cu.cuStreamCreate },
        { "cuStreamDestroy_v2", &b->cu.cuStreamDestroy },
        { "cuStreamSynchronize", &b->cu.cuStreamSynchronize },
        { "cuStreamWaitEvent", &b->cu.cuStreamWaitEvent },
        { "cuEventCreate", &b->cu.cuEventCreate },
        { "cuEventDestroy_v2", &b->cu.cuEventDestroy },
        { "cuEventRecord", &b->cu.cuEventRecord },
        { "cuEventQuery", &b->cu.cuEventQuery },
        { "cuEventSynchronize", &b->cu.cuEventSynchronize },
        { "cuModuleLoadData", &b->cu.cuModuleLoadData },
        { "cuModuleUnload", &b->cu.cuModuleUnload },
        { "cuModuleGetFunction", &b->cu.cuModuleGetFunction },
        { "cuFuncGetAttribute", &b->cu.cuFuncGetAttribute },
        { "cuLaunchKernel", &b->cu.cuLaunchKernel },
        { "cuGetErrorString", &b->cu.cuGetErrorString },
    };
    for (size_t i = 0; i < sizeof t / sizeof t[0]; i++) {
        if (load_sym(b->lib_cu, t[i].n, t[i].slot) != 0) {
            snprintf(why, wl, "CUDA driver lacks %s", t[i].n);
            return -1;
        }
    }
    return 0;
}

static void *open_nvrtc(void)
{
    const char *env = getenv("NTSE_NVRTC_DLL");
    if (env && *env) return ntse_dl_open(env);
#ifdef _WIN32
    static const char *names[] = { "nvrtc64_130_0.dll", "nvrtc64_120_0.dll",
                                   "nvrtc64_112_0.dll", "nvrtc64_111_0.dll" };
    const char *cp = getenv("CUDA_PATH");
    for (size_t i = 0; i < sizeof names / sizeof names[0]; i++) {
        void *h = NULL;
        if (cp && *cp) {
            char path[1024];
            snprintf(path, sizeof path, "%s\\bin\\%s", cp, names[i]);
            h = ntse_dl_open(path);
            if (!h) {
                snprintf(path, sizeof path, "%s\\bin\\x64\\%s", cp, names[i]);
                h = ntse_dl_open(path);
            }
        }
        if (!h) h = ntse_dl_open(names[i]);
        if (h) return h;
    }
    return NULL;
#else
    static const char *names[] = { "libnvrtc.so.13", "libnvrtc.so.12",
                                   "libnvrtc.so" };
    for (size_t i = 0; i < sizeof names / sizeof names[0]; i++) {
        void *h = ntse_dl_open(names[i]);
        if (h) return h;
    }
    return NULL;
#endif
}

static int load_rtc(struct ntse_be *b, char *why, size_t wl)
{
    b->lib_rtc = open_nvrtc();
    if (!b->lib_rtc) {
        snprintf(why, wl, "NVRTC library not found (set NTSE_NVRTC_DLL or CUDA_PATH)");
        return -1;
    }
    struct { const char *n; void *slot; int need; } t[] = {
        { "nvrtcVersion", &b->rtc.nvrtcVersion, 1 },
        { "nvrtcCreateProgram", &b->rtc.nvrtcCreateProgram, 1 },
        { "nvrtcCompileProgram", &b->rtc.nvrtcCompileProgram, 1 },
        { "nvrtcGetProgramLogSize", &b->rtc.nvrtcGetProgramLogSize, 1 },
        { "nvrtcGetProgramLog", &b->rtc.nvrtcGetProgramLog, 1 },
        { "nvrtcGetPTXSize", &b->rtc.nvrtcGetPTXSize, 1 },
        { "nvrtcGetPTX", &b->rtc.nvrtcGetPTX, 1 },
        { "nvrtcGetCUBINSize", &b->rtc.nvrtcGetCUBINSize, 0 },
        { "nvrtcGetCUBIN", &b->rtc.nvrtcGetCUBIN, 0 },
        { "nvrtcDestroyProgram", &b->rtc.nvrtcDestroyProgram, 1 },
    };
    for (size_t i = 0; i < sizeof t / sizeof t[0]; i++) {
        if (load_sym(b->lib_rtc, t[i].n, t[i].slot) != 0 && t[i].need) {
            snprintf(why, wl, "NVRTC lacks %s", t[i].n);
            return -1;
        }
    }
    return 0;
}

static int ck(struct ntse_be *b, CUresult r)
{
    b->last = r;
    return r == CUDA_SUCCESS ? NTSE_OK : NTSE_E_DEVICE;
}

static void cu_errstr(struct ntse_be *b, CUresult r, char *buf, size_t n)
{
    const char *s = NULL;
    if (b->cu.cuGetErrorString && b->cu.cuGetErrorString(r, &s) == CUDA_SUCCESS && s)
        snprintf(buf, n, "%s (%d)", s, r);
    else
        snprintf(buf, n, "CUresult %d", r);
}

/* Compile the kernels. arch_real=1: cubin for sm_XY; else PTX for
 * compute_XY. Returns a malloc'd image or NULL with a reason. */
static char *rtc_build(struct ntse_be *b, int major, int minor, int arch_real,
                       char *why, size_t wl)
{
    nvrtcProgram p = NULL;
    if (b->rtc.nvrtcCreateProgram(&p, ntse_cuda_src, "ntse_accel.cu", 0,
                                  NULL, NULL) != 0) {
        snprintf(why, wl, "nvrtcCreateProgram failed");
        return NULL;
    }
    char arch[64];
    snprintf(arch, sizeof arch, "--gpu-architecture=%s_%d%d",
             arch_real ? "sm" : "compute", major, minor);
    const char *opts[] = { arch, "--fmad=false", "--std=c++11",
                           "--ftz=false", "--prec-div=true", "--prec-sqrt=true" };
    nvrtcResult cr = b->rtc.nvrtcCompileProgram(p, 6, opts);
    if (cr != 0) {
        size_t ls = 0;
        b->rtc.nvrtcGetProgramLogSize(p, &ls);
        char log[600] = "";
        if (ls > 1 && ls < 1u << 20) {
            char *full = (char *)malloc(ls);
            if (full && b->rtc.nvrtcGetProgramLog(p, full) == 0)
                snprintf(log, sizeof log, "%s", full);
            free(full);
        }
        snprintf(why, wl, "NVRTC compile for %s failed: %.180s", arch + 19, log);
        b->rtc.nvrtcDestroyProgram(&p);
        return NULL;
    }
    size_t sz = 0;
    char *img = NULL;
    nvrtcResult gr;
    if (arch_real) {
        if (!b->rtc.nvrtcGetCUBINSize || !b->rtc.nvrtcGetCUBIN) {
            b->rtc.nvrtcDestroyProgram(&p);
            snprintf(why, wl, "NVRTC has no cubin output");
            return NULL;
        }
        gr = b->rtc.nvrtcGetCUBINSize(p, &sz);
        if (gr == 0 && sz > 0 && (img = (char *)malloc(sz)))
            gr = b->rtc.nvrtcGetCUBIN(p, img);
    } else {
        gr = b->rtc.nvrtcGetPTXSize(p, &sz);
        if (gr == 0 && sz > 0 && (img = (char *)malloc(sz)))
            gr = b->rtc.nvrtcGetPTX(p, img);
    }
    b->rtc.nvrtcDestroyProgram(&p);
    if (gr != 0 || !img) {
        free(img);
        snprintf(why, wl, "NVRTC output retrieval failed");
        return NULL;
    }
    return img;
}

static void cu_destroy(ntse_be *b)
{
    if (!b) return;
    if (b->ctx) b->cu.cuCtxSetCurrent(b->ctx);
    for (int i = 0; i < NTSE_NQUEUES; i++)
        if (b->q[i]) { b->cu.cuStreamSynchronize(b->q[i]); b->cu.cuStreamDestroy(b->q[i]); }
    for (int i = 0; i < NTSE_MAX_EVENTS; i++)
        if (b->ev[i]) b->cu.cuEventDestroy(b->ev[i]);
    if (b->mod) b->cu.cuModuleUnload(b->mod);
    if (b->ctx_retained) b->cu.cuDevicePrimaryCtxRelease(b->dev);
    ntse_dl_close(b->lib_rtc);
    ntse_dl_close(b->lib_cu);
    free(b);
}

static int cur(ntse_be *b) { return ck(b, b->cu.cuCtxSetCurrent(b->ctx)); }

static int cu_dalloc(ntse_be *b, uint64_t bytes, ntse_dmem *m)
{
    if (cur(b) != NTSE_OK) return NTSE_E_DEVICE;
    CUdeviceptr p = 0;
    CUresult r = b->cu.cuMemAlloc(&p, (size_t)bytes);
    if (r == 2 /* CUDA_ERROR_OUT_OF_MEMORY */) { b->last = r; return NTSE_E_BUDGET; }
    if (ck(b, r) != NTSE_OK) return NTSE_E_DEVICE;
    m->ptr = p; m->obj = NULL; m->bytes = bytes;
    return NTSE_OK;
}

static void cu_dfree(ntse_be *b, ntse_dmem *m)
{
    if (m->ptr) { cur(b); b->cu.cuMemFree(m->ptr); }
    m->ptr = 0;
}

static int cu_halloc(ntse_be *b, uint64_t bytes, void **host, void **tok)
{
    if (cur(b) != NTSE_OK) return NTSE_E_DEVICE;
    void *p = NULL;
    if (ck(b, b->cu.cuMemAllocHost(&p, (size_t)bytes)) != NTSE_OK || !p)
        return NTSE_E_DEVICE;
    *host = p;
    *tok = p;    /* non-NULL token marks pinned memory */
    return NTSE_OK;
}

static void cu_hfree(ntse_be *b, void *host, void *tok)
{
    (void)tok;
    if (host) { cur(b); b->cu.cuMemFreeHost(host); }
}

static int qok(int q) { return q >= 0 && q < NTSE_NQUEUES; }

static int cu_put(ntse_be *b, int q, const ntse_dmem *dst, uint64_t off,
                  const void *src, uint64_t n, void *tok)
{
    if (!qok(q) || off > dst->bytes || n > dst->bytes - off) return NTSE_E_INVAL;
    if (n == 0) return NTSE_OK;
    if (cur(b) != NTSE_OK) return NTSE_E_DEVICE;
    if (ck(b, b->cu.cuMemcpyHtoDAsync(dst->ptr + off, src, (size_t)n, b->q[q])) != NTSE_OK)
        return NTSE_E_DEVICE;
    if (!tok) return ck(b, b->cu.cuStreamSynchronize(b->q[q]));
    return NTSE_OK;
}

static int cu_get(ntse_be *b, int q, void *dst, const ntse_dmem *src,
                  uint64_t off, uint64_t n, void *tok)
{
    if (!qok(q) || off > src->bytes || n > src->bytes - off) return NTSE_E_INVAL;
    if (n == 0) return NTSE_OK;
    if (cur(b) != NTSE_OK) return NTSE_E_DEVICE;
    if (ck(b, b->cu.cuMemcpyDtoHAsync(dst, src->ptr + off, (size_t)n, b->q[q])) != NTSE_OK)
        return NTSE_E_DEVICE;
    if (!tok) return ck(b, b->cu.cuStreamSynchronize(b->q[q]));
    return NTSE_OK;
}

static int cu_zero(ntse_be *b, int q, const ntse_dmem *m)
{
    if (!qok(q)) return NTSE_E_INVAL;
    if (cur(b) != NTSE_OK) return NTSE_E_DEVICE;
    return ck(b, b->cu.cuMemsetD8Async(m->ptr, 0, (size_t)m->bytes, b->q[q]));
}

static int evok(int e) { return e >= 0 && e < NTSE_MAX_EVENTS; }

static int cu_ev_record(ntse_be *b, int q, int e)
{
    if (!qok(q) || !evok(e)) return NTSE_E_INVAL;
    if (cur(b) != NTSE_OK) return NTSE_E_DEVICE;
    return ck(b, b->cu.cuEventRecord(b->ev[e], b->q[q]));
}

static int cu_ev_wait(ntse_be *b, int q, int e)
{
    if (!qok(q) || !evok(e)) return NTSE_E_INVAL;
    if (cur(b) != NTSE_OK) return NTSE_E_DEVICE;
    return ck(b, b->cu.cuStreamWaitEvent(b->q[q], b->ev[e], 0));
}

static int cu_ev_sync(ntse_be *b, int e)
{
    if (!evok(e)) return NTSE_E_INVAL;
    if (cur(b) != NTSE_OK) return NTSE_E_DEVICE;
    return ck(b, b->cu.cuEventSynchronize(b->ev[e]));
}

static int cu_ev_query(ntse_be *b, int e)
{
    if (!evok(e)) return NTSE_E_INVAL;
    if (cur(b) != NTSE_OK) return NTSE_E_DEVICE;
    CUresult r = b->cu.cuEventQuery(b->ev[e]);
    if (r == CUDA_SUCCESS) return 1;
    if (r == CUDA_ERROR_NOT_READY) return 0;
    b->last = r;
    return NTSE_E_DEVICE;
}

static int cu_qsync(ntse_be *b, int q)
{
    if (!qok(q)) return NTSE_E_INVAL;
    if (cur(b) != NTSE_OK) return NTSE_E_DEVICE;
    return ck(b, b->cu.cuStreamSynchronize(b->q[q]));
}

static unsigned grid_for(uint64_t n, unsigned per)
{
    uint64_t g = (n + per - 1) / per;
    if (g < 1) g = 1;
    if (g > (1u << 20)) g = 1u << 20;   /* grid-stride loops cover the rest */
    return (unsigned)g;
}

static uint64_t P(const ntse_dref *r) { return r->m ? r->m->ptr + r->off : 0; }

static int cu_launch(ntse_be *b, int q, int kernel, const ntse_kargs *k)
{
    if (!qok(q) || kernel < 0 || kernel >= NTSE_K_COUNT
        || kernel == NTSE_K_LINEAR_BF16)
        return NTSE_E_INVAL;
    if (cur(b) != NTSE_OK) return NTSE_E_DEVICE;
    CUdeviceptr F = P(&k->F);
    CUdeviceptr bookT = P(&k->bookT), book = P(&k->book), idxT = P(&k->idxT),
                mnT = P(&k->mnT), scT = P(&k->scT), act = P(&k->act),
                x = P(&k->x), A = P(&k->A), T = P(&k->T), y = P(&k->y),
                W = P(&k->W);
    uint64_t nseg = k->nseg, cols = k->cols, rows = k->rows, nact = k->nact;
    unsigned B = k->block, K = k->entries, batch = k->batch;
    void *args[16];
    uint64_t n;
    unsigned block = 256, grid;
    switch (kernel) {
    case NTSE_K_SEGSUM: {
        void *a[] = { &x, &A, &nseg, &B, &batch, &cols };
        memcpy(args, a, sizeof a); n = (uint64_t)batch * nseg; break; }
    case NTSE_K_BUILD_EAGER: {
        void *a[] = { &bookT, &x, &T, &nseg, &B, &K, &batch, &cols };
        memcpy(args, a, sizeof a); n = (uint64_t)batch * nseg * K; break; }
    case NTSE_K_BUILD_ACTIVE: {
        void *a[] = { &bookT, &x, &T, &act, &nact, &nseg, &B, &K, &batch, &cols };
        memcpy(args, a, sizeof a); n = (uint64_t)batch * nact; break; }
    case NTSE_K_MARK: {
        void *a[] = { &idxT, &F, &rows, &nseg, &K };
        memcpy(args, a, sizeof a); n = rows * nseg; break; }
    case NTSE_K_BUILD_LAZY: {
        void *a[] = { &bookT, &x, &T, &F, &nseg, &B, &K, &batch, &cols };
        memcpy(args, a, sizeof a); n = (uint64_t)batch * nseg * K; break; }
    case NTSE_K_APPLY: {
        void *a[] = { &idxT, &mnT, &scT, &T, &A, &y, &rows, &nseg, &K, &batch };
        memcpy(args, a, sizeof a); n = (uint64_t)batch * rows; break; }
    case NTSE_K_APPLY_DIRECT: {
        void *a[] = { &book, &x, &idxT, &mnT, &scT, &A, &y, &rows, &nseg, &B,
                      &batch, &cols };
        memcpy(args, a, sizeof a); n = (uint64_t)batch * rows; break; }
    case NTSE_K_EXPAND: {
        void *a[] = { &book, &idxT, &mnT, &scT, &W, &rows, &cols, &B };
        memcpy(args, a, sizeof a); n = rows * cols; break; }
    case NTSE_K_GEMV: {
        void *a[] = { &W, &x, &y, &rows, &cols, &batch };
        memcpy(args, a, sizeof a); n = (uint64_t)batch * rows;
        grid = n > 65535 ? 65535 : (unsigned)n;
        return ck(b, b->cu.cuLaunchKernel(b->fn[kernel], grid, 1, 1, block, 1, 1,
                                          0, b->q[q], args, NULL)); }
    default:
        return NTSE_E_INVAL;
    }
    if (n == 0) return NTSE_OK;
    grid = grid_for(n, block);
    return ck(b, b->cu.cuLaunchKernel(b->fn[kernel], grid, 1, 1, block, 1, 1,
                                      0, b->q[q], args, NULL));
}

static int cu_linear(ntse_be *b, int q, ntse_dref W, ntse_dref x, ntse_dref y,
                     uint32_t out, uint32_t in)
{
    if (!qok(q) || out == 0 || in == 0) return NTSE_E_INVAL;
    if (cur(b) != NTSE_OK) return NTSE_E_DEVICE;
    CUdeviceptr pW = P(&W), px = P(&x), py = P(&y);
    unsigned o = out, i = in;
    void *args[] = { &pW, &px, &py, &o, &i };
    unsigned grid = out > 65535 ? 65535 : out;
    return ck(b, b->cu.cuLaunchKernel(b->fn[NTSE_K_LINEAR_BF16], grid, 1, 1,
                                      256, 1, 1, 0, b->q[q], args, NULL));
}

static const ntse_be_ops cu_ops = {
    "cuda", cu_destroy, cu_dalloc, cu_dfree, cu_halloc, cu_hfree, cu_put,
    cu_get, cu_zero, cu_ev_record, cu_ev_wait, cu_ev_sync, cu_ev_query,
    cu_qsync, cu_launch, cu_linear
};

int ntse_cuda_create(ntse_be **out, const ntse_be_ops **ops, ntse_be_info *info,
                     char *why, size_t wl)
{
    *out = NULL;
    struct ntse_be *b = (struct ntse_be *)calloc(1, sizeof *b);
    if (!b) { snprintf(why, wl, "out of host memory"); return NTSE_E_NOMEM; }
    char es[160];
    CUresult r;
    if (load_cu(b, why, wl) != 0) goto nodev;
    if ((r = b->cu.cuInit(0)) != CUDA_SUCCESS) {
        cu_errstr(b, r, es, sizeof es);
        snprintf(why, wl, "cuInit: %s", es);
        goto nodev;
    }
    int count = 0;
    if (b->cu.cuDeviceGetCount(&count) != CUDA_SUCCESS || count < 1) {
        snprintf(why, wl, "no CUDA device");
        goto nodev;
    }
    int ord = 0;
    const char *ev = getenv("NTSE_CUDA_DEVICE");
    if (ev && *ev) ord = atoi(ev);
    if (ord < 0 || ord >= count) { snprintf(why, wl, "NTSE_CUDA_DEVICE out of range"); goto nodev; }
    if (b->cu.cuDeviceGet(&b->dev, ord) != CUDA_SUCCESS) {
        snprintf(why, wl, "cuDeviceGet failed");
        goto nodev;
    }
    char name[200] = "";
    int major = 0, minor = 0, integ = 0, drv = 0;
    size_t total = 0, freeb = 0;
    if (b->cu.cuDeviceGetName(name, (int)sizeof name - 1, b->dev) != CUDA_SUCCESS
        || b->cu.cuDeviceGetAttribute(&major, CU_ATTR_CC_MAJOR, b->dev) != CUDA_SUCCESS
        || b->cu.cuDeviceGetAttribute(&minor, CU_ATTR_CC_MINOR, b->dev) != CUDA_SUCCESS
        || b->cu.cuDeviceGetAttribute(&b->mp_count, CU_ATTR_MP_COUNT, b->dev) != CUDA_SUCCESS
        || b->cu.cuDeviceGetAttribute(&integ, CU_ATTR_INTEGRATED, b->dev) != CUDA_SUCCESS
        || b->cu.cuDeviceTotalMem(&total, b->dev) != CUDA_SUCCESS) {
        snprintf(why, wl, "CUDA device query failed");
        goto nodev;
    }
    b->cu.cuDriverGetVersion(&drv);
    if ((r = b->cu.cuDevicePrimaryCtxRetain(&b->ctx, b->dev)) != CUDA_SUCCESS) {
        cu_errstr(b, r, es, sizeof es);
        snprintf(why, wl, "primary context: %s", es);
        goto nodev;
    }
    b->ctx_retained = 1;
    if (cur(b) != NTSE_OK) { snprintf(why, wl, "cuCtxSetCurrent failed"); goto nodev; }
    if (b->cu.cuMemGetInfo(&freeb, &total) != CUDA_SUCCESS) {
        snprintf(why, wl, "cuMemGetInfo failed");
        goto nodev;
    }
    for (int i = 0; i < NTSE_NQUEUES; i++)
        if (b->cu.cuStreamCreate(&b->q[i], CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS) {
            snprintf(why, wl, "cuStreamCreate failed");
            goto nodev;
        }
    for (int i = 0; i < NTSE_MAX_EVENTS; i++)
        if (b->cu.cuEventCreate(&b->ev[i], CU_EVENT_DISABLE_TIMING) != CUDA_SUCCESS) {
            snprintf(why, wl, "cuEventCreate failed");
            goto nodev;
        }
    if (load_rtc(b, why, wl) != 0) goto nodev;
    int rmaj = 0, rmin = 0;
    b->rtc.nvrtcVersion(&rmaj, &rmin);
    char *img = rtc_build(b, major, minor, 1, why, wl);
    int real = 1;
    if (!img) { img = rtc_build(b, major, minor, 0, why, wl); real = 0; }
    if (!img) goto nodev;
    r = b->cu.cuModuleLoadData(&b->mod, img);
    free(img);
    if (r != CUDA_SUCCESS) {
        cu_errstr(b, r, es, sizeof es);
        snprintf(why, wl, "cuModuleLoadData: %s", es);
        goto nodev;
    }
    for (int i = 0; i < NTSE_K_COUNT; i++) {
        int mt = 0;
        if (b->cu.cuModuleGetFunction(&b->fn[i], b->mod, kernel_names[i]) != CUDA_SUCCESS
            || b->cu.cuFuncGetAttribute(&mt, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                        b->fn[i]) != CUDA_SUCCESS
            || mt < 256) {
            snprintf(why, wl, "kernel %s missing or cannot run 256 threads",
                     kernel_names[i]);
            goto nodev;
        }
    }
    snprintf(info->device, sizeof info->device,
             "%s (sm_%d%d, %d SMs, driver CUDA %d.%d, NVRTC %d.%d, %s)",
             name, major, minor, b->mp_count, drv / 1000, (drv % 1000) / 10,
             rmaj, rmin, real ? "cubin" : "ptx-jit");
    info->total_mem = total;
    info->free_mem = freeb;
    info->max_alloc = total;
    info->integrated = integ;
    info->has_fp64 = 1;
    info->denorms = 1;    /* --ftz=false */
    *out = b;
    *ops = &cu_ops;
    return NTSE_OK;
nodev:
    cu_destroy(b);
    return NTSE_E_NODEV;
}
