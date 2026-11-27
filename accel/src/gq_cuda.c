/* gq_cuda.c - CUDA lane for nq2-quantize (research, scratch).
 *
 * Direction: JD. Co-implemented by Cortana on Elfie's quantizer.
 *
 * Same pattern as the engine's CUDA lane (accel_cuda.c): the CUDA
 * driver API (nvcuda.dll / libcuda.so.1) and NVRTC are loaded at
 * run time, nothing links against CUDA, and the build needs only
 * gcc. The kernels in gq_kernels_cuda.h are compiled once, for the
 * device's own architecture (cubin), or as PTX when NVRTC does not
 * know the architecture.
 *
 * Threads: gq_init() runs once on the main thread before any
 * worker starts. After that each calling thread binds the shared
 * primary context, creates its own stream, and keeps its own
 * device buffers in thread-local storage. No state is shared
 * between threads after init and nothing is ever locked.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "gq_cuda.h"
#include "gq_kernels_cuda.h"

typedef int CUresult;
typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;
typedef int nvrtcResult;
typedef struct _nvrtcProgram *nvrtcProgram;

#define CUDA_SUCCESS 0
#define CU_ATTR_CC_MAJOR 75
#define CU_ATTR_CC_MINOR 76
#define CU_STREAM_NON_BLOCKING 1

static struct {
    CUresult (*cuInit)(unsigned);
    CUresult (*cuDeviceGetCount)(int *);
    CUresult (*cuDeviceGet)(CUdevice *, int);
    CUresult (*cuDeviceGetName)(char *, int, CUdevice);
    CUresult (*cuDeviceGetAttribute)(int *, int, CUdevice);
    CUresult (*cuDevicePrimaryCtxRetain)(CUcontext *, CUdevice);
    CUresult (*cuDevicePrimaryCtxRelease)(CUdevice);
    CUresult (*cuCtxSetCurrent)(CUcontext);
    CUresult (*cuMemAlloc)(CUdeviceptr *, size_t);
    CUresult (*cuMemFree)(CUdeviceptr);
    CUresult (*cuMemcpyHtoDAsync)(CUdeviceptr, const void *, size_t, CUstream);
    CUresult (*cuMemcpyDtoHAsync)(void *, CUdeviceptr, size_t, CUstream);
    CUresult (*cuStreamCreate)(CUstream *, unsigned);
    CUresult (*cuStreamDestroy)(CUstream);
    CUresult (*cuStreamSynchronize)(CUstream);
    CUresult (*cuModuleLoadData)(CUmodule *, const void *);
    CUresult (*cuModuleUnload)(CUmodule);
    CUresult (*cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
    CUresult (*cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned,
                               unsigned, unsigned, unsigned, unsigned,
                               CUstream, void **, void **);
    CUresult (*cuGetErrorString)(CUresult, const char **);
} cu;

static struct {
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
} rtc;

/* Fixed widths with an unrolled kernel; index 0 is the runtime-d
 * kernel. */
#define GQ_NK 6
static const char *const k_names[GQ_NK] = {
    "k_assign_any", "k_assign_d1", "k_assign_d2", "k_assign_d4",
    "k_assign_d8", "k_assign_d16"
};
static const unsigned k_width[GQ_NK] = { 0, 1, 2, 4, 8, 16 };

/* Written only by gq_init and gq_shutdown, on the main thread,
 * while no worker runs. */
static void *g_lib_cu, *g_lib_rtc;
static CUdevice g_dev;
static CUcontext g_ctx;
static int g_ctx_retained;
static CUmodule g_mod;
static CUfunction g_fn[GQ_NK];
static char g_name[128];
static int g_ready;

/* ---------------- run-time loading ---------------- */

static void *dl_open(const char *name)
{
#ifdef _WIN32
    if (strchr(name, '\\') || strchr(name, '/'))
        return (void *)LoadLibraryExA(name, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    return (void *)LoadLibraryA(name);
#else
    return dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
}

static void *dl_sym(void *lib, const char *sym)
{
    if (!lib) return NULL;
#ifdef _WIN32
    FARPROC p = GetProcAddress((HMODULE)lib, sym);
    void *v;
    memcpy(&v, &p, sizeof v);
    return v;
#else
    return dlsym(lib, sym);
#endif
}

static void dl_close(void *lib)
{
    if (!lib) return;
#ifdef _WIN32
    FreeLibrary((HMODULE)lib);
#else
    dlclose(lib);
#endif
}

static int load_sym(void *lib, const char *name, void *slot)
{
    void *p = dl_sym(lib, name);
    if (!p) return -1;
    memcpy(slot, &p, sizeof p);
    return 0;
}

static int load_cu(char *why, size_t wl)
{
#ifdef _WIN32
    g_lib_cu = dl_open("nvcuda.dll");
#else
    g_lib_cu = dl_open("libcuda.so.1");
#endif
    if (!g_lib_cu) { snprintf(why, wl, "CUDA driver library not found"); return -1; }
    struct { const char *n; void *slot; } t[] = {
        { "cuInit", &cu.cuInit },
        { "cuDeviceGetCount", &cu.cuDeviceGetCount },
        { "cuDeviceGet", &cu.cuDeviceGet },
        { "cuDeviceGetName", &cu.cuDeviceGetName },
        { "cuDeviceGetAttribute", &cu.cuDeviceGetAttribute },
        { "cuDevicePrimaryCtxRetain", &cu.cuDevicePrimaryCtxRetain },
        { "cuDevicePrimaryCtxRelease_v2", &cu.cuDevicePrimaryCtxRelease },
        { "cuCtxSetCurrent", &cu.cuCtxSetCurrent },
        { "cuMemAlloc_v2", &cu.cuMemAlloc },
        { "cuMemFree_v2", &cu.cuMemFree },
        { "cuMemcpyHtoDAsync_v2", &cu.cuMemcpyHtoDAsync },
        { "cuMemcpyDtoHAsync_v2", &cu.cuMemcpyDtoHAsync },
        { "cuStreamCreate", &cu.cuStreamCreate },
        { "cuStreamDestroy_v2", &cu.cuStreamDestroy },
        { "cuStreamSynchronize", &cu.cuStreamSynchronize },
        { "cuModuleLoadData", &cu.cuModuleLoadData },
        { "cuModuleUnload", &cu.cuModuleUnload },
        { "cuModuleGetFunction", &cu.cuModuleGetFunction },
        { "cuLaunchKernel", &cu.cuLaunchKernel },
        { "cuGetErrorString", &cu.cuGetErrorString },
    };
    for (size_t i = 0; i < sizeof t / sizeof t[0]; i++)
        if (load_sym(g_lib_cu, t[i].n, t[i].slot) != 0) {
            snprintf(why, wl, "CUDA driver lacks %s", t[i].n);
            return -1;
        }
    return 0;
}

static void *open_nvrtc(void)
{
    const char *env = getenv("NTSE_NVRTC_DLL");
    if (env && *env) return dl_open(env);
#ifdef _WIN32
    static const char *names[] = { "nvrtc64_130_0.dll", "nvrtc64_120_0.dll",
                                   "nvrtc64_112_0.dll", "nvrtc64_111_0.dll" };
    const char *cp = getenv("CUDA_PATH");
    for (size_t i = 0; i < sizeof names / sizeof names[0]; i++) {
        void *h = NULL;
        if (cp && *cp) {
            char path[1024];
            snprintf(path, sizeof path, "%s\\bin\\%s", cp, names[i]);
            h = dl_open(path);
            if (!h) {
                snprintf(path, sizeof path, "%s\\bin\\x64\\%s", cp, names[i]);
                h = dl_open(path);
            }
        }
        if (!h) h = dl_open(names[i]);
        if (h) return h;
    }
    return NULL;
#else
    static const char *names[] = { "libnvrtc.so.13", "libnvrtc.so.12",
                                   "libnvrtc.so" };
    for (size_t i = 0; i < sizeof names / sizeof names[0]; i++) {
        void *h = dl_open(names[i]);
        if (h) return h;
    }
    return NULL;
#endif
}

static int load_rtc(char *why, size_t wl)
{
    g_lib_rtc = open_nvrtc();
    if (!g_lib_rtc) {
        snprintf(why, wl, "NVRTC library not found (set NTSE_NVRTC_DLL or CUDA_PATH)");
        return -1;
    }
    struct { const char *n; void *slot; int need; } t[] = {
        { "nvrtcCreateProgram", &rtc.nvrtcCreateProgram, 1 },
        { "nvrtcCompileProgram", &rtc.nvrtcCompileProgram, 1 },
        { "nvrtcGetProgramLogSize", &rtc.nvrtcGetProgramLogSize, 1 },
        { "nvrtcGetProgramLog", &rtc.nvrtcGetProgramLog, 1 },
        { "nvrtcGetPTXSize", &rtc.nvrtcGetPTXSize, 1 },
        { "nvrtcGetPTX", &rtc.nvrtcGetPTX, 1 },
        { "nvrtcGetCUBINSize", &rtc.nvrtcGetCUBINSize, 0 },
        { "nvrtcGetCUBIN", &rtc.nvrtcGetCUBIN, 0 },
        { "nvrtcDestroyProgram", &rtc.nvrtcDestroyProgram, 1 },
    };
    for (size_t i = 0; i < sizeof t / sizeof t[0]; i++)
        if (load_sym(g_lib_rtc, t[i].n, t[i].slot) != 0 && t[i].need) {
            snprintf(why, wl, "NVRTC lacks %s", t[i].n);
            return -1;
        }
    return 0;
}

static void cu_err(CUresult r, const char *what, char *why, size_t wl)
{
    const char *s = NULL;
    if (cu.cuGetErrorString && cu.cuGetErrorString(r, &s) == CUDA_SUCCESS && s)
        snprintf(why, wl, "%s: %s (%d)", what, s, r);
    else
        snprintf(why, wl, "%s: CUresult %d", what, r);
}

static char *rtc_build(int major, int minor, int arch_real, char *why, size_t wl)
{
    nvrtcProgram p = NULL;
    if (rtc.nvrtcCreateProgram(&p, gq_cuda_src, "gq_assign.cu", 0, NULL, NULL) != 0) {
        snprintf(why, wl, "nvrtcCreateProgram failed");
        return NULL;
    }
    char arch[64];
    snprintf(arch, sizeof arch, "--gpu-architecture=%s_%d%d",
             arch_real ? "sm" : "compute", major, minor);
    const char *opts[] = { arch, "--fmad=false", "--std=c++11",
                           "--ftz=false", "--prec-div=true", "--prec-sqrt=true" };
    if (rtc.nvrtcCompileProgram(p, 6, opts) != 0) {
        size_t ls = 0;
        rtc.nvrtcGetProgramLogSize(p, &ls);
        char log[600] = "";
        if (ls > 1 && ls < 1u << 20) {
            char *full = (char *)malloc(ls);
            if (full && rtc.nvrtcGetProgramLog(p, full) == 0)
                snprintf(log, sizeof log, "%s", full);
            free(full);
        }
        snprintf(why, wl, "NVRTC compile for %s failed: %.300s", arch + 19, log);
        rtc.nvrtcDestroyProgram(&p);
        return NULL;
    }
    size_t sz = 0;
    char *img = NULL;
    nvrtcResult gr;
    if (arch_real) {
        if (!rtc.nvrtcGetCUBINSize || !rtc.nvrtcGetCUBIN) {
            rtc.nvrtcDestroyProgram(&p);
            snprintf(why, wl, "NVRTC has no cubin output");
            return NULL;
        }
        gr = rtc.nvrtcGetCUBINSize(p, &sz);
        if (gr == 0 && sz > 0 && (img = (char *)malloc(sz)))
            gr = rtc.nvrtcGetCUBIN(p, img);
    } else {
        gr = rtc.nvrtcGetPTXSize(p, &sz);
        if (gr == 0 && sz > 0 && (img = (char *)malloc(sz)))
            gr = rtc.nvrtcGetPTX(p, img);
    }
    rtc.nvrtcDestroyProgram(&p);
    if (gr != 0 || !img) {
        free(img);
        snprintf(why, wl, "NVRTC output retrieval failed");
        return NULL;
    }
    return img;
}

int gq_init(int device, char *why, size_t wl)
{
    if (g_ready) return 0;
    if (load_cu(why, wl) != 0 || load_rtc(why, wl) != 0) { gq_shutdown(); return -1; }
    CUresult r;
    int n = 0;
    if ((r = cu.cuInit(0)) != CUDA_SUCCESS) { cu_err(r, "cuInit", why, wl); gq_shutdown(); return -1; }
    if ((r = cu.cuDeviceGetCount(&n)) != CUDA_SUCCESS || n <= device) {
        snprintf(why, wl, "no CUDA device %d (%d found)", device, n);
        gq_shutdown();
        return -1;
    }
    if ((r = cu.cuDeviceGet(&g_dev, device)) != CUDA_SUCCESS) { cu_err(r, "cuDeviceGet", why, wl); gq_shutdown(); return -1; }
    cu.cuDeviceGetName(g_name, (int)sizeof g_name - 1, g_dev);
    int major = 0, minor = 0;
    cu.cuDeviceGetAttribute(&major, CU_ATTR_CC_MAJOR, g_dev);
    cu.cuDeviceGetAttribute(&minor, CU_ATTR_CC_MINOR, g_dev);
    if ((r = cu.cuDevicePrimaryCtxRetain(&g_ctx, g_dev)) != CUDA_SUCCESS) { cu_err(r, "context", why, wl); gq_shutdown(); return -1; }
    g_ctx_retained = 1;
    if ((r = cu.cuCtxSetCurrent(g_ctx)) != CUDA_SUCCESS) { cu_err(r, "cuCtxSetCurrent", why, wl); gq_shutdown(); return -1; }
    char why2[512] = "";
    char *img = rtc_build(major, minor, 1, why2, sizeof why2);
    if (!img) img = rtc_build(major, minor, 0, why, wl);
    if (!img) { gq_shutdown(); return -1; }
    r = cu.cuModuleLoadData(&g_mod, img);
    free(img);
    if (r != CUDA_SUCCESS) { cu_err(r, "cuModuleLoadData", why, wl); gq_shutdown(); return -1; }
    for (int i = 0; i < GQ_NK; i++)
        if ((r = cu.cuModuleGetFunction(&g_fn[i], g_mod, k_names[i])) != CUDA_SUCCESS) {
            cu_err(r, k_names[i], why, wl);
            gq_shutdown();
            return -1;
        }
    g_ready = 1;
    return 0;
}

const char *gq_device_name(void) { return g_name; }

void gq_shutdown(void)
{
    gq_thread_release();
    if (g_mod) { cu.cuModuleUnload(g_mod); g_mod = NULL; }
    if (g_ctx_retained) { cu.cuDevicePrimaryCtxRelease(g_dev); g_ctx_retained = 0; }
    g_ctx = NULL;
    dl_close(g_lib_rtc); g_lib_rtc = NULL;
    dl_close(g_lib_cu); g_lib_cu = NULL;
    g_ready = 0;
}

/* ---------------- per-thread state ---------------- */

typedef struct {
    int bound;
    CUstream st;
    CUdeviceptr x, books, out, how;
    size_t x_cap, books_cap, out_cap, how_cap;
    uint64_t n;
    uint32_t stride;
    uint8_t *how_h;
    size_t how_h_cap;
    uint64_t rows, multi, full;
} GqThread;

static __thread GqThread t_gq;

static int bind_thread(void)
{
    GqThread *t = &t_gq;
    if (!g_ready) return -1;
    if (cu.cuCtxSetCurrent(g_ctx) != CUDA_SUCCESS) return -1;
    if (!t->bound) {
        if (cu.cuStreamCreate(&t->st, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS) return -1;
        t->bound = 1;
    }
    return 0;
}

static int grow(CUdeviceptr *p, size_t *cap, size_t want)
{
    if (want <= *cap && *p) return 0;
    if (*p) { cu.cuMemFree(*p); *p = 0; *cap = 0; }
    size_t sz = want + want / 4 + 256;
    if (cu.cuMemAlloc(p, sz) != CUDA_SUCCESS) { *p = 0; return -1; }
    *cap = sz;
    return 0;
}

int gq_set_rows(const float *x, uint64_t n, uint32_t stride)
{
    GqThread *t = &t_gq;
    if (bind_thread() != 0) return -1;
    size_t bytes = (size_t)n * stride * sizeof(float);
    if (grow(&t->x, &t->x_cap, bytes ? bytes : 4) != 0) return -1;
    if (bytes && cu.cuMemcpyHtoDAsync(t->x, x, bytes, t->st) != CUDA_SUCCESS) return -1;
    if (cu.cuStreamSynchronize(t->st) != CUDA_SUCCESS) return -1;
    t->n = n;
    t->stride = stride;
    return 0;
}

int gq_assign_rows(uint32_t nsub, uint32_t d, const float *books,
                   uint32_t entries, uint8_t *out)
{
    GqThread *t = &t_gq;
    if (bind_thread() != 0) return -1;
    if (nsub == 0 || d == 0 || entries == 0 || entries > 256) return -1;
    if ((uint64_t)nsub * d > t->stride) return -1;
    if (t->n == 0) return 0;
    if (nsub > 65535u) return -1;
    size_t bbytes = (size_t)nsub * entries * d * sizeof(float);
    size_t obytes = (size_t)t->n * nsub;
    if (grow(&t->books, &t->books_cap, bbytes) != 0) return -1;
    if (grow(&t->out, &t->out_cap, obytes) != 0) return -1;
    if (grow(&t->how, &t->how_cap, obytes) != 0) return -1;
    if (obytes > t->how_h_cap) {
        free(t->how_h);
        t->how_h = (uint8_t *)malloc(obytes);
        if (!t->how_h) { t->how_h_cap = 0; return -1; }
        t->how_h_cap = obytes;
    }
    if (cu.cuMemcpyHtoDAsync(t->books, books, bbytes, t->st) != CUDA_SUCCESS) return -1;

    /* Error bound of the fp32 screen; see gq_kernels_cuda.h. All
     * three values are exact in fp32 for d <= 4096. */
    double m = (double)d + 8.0;
    double p22 = 1.0, p140 = 1.0;
    for (int i = 0; i < 22; i++) p22 *= 0.5;
    for (int i = 0; i < 140; i++) p140 *= 0.5;
    float onep = (float)(1.0 + m * p22);
    float onem = (float)(1.0 - m * p22);
    float A = (float)(m * p140);

    int ki = 0;
    for (int i = 1; i < GQ_NK; i++) if (k_width[i] == d) ki = i;
    size_t shbytes = (size_t)entries * d * sizeof(float);
    int use_sh = shbytes <= 48u * 1024u;
    unsigned shm = use_sh ? (unsigned)shbytes : 0u;
    const unsigned tpb = 128;
    unsigned long long n = t->n;
    unsigned gx = (unsigned)((n + tpb - 1) / tpb);
    uint32_t stride = t->stride;
    void *args[] = { &t->x, &n, &stride, &nsub, &d, &t->books, &entries,
                     &use_sh, &onep, &onem, &A, &t->out, &t->how };
    if (cu.cuLaunchKernel(g_fn[ki], gx, nsub, 1, tpb, 1, 1, shm, t->st,
                          args, NULL) != CUDA_SUCCESS) return -1;
    if (cu.cuMemcpyDtoHAsync(out, t->out, obytes, t->st) != CUDA_SUCCESS) return -1;
    if (cu.cuMemcpyDtoHAsync(t->how_h, t->how, obytes, t->st) != CUDA_SUCCESS) return -1;
    if (cu.cuStreamSynchronize(t->st) != CUDA_SUCCESS) return -1;
    for (size_t i = 0; i < obytes; i++) {
        if (t->how_h[i] == 1) t->multi++;
        else if (t->how_h[i] == 2) t->full++;
    }
    t->rows += obytes;
    return 0;
}

void gq_thread_stats(uint64_t *rows, uint64_t *multi, uint64_t *full)
{
    if (rows) *rows = t_gq.rows;
    if (multi) *multi = t_gq.multi;
    if (full) *full = t_gq.full;
}

void gq_thread_release(void)
{
    GqThread *t = &t_gq;
    if (t->bound && g_ready && cu.cuCtxSetCurrent(g_ctx) == CUDA_SUCCESS) {
        cu.cuStreamSynchronize(t->st);
        if (t->x) cu.cuMemFree(t->x);
        if (t->books) cu.cuMemFree(t->books);
        if (t->out) cu.cuMemFree(t->out);
        if (t->how) cu.cuMemFree(t->how);
        cu.cuStreamDestroy(t->st);
    }
    free(t->how_h);
    memset(t, 0, sizeof *t);
}
