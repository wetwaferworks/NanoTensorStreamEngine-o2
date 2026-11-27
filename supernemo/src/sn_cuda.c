/* sn_cuda.c - see sn_cuda.h. Pure C99; the device code is a string compiled by NVRTC at run time. */
#include "sn_cuda.h"
#include "sn_pqk.h"
#include "../../accel/sn_stream_kernels.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
typedef HMODULE dl_t;
static dl_t dl_open(const char *n) {
    if (strchr(n, '\\') || strchr(n, '/'))
        return LoadLibraryExA(n, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    return LoadLibraryA(n);
}
static void *dl_sym(dl_t h, const char *n) { return (void *)GetProcAddress(h, n); }
static void dl_close(dl_t h) { if (h) FreeLibrary(h); }
#else
#include <dlfcn.h>
typedef void *dl_t;
static dl_t dl_open(const char *n) { return dlopen(n, RTLD_NOW); }
static void *dl_sym(dl_t h, const char *n) { return dlsym(h, n); }
static void dl_close(dl_t h) { if (h) dlclose(h); }
#endif

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext, *CUmodule, *CUfunction, *CUstream, *CUevent;
typedef unsigned long long CUdeviceptr;
typedef int nvrtcResult;
typedef void *nvrtcProgram;

static const char *KERNELS =
"typedef unsigned char u8; typedef unsigned short u16; typedef unsigned long long u64;\n"
"__device__ __forceinline__ float h2f(u16 h) { float f; asm(\"cvt.f32.f16 %0, %1;\" : \"=f\"(f) : \"h\"(h)); return f; }\n"
"__device__ __forceinline__ float block_sum(float v, float *red) {\n"
"  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);\n"
"  int w = threadIdx.x >> 5, l = threadIdx.x & 31;\n"
"  __syncthreads();\n"
"  if (l == 0) red[w] = v;\n"
"  __syncthreads();\n"
"  float s = 0.f; int nw = blockDim.x >> 5;\n"
"  for (int i = 0; i < nw; i++) s += red[i];\n"
"  return s;\n"
"}\n"
/* PQ4: index byte i of a tensor covers weights 4i..4i+3 of it; block b = w >> 8, sub-book = (w & 255) >> 2 */
"extern \"C\" __global__ void pq_rows(const u8 *idx, const u16 *mn, const u16 *sc, const float4 *book,\n"
"                                    const float *x, float *y, int cols, int rows, int rpb) {\n"
"  extern __shared__ float xs[];\n"
"  __shared__ float red[32];\n"
"  for (int i = threadIdx.x; i < cols; i += blockDim.x) xs[i] = x[i];\n"
"  __syncthreads();\n"
"  for (int j = 0; j < rpb; j++) {\n"
"    int row = blockIdx.x * rpb + j;\n"
"    if (row >= rows) break;\n"
"    u64 w0 = (u64)row * (u64)cols;\n"
"    float acc = 0.f;\n"
"    for (int g = threadIdx.x; g < (cols >> 2); g += blockDim.x) {\n"
"      u64 wi = w0 + 4ull * (u64)g;\n"
"      u64 b = wi >> 8; int sub = (int)((wi & 255ull) >> 2);\n"
"      float4 c = book[sub * 256 + idx[wi >> 2]];\n"
"      float4 xv = *(const float4 *)(xs + 4 * g);\n"
"      acc += h2f(mn[b]) * (xv.x + xv.y + xv.z + xv.w) + h2f(sc[b]) * (c.x * xv.x + c.y * xv.y + c.z * xv.z + c.w * xv.w);\n"
"    }\n"
"    float s = block_sum(acc, red);\n"
"    if (threadIdx.x == 0) y[row] = s;\n"
"  }\n"
"}\n"
/* DTC: table build then one thread per row. Every float and double operation mirrors sn_pqk.c operation for operation
   (compiled with --fmad=false), so the result is bit-identical to the CPU DTC kernel. */
"extern \"C\" __global__ void dtc_build(const float4 *book, const float *x, float *T, int ng, int gph) {\n"
"  int pg = blockIdx.x; int ph = pg / ng; int G = pg - ph * ng;\n"
"  int sub = (((ph * gph) >> 2) + G) & 63;\n"
"  float4 c = book[sub * 256 + threadIdx.x];\n"
"  float x0 = x[4 * G], x1 = x[4 * G + 1], x2 = x[4 * G + 2], x3 = x[4 * G + 3];\n"
"  T[(size_t)pg * 256 + threadIdx.x] = c.x * x0 + c.y * x1 + c.z * x2 + c.w * x3;\n"
"}\n"
"extern \"C\" __global__ void dtc_rows(const u8 *idx, const u16 *mn, const u16 *sc, const float *T, const double *P, float *y,\n"
"                                     int cols, int rows, int ng, int nph, int gph) {\n"
"  int row = blockIdx.x * blockDim.x + threadIdx.x;\n"
"  if (row >= rows) return;\n"
"  u64 flat0 = (u64)row * (u64)cols, flat1 = flat0 + (u64)cols;\n"
"  size_t phbase = 0;\n"
"  if (nph > 1) phbase = (size_t)((flat0 & 255ull) / (u64)gph) * (size_t)ng * 256;\n"
"  double acc = 0.0;\n"
"  for (u64 b = flat0 >> 8; b <= ((flat1 - 1) >> 8); b++) {\n"
"    u64 bf0 = b << 8;\n"
"    u64 cur = bf0 > flat0 ? bf0 : flat0;\n"
"    u64 hi = bf0 + 256 < flat1 ? bf0 + 256 : flat1;\n"
"    float m = h2f(mn[b]), s = h2f(sc[b]);\n"
"    const u8 *ix = idx + b * 64;\n"
"    int sub0 = (int)((cur - bf0) >> 2), sub1 = (int)((hi - bf0) >> 2);\n"
"    int col = (int)(cur - flat0), len = (int)(hi - cur);\n"
"    const float *Tp = T + phbase + (size_t)(col >> 2) * 256;\n"
"    int n = sub1 - sub0, k = 0;\n"
"    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;\n"
"    for (; k + 4 <= n; k += 4) {\n"
"      a0 += Tp[(size_t)k * 256 + ix[sub0 + k]];\n"
"      a1 += Tp[(size_t)(k + 1) * 256 + ix[sub0 + k + 1]];\n"
"      a2 += Tp[(size_t)(k + 2) * 256 + ix[sub0 + k + 2]];\n"
"      a3 += Tp[(size_t)(k + 3) * 256 + ix[sub0 + k + 3]];\n"
"    }\n"
"    if (k < n) { a0 += Tp[(size_t)k * 256 + ix[sub0 + k]]; k++; }\n"
"    if (k < n) { a1 += Tp[(size_t)k * 256 + ix[sub0 + k]]; k++; }\n"
"    if (k < n) { a2 += Tp[(size_t)k * 256 + ix[sub0 + k]]; }\n"
"    float dotc = (a0 + a1) + (a2 + a3);\n"
"    double pd = P[col + len] - P[col];\n"
"    acc = acc + ((double)m * pd + (double)s * (double)dotc);\n"
"  }\n"
"  y[row] = (float)acc;\n"
"}\n"
"extern \"C\" __global__ void raw_rows(const u16 *w, const float *x, float *y, int cols, int rows, int rpb) {\n"
"  extern __shared__ float xs[];\n"
"  __shared__ float red[32];\n"
"  for (int i = threadIdx.x; i < cols; i += blockDim.x) xs[i] = x[i];\n"
"  __syncthreads();\n"
"  for (int j = 0; j < rpb; j++) {\n"
"    int row = blockIdx.x * rpb + j;\n"
"    if (row >= rows) break;\n"
"    const u16 *wr = w + (u64)row * (u64)cols;\n"
"    float acc = 0.f;\n"
"    for (int k = threadIdx.x; k < cols; k += blockDim.x) acc += h2f(wr[k]) * xs[k];\n"
"    float s = block_sum(acc, red);\n"
"    if (threadIdx.x == 0) y[row] = s;\n"
"  }\n"
"}\n";

typedef struct { CUdeviceptr idx, mn, sc, book; long long bytes; int kind; int cache; } dres;   /* kind 0 none, 1 pq, 2 raw */

struct sn_cuda {
    dl_t drv, rtc;
    CUresult (*cuInit)(unsigned);
    CUresult (*cuDeviceGet)(CUdevice *, int);
    CUresult (*cuDeviceGetName)(char *, int, CUdevice);
    CUresult (*cuDeviceGetAttribute)(int *, int, CUdevice);
    CUresult (*cuMemGetInfo)(size_t *, size_t *);
    CUresult (*cuDevicePrimaryCtxRetain)(CUcontext *, CUdevice);
    CUresult (*cuCtxSetCurrent)(CUcontext);
    CUresult (*cuModuleLoadData)(CUmodule *, const void *);
    CUresult (*cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
    CUresult (*cuMemAlloc)(CUdeviceptr *, size_t);
    CUresult (*cuMemFree)(CUdeviceptr);
    CUresult (*cuMemcpyHtoD)(CUdeviceptr, const void *, size_t);
    CUresult (*cuMemcpyDtoH)(void *, CUdeviceptr, size_t);
    CUresult (*cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, CUstream, void **, void **);
    CUresult (*cuDevicePrimaryCtxRelease)(CUdevice);
    /* expert stream (optional: if the driver lacks one of these, the stream lane stays off) */
    CUresult (*cuStreamCreate)(CUstream *, unsigned);
    CUresult (*cuStreamDestroy)(CUstream);
    CUresult (*cuStreamSynchronize)(CUstream);
    CUresult (*cuStreamWaitEvent)(CUstream, CUevent, unsigned);
    CUresult (*cuEventCreate)(CUevent *, unsigned);
    CUresult (*cuEventRecord)(CUevent, CUstream);
    CUresult (*cuEventSynchronize)(CUevent);
    CUresult (*cuEventDestroy)(CUevent);
    CUresult (*cuMemHostAlloc)(void **, size_t, unsigned);
    CUresult (*cuMemFreeHost)(void *);
    CUresult (*cuMemcpyHtoDAsync)(CUdeviceptr, const void *, size_t, CUstream);
    CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction f_pq, f_raw, f_dtcb, f_dtcr;
    CUfunction f_xexp, f_xraw, f_xgemm, f_xrelu;
    char name[128];
    long long budget, used; int count;
    CUdeviceptr dx, dy; size_t dx_n, dy_n;
    CUdeviceptr dT, dP; size_t dT_n, dP_n;      /* DTC table and prefix-sum scratch */
    long long cache_budget, cache_used; int cache_count;
    dres *res; int nres;
    struct xs_lane *xs;                         /* expert stream, NULL until sn_cuda_xs_open */
};

static void xs_free(sn_cuda *c);

#define LOADSYM(field, sym) do { c->field = (void *)dl_sym(c->drv, sym); if (!c->field) { snprintf(why, (size_t)whylen, "the driver has no %s", sym); goto fail; } } while (0)

static dl_t open_nvrtc(void)
{
    const char *env = getenv("SN_NVRTC");
    if (env && *env) { dl_t h = dl_open(env); if (h) return h; }
#if defined(_WIN32)
    const char *cp = getenv("CUDA_PATH");
    if (cp && *cp) {
        char pat[1024]; snprintf(pat, sizeof pat, "%s\\bin\\nvrtc64_*_0.dll", cp);
        WIN32_FIND_DATAA fd; HANDLE fh = FindFirstFileA(pat, &fd);
        if (fh != INVALID_HANDLE_VALUE) {
            char full[1100]; snprintf(full, sizeof full, "%s\\bin\\%s", cp, fd.cFileName); FindClose(fh);
            dl_t h = dl_open(full); if (h) return h;
        }
    }
    static const char *names[] = { "nvrtc64_130_0.dll", "nvrtc64_120_0.dll", "nvrtc64_112_0.dll", NULL };
#else
    static const char *names[] = { "libnvrtc.so", "libnvrtc.so.12", "libnvrtc.so.13", NULL };
#endif
    for (int i = 0; names[i]; i++) { dl_t h = dl_open(names[i]); if (h) return h; }
    return NULL;
}

void sn_cuda_close(sn_cuda *c)
{
    if (!c) return;
    if (c->res) for (int i = 0; i < c->nres; i++) if (c->res[i].kind && c->cuMemFree) {
        c->cuMemFree(c->res[i].idx);
        if (c->res[i].kind == 1) { c->cuMemFree(c->res[i].mn); c->cuMemFree(c->res[i].sc); c->cuMemFree(c->res[i].book); }
    }
    if (c->dx && c->cuMemFree) c->cuMemFree(c->dx);
    if (c->dy && c->cuMemFree) c->cuMemFree(c->dy);
    if (c->dT && c->cuMemFree) c->cuMemFree(c->dT);
    if (c->dP && c->cuMemFree) c->cuMemFree(c->dP);
    xs_free(c);
    if (c->cuDevicePrimaryCtxRelease && c->ctx) c->cuDevicePrimaryCtxRelease(c->dev);
    dl_close(c->rtc);
    dl_close(c->drv);
    free(c->res); free(c);
}

const char *sn_cuda_name(const sn_cuda *c) { return c->name; }
long long sn_cuda_used(const sn_cuda *c) { return c->used; }
int sn_cuda_count(const sn_cuda *c) { return c->count; }

sn_cuda *sn_cuda_open(long long budget_bytes, int pack_tensors, char *why, int whylen)
{
    sn_cuda *c = (sn_cuda *)calloc(1, sizeof *c);
    if (!c) { snprintf(why, (size_t)whylen, "out of memory"); return NULL; }
    c->drv = dl_open(
#if defined(_WIN32)
        "nvcuda.dll"
#else
        "libcuda.so.1"
#endif
    );
    if (!c->drv) { snprintf(why, (size_t)whylen, "no NVIDIA driver (nvcuda) found"); goto fail; }
    LOADSYM(cuInit, "cuInit"); LOADSYM(cuDeviceGet, "cuDeviceGet"); LOADSYM(cuDeviceGetName, "cuDeviceGetName");
    LOADSYM(cuDeviceGetAttribute, "cuDeviceGetAttribute"); LOADSYM(cuMemGetInfo, "cuMemGetInfo_v2");
    LOADSYM(cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain"); LOADSYM(cuCtxSetCurrent, "cuCtxSetCurrent");
    LOADSYM(cuModuleLoadData, "cuModuleLoadData"); LOADSYM(cuModuleGetFunction, "cuModuleGetFunction");
    LOADSYM(cuMemAlloc, "cuMemAlloc_v2"); LOADSYM(cuMemFree, "cuMemFree_v2");
    LOADSYM(cuMemcpyHtoD, "cuMemcpyHtoD_v2"); LOADSYM(cuMemcpyDtoH, "cuMemcpyDtoH_v2");
    LOADSYM(cuLaunchKernel, "cuLaunchKernel"); LOADSYM(cuDevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease_v2");
    c->cuStreamCreate = (void *)dl_sym(c->drv, "cuStreamCreate"); c->cuStreamDestroy = (void *)dl_sym(c->drv, "cuStreamDestroy_v2");
    c->cuStreamSynchronize = (void *)dl_sym(c->drv, "cuStreamSynchronize"); c->cuStreamWaitEvent = (void *)dl_sym(c->drv, "cuStreamWaitEvent");
    c->cuEventCreate = (void *)dl_sym(c->drv, "cuEventCreate"); c->cuEventRecord = (void *)dl_sym(c->drv, "cuEventRecord");
    c->cuEventSynchronize = (void *)dl_sym(c->drv, "cuEventSynchronize"); c->cuEventDestroy = (void *)dl_sym(c->drv, "cuEventDestroy_v2");
    c->cuMemHostAlloc = (void *)dl_sym(c->drv, "cuMemHostAlloc"); c->cuMemFreeHost = (void *)dl_sym(c->drv, "cuMemFreeHost");
    c->cuMemcpyHtoDAsync = (void *)dl_sym(c->drv, "cuMemcpyHtoDAsync_v2");
    if (c->cuInit(0) != 0 || c->cuDeviceGet(&c->dev, 0) != 0) { snprintf(why, (size_t)whylen, "no CUDA device"); goto fail; }
    c->cuDeviceGetName(c->name, (int)sizeof c->name, c->dev);
    int major = 0, minor = 0;
    c->cuDeviceGetAttribute(&major, 75, c->dev); c->cuDeviceGetAttribute(&minor, 76, c->dev);
    if (c->cuDevicePrimaryCtxRetain(&c->ctx, c->dev) != 0 || c->cuCtxSetCurrent(c->ctx) != 0) { snprintf(why, (size_t)whylen, "cannot create a CUDA context"); goto fail; }

    c->rtc = open_nvrtc();
    if (!c->rtc) { snprintf(why, (size_t)whylen, "NVRTC not found (set CUDA_PATH to the CUDA toolkit, or SN_NVRTC to nvrtc64_*.dll)"); goto fail; }
    nvrtcResult (*nvrtcCreateProgram)(nvrtcProgram *, const char *, const char *, int, const char *const *, const char *const *) = dl_sym(c->rtc, "nvrtcCreateProgram");
    nvrtcResult (*nvrtcCompileProgram)(nvrtcProgram, int, const char *const *) = dl_sym(c->rtc, "nvrtcCompileProgram");
    nvrtcResult (*nvrtcGetCUBINSize)(nvrtcProgram, size_t *) = dl_sym(c->rtc, "nvrtcGetCUBINSize");
    nvrtcResult (*nvrtcGetCUBIN)(nvrtcProgram, char *) = dl_sym(c->rtc, "nvrtcGetCUBIN");
    nvrtcResult (*nvrtcGetProgramLogSize)(nvrtcProgram, size_t *) = dl_sym(c->rtc, "nvrtcGetProgramLogSize");
    nvrtcResult (*nvrtcGetProgramLog)(nvrtcProgram, char *) = dl_sym(c->rtc, "nvrtcGetProgramLog");
    nvrtcResult (*nvrtcDestroyProgram)(nvrtcProgram *) = dl_sym(c->rtc, "nvrtcDestroyProgram");
    if (!nvrtcCreateProgram || !nvrtcCompileProgram || !nvrtcGetCUBINSize || !nvrtcGetCUBIN || !nvrtcGetProgramLogSize || !nvrtcGetProgramLog || !nvrtcDestroyProgram) {
        snprintf(why, (size_t)whylen, "NVRTC is missing entry points"); goto fail;
    }
    char arch[32]; snprintf(arch, sizeof arch, "--gpu-architecture=sm_%d%d", major, minor);
    const char *opts[] = { arch, "--std=c++14", "--fmad=false" };
    nvrtcProgram prog = NULL;
    size_t kl0 = strlen(KERNELS), kl1 = strlen(SN_STREAM_KERNELS);
    char *src = (char *)malloc(kl0 + kl1 + 1);
    if (!src) { snprintf(why, (size_t)whylen, "out of memory"); goto fail; }
    memcpy(src, KERNELS, kl0); memcpy(src + kl0, SN_STREAM_KERNELS, kl1 + 1);
    nvrtcResult cr = nvrtcCreateProgram(&prog, src, "sn_kernels.cu", 0, NULL, NULL);
    free(src);
    if (cr != 0) { snprintf(why, (size_t)whylen, "NVRTC cannot create the program"); goto fail; }
    if (nvrtcCompileProgram(prog, 3, opts) != 0) {
        size_t ls = 0; nvrtcGetProgramLogSize(prog, &ls);
        char *lg = (char *)calloc(ls + 1, 1);
        if (lg) { nvrtcGetProgramLog(prog, lg); snprintf(why, (size_t)whylen, "NVRTC compile failed: %.300s", lg); free(lg); }
        nvrtcDestroyProgram(&prog); goto fail;
    }
    size_t cs = 0; nvrtcGetCUBINSize(prog, &cs);
    char *cubin = (char *)malloc(cs);
    if (!cubin || nvrtcGetCUBIN(prog, cubin) != 0) { free(cubin); nvrtcDestroyProgram(&prog); snprintf(why, (size_t)whylen, "NVRTC gave no code"); goto fail; }
    nvrtcDestroyProgram(&prog);
    CUresult mr = c->cuModuleLoadData(&c->mod, cubin);
    free(cubin);
    if (mr != 0 || c->cuModuleGetFunction(&c->f_pq, c->mod, "pq_rows") != 0 || c->cuModuleGetFunction(&c->f_raw, c->mod, "raw_rows") != 0
        || c->cuModuleGetFunction(&c->f_dtcb, c->mod, "dtc_build") != 0 || c->cuModuleGetFunction(&c->f_dtcr, c->mod, "dtc_rows") != 0
        || c->cuModuleGetFunction(&c->f_xexp, c->mod, "xs_expand_pq") != 0 || c->cuModuleGetFunction(&c->f_xraw, c->mod, "xs_expand_raw") != 0
        || c->cuModuleGetFunction(&c->f_xgemm, c->mod, "xs_gemm") != 0 || c->cuModuleGetFunction(&c->f_xrelu, c->mod, "xs_relu2_prefix") != 0) {
        snprintf(why, (size_t)whylen, "the GPU would not load the kernels (error %d)", mr); goto fail;
    }
    c->dx_n = 32768; c->dy_n = 140000;
    if (c->cuMemAlloc(&c->dx, c->dx_n * 4) != 0 || c->cuMemAlloc(&c->dy, c->dy_n * 4) != 0) { snprintf(why, (size_t)whylen, "no VRAM for the work buffers"); goto fail; }
    c->dT_n = 2u << 20; c->dP_n = 16400;
    if (c->cuMemAlloc(&c->dT, c->dT_n * 4) != 0 || c->cuMemAlloc(&c->dP, c->dP_n * 8) != 0) { snprintf(why, (size_t)whylen, "no VRAM for the DTC buffers"); goto fail; }
    c->budget = budget_bytes; c->nres = pack_tensors;
    c->res = (dres *)calloc((size_t)(pack_tensors > 0 ? pack_tensors : 1), sizeof(dres));
    if (!c->res) { snprintf(why, (size_t)whylen, "out of memory"); goto fail; }
    return c;
fail:
    sn_cuda_close(c);
    return NULL;
}

static int seek64(FILE *f, uint64_t off)
{
#if defined(_WIN32)
    return _fseeki64(f, (long long)off, SEEK_SET);
#else
    return fseeko(f, (off_t)off, SEEK_SET);
#endif
}

/* stream n bytes from the file at the current position into device memory in 32 MB slabs */
static int upload_stream(sn_cuda *c, FILE *f, CUdeviceptr dst, uint64_t n)
{
    enum { SLAB = 32 << 20 };
    unsigned char *buf = (unsigned char *)malloc(n < SLAB ? (size_t)n : SLAB);
    if (!buf) return -1;
    int rc = 0;
    for (uint64_t done = 0; done < n && !rc; ) {
        size_t k = n - done < SLAB ? (size_t)(n - done) : SLAB;
        if (fread(buf, 1, k, f) != k || c->cuMemcpyHtoD(dst + done, buf, k) != 0) rc = -1;
        done += k;
    }
    free(buf);
    return rc;
}

int sn_cuda_resident(const sn_cuda *c, const sn_pack *p, const sn_tensor *t)
{
    long i = (long)(t - p->t);
    return i >= 0 && i < c->nres && c->res[i].kind != 0;
}

int sn_cuda_load(sn_cuda *c, const sn_pack *p, sn_rd *rd, const sn_tensor *t)
{
    long ti = (long)(t - p->t);
    if (ti < 0 || ti >= c->nres) return -1;
    if (c->res[ti].kind) return 0;
    uint64_t B = p->block, nsub = p->nsub;
    long long need = t->raw ? (long long)(t->count * 2) : (long long)(t->nblocks * (nsub + 4) + p->book_floats * 4);
    if (c->used + need > c->budget) return -1;
    size_t fr = 0, tot = 0;
    if (c->cuMemGetInfo(&fr, &tot) == 0 && (long long)fr < need + (64ll << 20)) return -1;
    dres r; memset(&r, 0, sizeof r);
    if (t->raw) {
        r.kind = 2; r.bytes = need;
        if (c->cuMemAlloc(&r.idx, (size_t)need) != 0) return -1;
        if (seek64(rd->fs, t->raw_off) != 0 || upload_stream(c, rd->fs, r.idx, t->count * 2) != 0) { c->cuMemFree(r.idx); return -1; }
    } else {
        r.kind = 1; r.bytes = need;
        uint64_t nidx = t->nblocks * nsub, nh = t->nblocks * 2;
        float *book = (float *)malloc(p->book_floats * 4);
        if (!book || sn_load_book(p, rd, t, book) != 0) { free(book); return -1; }
        int ok = c->cuMemAlloc(&r.idx, (size_t)nidx) == 0 && c->cuMemAlloc(&r.mn, (size_t)nh) == 0 && c->cuMemAlloc(&r.sc, (size_t)nh) == 0 && c->cuMemAlloc(&r.book, p->book_floats * 4) == 0;
        if (ok) ok = c->cuMemcpyHtoD(r.book, book, p->book_floats * 4) == 0;
        if (ok) ok = seek64(rd->fs, p->body_off + t->block0 * nsub) == 0 && upload_stream(c, rd->fs, r.idx, nidx) == 0;
        if (ok) ok = seek64(rd->fs, p->body_off + p->tot_blocks * nsub + t->block0 * 2) == 0 && upload_stream(c, rd->fs, r.mn, nh) == 0;
        if (ok) ok = seek64(rd->fs, p->body_off + p->tot_blocks * (nsub + 2ull) + t->block0 * 2) == 0 && upload_stream(c, rd->fs, r.sc, nh) == 0;
        free(book);
        if (!ok) { if (r.idx) c->cuMemFree(r.idx); if (r.mn) c->cuMemFree(r.mn); if (r.sc) c->cuMemFree(r.sc); if (r.book) c->cuMemFree(r.book); return -1; }
        (void)B;
    }
    c->res[ti] = r; c->used += r.bytes; c->count++;
    return 0;
}

int sn_cuda_matvec(sn_cuda *c, const sn_pack *p, const sn_tensor *t, int cols, int rows, const float *x, float *y)
{
    long ti = (long)(t - p->t);
    if (ti < 0 || ti >= c->nres || !c->res[ti].kind) return -1;
    if ((uint64_t)cols * (uint64_t)rows != t->count || (size_t)cols > c->dx_n || (size_t)rows > c->dy_n || cols * 4 > 48 * 1024 || (cols & 3)) return -1;
    const dres *r = &c->res[ti];
    int rpb = rows >= 8192 ? 8 : 1, c_ = cols, r_ = rows;
    unsigned grid = (unsigned)((rows + rpb - 1) / rpb);
    if (c->cuMemcpyHtoD(c->dx, x, (size_t)cols * 4) != 0) return -1;
    CUresult rc;
    if (r->kind == 1) {
        void *args[] = { (void *)&r->idx, (void *)&r->mn, (void *)&r->sc, (void *)&r->book, &c->dx, &c->dy, &c_, &r_, &rpb };
        rc = c->cuLaunchKernel(c->f_pq, grid, 1, 1, 128, 1, 1, (unsigned)cols * 4, NULL, args, NULL);
    } else {
        void *args[] = { (void *)&r->idx, &c->dx, &c->dy, &c_, &r_, &rpb };
        rc = c->cuLaunchKernel(c->f_raw, grid, 1, 1, 128, 1, 1, (unsigned)cols * 4, NULL, args, NULL);
    }
    if (rc != 0) return -1;
    if (c->cuMemcpyDtoH(y, c->dy, (size_t)rows * 4) != 0) return -1;
    return 0;
}

/* ------------------------------------------------------------------ DTC and the hot-expert GPU tier */

int sn_cuda_dtc_rows(sn_cuda *c, const sn_pack *p, const sn_tensor *t, int cols, int rows, const float *x, const double *P, float *y)
{
    long ti = (long)(t - p->t);
    int nph = 0, ng = 0;
    if (ti < 0 || ti >= c->nres || c->res[ti].kind != 1) return -1;
    if (p->pq_d != 4 || p->block != 256 || p->entries != 256 || p->nsub != 64 || sn_dtc_geom(p, cols, &nph, &ng) != 0) return -1;
    if ((uint64_t)cols * (uint64_t)rows != t->count || (size_t)cols > c->dx_n || (size_t)rows > c->dy_n || (size_t)cols + 1 > c->dP_n) return -1;
    if ((size_t)nph * (size_t)ng * 256 > c->dT_n) return -1;
    const dres *r = &c->res[ti];
    int gph = 256 / nph, c_ = cols, r_ = rows, ng_ = ng, nph_ = nph;
    if (c->cuMemcpyHtoD(c->dx, x, (size_t)cols * 4) != 0 || c->cuMemcpyHtoD(c->dP, P, ((size_t)cols + 1) * 8) != 0) return -1;
    void *a1[] = { (void *)&r->book, &c->dx, &c->dT, &ng_, &gph };
    if (c->cuLaunchKernel(c->f_dtcb, (unsigned)(nph * ng), 1, 1, 256, 1, 1, 0, NULL, a1, NULL) != 0) return -1;
    void *a2[] = { (void *)&r->idx, (void *)&r->mn, (void *)&r->sc, &c->dT, &c->dP, &c->dy, &c_, &r_, &ng_, &nph_, &gph };
    if (c->cuLaunchKernel(c->f_dtcr, (unsigned)((rows + 127) / 128), 1, 1, 128, 1, 1, 0, NULL, a2, NULL) != 0) return -1;
    return c->cuMemcpyDtoH(y, c->dy, (size_t)rows * 4) == 0 ? 0 : -1;
}

void sn_cuda_set_cache_budget(sn_cuda *c, long long bytes) { c->cache_budget = bytes; }
long long sn_cuda_cache_used(const sn_cuda *c) { return c->cache_used; }
int sn_cuda_cache_count(const sn_cuda *c) { return c->cache_count; }

int sn_cuda_cache_put(sn_cuda *c, const sn_pack *p, const sn_tensor *t, const sn_pq_res *res)
{
    long ti = (long)(t - p->t);
    if (ti < 0 || ti >= c->nres || t->raw || p->nsub != 64) return -1;
    if (c->res[ti].kind) return 0;
    long long need = (long long)res->bytes;
    if (c->cache_used + need > c->cache_budget) return -1;
    size_t fr = 0, tot = 0;
    if (c->cuMemGetInfo(&fr, &tot) == 0 && (long long)fr < need + (96ll << 20)) return -1;
    dres r; memset(&r, 0, sizeof r);
    r.kind = 1; r.cache = 1; r.bytes = need;
    uint64_t nidx = res->nblocks * p->nsub, nh = res->nblocks * 2;
    int ok = c->cuMemAlloc(&r.idx, (size_t)nidx) == 0 && c->cuMemAlloc(&r.mn, (size_t)nh) == 0 && c->cuMemAlloc(&r.sc, (size_t)nh) == 0 && c->cuMemAlloc(&r.book, p->book_floats * 4) == 0;
    if (ok) ok = c->cuMemcpyHtoD(r.idx, res->idx, (size_t)nidx) == 0 && c->cuMemcpyHtoD(r.mn, res->mn, (size_t)nh) == 0
              && c->cuMemcpyHtoD(r.sc, res->sc, (size_t)nh) == 0 && c->cuMemcpyHtoD(r.book, res->book, p->book_floats * 4) == 0;
    if (!ok) { if (r.idx) c->cuMemFree(r.idx); if (r.mn) c->cuMemFree(r.mn); if (r.sc) c->cuMemFree(r.sc); if (r.book) c->cuMemFree(r.book); return -1; }
    c->res[ti] = r; c->cache_used += r.bytes; c->cache_count++;
    return 0;
}

void sn_cuda_cache_drop(sn_cuda *c, const sn_pack *p, const sn_tensor *t)
{
    long ti = (long)(t - p->t);
    if (ti < 0 || ti >= c->nres || !c->res[ti].kind || !c->res[ti].cache) return;
    dres *r = &c->res[ti];
    c->cuMemFree(r->idx); c->cuMemFree(r->mn); c->cuMemFree(r->sc); c->cuMemFree(r->book);
    c->cache_used -= r->bytes; c->cache_count--;
    memset(r, 0, sizeof *r);
}

/* ------------------------------------------------------------------ expert stream (JD's standing CUDA design, 2026-11-27)
 *
 * Compressed experts travel to the card and are expanded and multiplied there; their VRAM slots are reused for the next
 * token. Layout of one slot (and of one pinned staging buffer, byte for byte): the up matrix, then the down matrix, each as
 * index bytes | fp16 minimums | fp16 scales | book, every part on a 256-byte boundary. One slot is one expert pair.
 *
 * Two CUDA streams: "copy" moves staging buffers into slots, "comp" runs the kernels. Per slot, ev_ready (copy finished) and
 * ev_free (last kernel reading it finished) order the two; per staging buffer, ev_staged says the host may refill it. Every
 * call happens on the engine's main thread; the lanes only fill disjoint staging buffers. No locks anywhere. */

#define XS_AL(x) (((x) + 255u) & ~(size_t)255u)

typedef struct xs_lane {
    int Lt, I, K, nmax, simd, nkeys;
    uint64_t nb[2];                  /* blocks of the up / down matrix */
    size_t off[2][4];                /* idx, mn, sc, book offsets inside a slot */
    size_t slot_bytes;
    int nslots; CUdeviceptr slots;
    int *slot_key; uint64_t *slot_tick; uint32_t *slot_pin; int *key_slot;
    CUevent *ev_ready, *ev_free;
    int nstage, chunk; unsigned char *stage; CUevent *ev_staged;
    CUstream s_copy, s_comp;
    CUdeviceptr dLat, dPL, dEO, dU, dPmid, dT, dW, dTok, dOidx, dX, dY;
    size_t dT_n, dW_n, dX_n, dY_n;   /* floats */
    uint64_t tick; uint32_t pin;
    uint32_t *key_cnt; uint64_t uses, age_every; int policy;   /* policy 1: keep hot experts (use counts, halved now and then) */
    int nph[2], ng[2];
    sn_xs_info info;
} xs_lane;

static void xs_free(sn_cuda *c)
{
    xs_lane *x = c->xs;
    if (!x) return;
    if (x->s_comp && c->cuStreamSynchronize) c->cuStreamSynchronize(x->s_comp);
    if (x->s_copy && c->cuStreamSynchronize) c->cuStreamSynchronize(x->s_copy);
    for (int i = 0; i < x->nslots; i++) {
        if (x->ev_ready && x->ev_ready[i]) c->cuEventDestroy(x->ev_ready[i]);
        if (x->ev_free && x->ev_free[i]) c->cuEventDestroy(x->ev_free[i]);
    }
    for (int i = 0; i < x->nstage; i++) if (x->ev_staged && x->ev_staged[i]) c->cuEventDestroy(x->ev_staged[i]);
    CUdeviceptr *bufs[] = { &x->slots, &x->dLat, &x->dPL, &x->dEO, &x->dU, &x->dPmid, &x->dT, &x->dW, &x->dTok, &x->dOidx, &x->dX, &x->dY };
    for (size_t i = 0; i < sizeof bufs / sizeof bufs[0]; i++) if (*bufs[i]) c->cuMemFree(*bufs[i]);
    if (x->stage) c->cuMemFreeHost(x->stage);
    if (x->s_copy) c->cuStreamDestroy(x->s_copy);
    if (x->s_comp) c->cuStreamDestroy(x->s_comp);
    free(x->slot_key); free(x->slot_tick); free(x->slot_pin); free(x->key_slot); free(x->key_cnt);
    free(x->ev_ready); free(x->ev_free); free(x->ev_staged);
    free(x); c->xs = NULL;
}

/* The engine's hardware calibration (written by the Nemo engine's lane probe, nemo-calib.txt): which lane it selected for
 * this machine and how much VRAM was free when it measured. 1 = found for this GPU, 0 = no file / another GPU. */
static int xs_read_calib(const sn_cuda *c, long long *free_bytes, char *lane, size_t lanel, char *path, size_t pathl)
{
    const char *e = getenv("NEMO_CALIB_FILE");
    if (e && *e) snprintf(path, pathl, "%s", e);
    else {
#if defined(_WIN32)
        const char *la = getenv("LOCALAPPDATA");
        snprintf(path, pathl, "%s\\NemoChat\\nemo-calib.txt", la && *la ? la : ".");
#else
        const char *h = getenv("HOME");
        snprintf(path, pathl, "%s/.cache/nemochat/nemo-calib.txt", h && *h ? h : ".");
#endif
    }
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    char line[1024], dev[200] = "", why[1024] = "";
    lane[0] = 0;
    while (fgets(line, sizeof line, f)) {
        size_t L = strlen(line); while (L && (line[L - 1] == '\n' || line[L - 1] == '\r')) line[--L] = 0;
        if (!strncmp(line, "device=", 7)) snprintf(dev, sizeof dev, "%.199s", line + 7);
        else if (!strncmp(line, "lane=", 5)) snprintf(lane, lanel, "%.15s", line + 5);
        else if (!strncmp(line, "cuda_why=", 9)) snprintf(why, sizeof why, "%.1000s", line + 9);
    }
    fclose(f);
    if (strcmp(dev, c->name) != 0) return 0;
    *free_bytes = -1;
    const char *m = strstr(why, " MiB free");
    if (m) {
        const char *b = m; while (b > why && b[-1] >= '0' && b[-1] <= '9') b--;
        if (b < m) *free_bytes = atoll(b) << 20;
    }
    return 1;
}

static int xs_mem(sn_cuda *c, CUdeviceptr *p, size_t bytes) { return c->cuMemAlloc(p, bytes ? bytes : 256) == 0 ? 0 : -1; }

int sn_cuda_xs_open(sn_cuda *c, const sn_pack *p, const sn_xs_geom *g, char *why, int whylen)
{
    xs_free(c);
    why[0] = 0;
    if (!c->cuStreamCreate || !c->cuStreamDestroy || !c->cuStreamSynchronize || !c->cuStreamWaitEvent || !c->cuEventCreate || !c->cuEventRecord
        || !c->cuEventSynchronize || !c->cuEventDestroy || !c->cuMemHostAlloc || !c->cuMemFreeHost || !c->cuMemcpyHtoDAsync) {
        snprintf(why, (size_t)whylen, "the driver lacks the stream, event or pinned-memory calls"); return -1;
    }
    if (p->pq_d != 4 || p->block != 256 || p->entries != 256 || p->nsub != 64) { snprintf(why, (size_t)whylen, "the pack is not block 256 PQ d4 with 256-entry books"); return -1; }
    if (g->Lt <= 0 || g->I <= 0 || g->K <= 0 || g->nmax <= 0 || (g->Lt & 7) || (g->I & 7)) { snprintf(why, (size_t)whylen, "expert shapes the stream does not take"); return -1; }
    /* calibration first: the lane it selected, and the VRAM it measured free */
    char lane[32] = "", cpath[600] = ""; long long cal_free = -1;
    int have_cal = xs_read_calib(c, &cal_free, lane, sizeof lane, cpath, sizeof cpath);
    if (have_cal && strcmp(lane, "cuda") != 0) { snprintf(why, (size_t)whylen, "the hardware calibration (%s) selected the %s lane for this machine", cpath, lane[0] ? lane : "cpu"); return -1; }
    xs_lane *x = (xs_lane *)calloc(1, sizeof *x);
    if (!x) { snprintf(why, (size_t)whylen, "out of memory"); return -1; }
    c->xs = x;
    x->Lt = g->Lt; x->I = g->I; x->K = g->K; x->nmax = g->nmax; x->simd = g->simd; x->nkeys = g->nkeys;
    x->nb[0] = g->up_blocks; x->nb[1] = g->dn_blocks;
    size_t o = 0;
    for (int w = 0; w < 2; w++) {
        x->off[w][0] = o; o = XS_AL(o + x->nb[w] * 64);
        x->off[w][1] = o; o = XS_AL(o + x->nb[w] * 2);
        x->off[w][2] = o; o = XS_AL(o + x->nb[w] * 2);
        x->off[w][3] = o; o = XS_AL(o + p->book_floats * 4);
    }
    x->slot_bytes = o;
    sn_dtc_geom(p, g->Lt, &x->nph[0], &x->ng[0]);
    sn_dtc_geom(p, g->I, &x->nph[1], &x->ng[1]);
    size_t t0 = (size_t)x->nph[0] * x->ng[0] * 256, t1 = (size_t)x->nph[1] * x->ng[1] * 256;
    x->dT_n = t0 > t1 ? t0 : t1;
    x->dW_n = (size_t)g->I * (size_t)g->Lt;
    /* the fixed work set: activations and outputs of nmax tokens, one expanded matrix, DTC tables */
    size_t nm = (size_t)g->nmax, KK = (size_t)g->K;
    int bad = xs_mem(c, &x->dLat, nm * g->Lt * 4) || xs_mem(c, &x->dPL, nm * (g->Lt + 1) * 8) || xs_mem(c, &x->dEO, nm * KK * g->Lt * 4)
           || xs_mem(c, &x->dU, nm * g->I * 4) || xs_mem(c, &x->dPmid, ((size_t)g->I + 1) * 8) || xs_mem(c, &x->dT, x->dT_n * 4)
           || xs_mem(c, &x->dW, x->dW_n * 4) || xs_mem(c, &x->dTok, nm * KK * 4) || xs_mem(c, &x->dOidx, nm * KK * 4);
    if (bad) { snprintf(why, (size_t)whylen, "no VRAM for the expert work buffers"); xs_free(c); return -1; }
    long long work = (long long)(nm * g->Lt * 4 + nm * (g->Lt + 1) * 8 + nm * KK * g->Lt * 4 + nm * g->I * 4 + ((size_t)g->I + 1) * 8
                                 + x->dT_n * 4 + x->dW_n * 4 + 2 * nm * KK * 4);
    /* slots: as many as the calibrated VRAM allows, never more than the device reports free right now */
    size_t fr = 0, tot = 0;
    if (c->cuMemGetInfo(&fr, &tot) != 0) { snprintf(why, (size_t)whylen, "cannot read free VRAM"); xs_free(c); return -1; }
    long long avail = (long long)fr;
    if (have_cal && cal_free > 0) {
        long long cal_left = cal_free - c->used - c->cache_used - work
                           - (long long)(c->dx_n * 4 + c->dy_n * 4 + c->dT_n * 4 + c->dP_n * 8);
        if (cal_left < avail) avail = cal_left;
        snprintf(x->info.sized_by, sizeof x->info.sized_by, "hardware calibration %s (%lld MiB free when measured), device now %lld MiB free",
                 cpath, cal_free >> 20, (long long)(fr >> 20));
    } else snprintf(x->info.sized_by, sizeof x->info.sized_by, "no calibration for this GPU (%s); device reading %lld MiB free", cpath, (long long)(fr >> 20));
    avail -= 64ll << 20;                                   /* the same headroom the dense loader keeps */
    long long ns = avail > 0 ? avail / (long long)x->slot_bytes : 0;
    if (ns > g->nkeys) ns = g->nkeys;
    {   /* test knob: SN_XS_MAX_SLOTS caps the slot count so tests can force eviction and small staging halves */
        const char *cap = getenv("SN_XS_MAX_SLOTS");
        if (cap && atoll(cap) > 0 && atoll(cap) < ns) ns = atoll(cap);
    }
    if (ns < 2) { snprintf(why, (size_t)whylen, "room for %lld expert slot(s) only (%s)", ns, x->info.sized_by); xs_free(c); return -1; }
    x->nslots = (int)ns;
    while (x->nslots >= 2 && c->cuMemAlloc(&x->slots, (size_t)x->nslots * x->slot_bytes) != 0) x->nslots = x->nslots * 7 / 8;
    if (x->nslots < 2 || !x->slots) { snprintf(why, (size_t)whylen, "the expert slots would not allocate"); xs_free(c); return -1; }
    /* staging: two halves of one routed set (top-k experts) each, so the lanes fill one half while the card copies the other */
    x->chunk = g->K < x->nslots / 2 ? g->K : x->nslots / 2;
    if (x->chunk < 1) x->chunk = 1;
    x->nstage = 2 * x->chunk;
    void *hp = NULL;
    if (c->cuMemHostAlloc(&hp, (size_t)x->nstage * x->slot_bytes, 0) != 0 || !hp) { snprintf(why, (size_t)whylen, "no pinned RAM for staging"); xs_free(c); return -1; }
    x->stage = (unsigned char *)hp;
    x->slot_key = (int *)malloc((size_t)x->nslots * sizeof(int)); x->slot_tick = (uint64_t *)calloc((size_t)x->nslots, 8);
    x->slot_pin = (uint32_t *)calloc((size_t)x->nslots, 4); x->key_slot = (int *)malloc((size_t)g->nkeys * sizeof(int));
    x->ev_ready = (CUevent *)calloc((size_t)x->nslots, sizeof(CUevent)); x->ev_free = (CUevent *)calloc((size_t)x->nslots, sizeof(CUevent));
    x->ev_staged = (CUevent *)calloc((size_t)x->nstage, sizeof(CUevent));
    if (!x->slot_key || !x->slot_tick || !x->slot_pin || !x->key_slot || !x->ev_ready || !x->ev_free || !x->ev_staged) { snprintf(why, (size_t)whylen, "out of memory"); xs_free(c); return -1; }
    for (int i = 0; i < x->nslots; i++) x->slot_key[i] = -1;
    for (int i = 0; i < g->nkeys; i++) x->key_slot[i] = -1;
    for (int i = 0; i < x->nslots; i++) if (c->cuEventCreate(&x->ev_ready[i], 2) != 0 || c->cuEventCreate(&x->ev_free[i], 2) != 0) bad = 1;
    for (int i = 0; i < x->nstage; i++) if (c->cuEventCreate(&x->ev_staged[i], 2) != 0) bad = 1;
    if (bad || c->cuStreamCreate(&x->s_copy, 0) != 0 || c->cuStreamCreate(&x->s_comp, 0) != 0) { snprintf(why, (size_t)whylen, "cannot create CUDA streams or events"); xs_free(c); return -1; }
    x->pin = 1;
    {   /* slot policy: least recently used (default); SN_XS_POLICY=hot keeps the experts used most often in their slots
         * (measured on the real pack: more slot hits, no clear speed change at about 100 slots; results identical) */
        const char *pol = getenv("SN_XS_POLICY");
        x->policy = (pol && !strcmp(pol, "hot")) ? 1 : 0;
        x->key_cnt = (uint32_t *)calloc((size_t)g->nkeys, 4);
        if (!x->key_cnt) { snprintf(why, (size_t)whylen, "out of memory"); xs_free(c); return -1; }
        x->age_every = (uint64_t)x->nslots * 64u;
    }
    x->info.on = 1; x->info.slots = x->nslots; x->info.stage = x->nstage; x->info.slot_bytes = (long long)x->slot_bytes;
    x->info.vram_bytes = (long long)x->nslots * (long long)x->slot_bytes; x->info.work_bytes = work;
    x->info.pinned_bytes = (long long)x->nstage * (long long)x->slot_bytes;
    return 0;
}

int sn_cuda_xs_on(const sn_cuda *c) { return c && c->xs != NULL; }
int sn_cuda_xs_chunk(const sn_cuda *c) { return c->xs ? c->xs->chunk : 0; }
int sn_cuda_xs_nstage(const sn_cuda *c) { return c->xs ? c->xs->nstage : 0; }
void sn_cuda_xs_info(const sn_cuda *c, sn_xs_info *out)
{
    if (c && c->xs) *out = c->xs->info; else memset(out, 0, sizeof *out);
}

void sn_cuda_xs_stage_ptrs(sn_cuda *c, int j, int which, uint8_t **idx, uint16_t **mn, uint16_t **sc, float **book)
{
    xs_lane *x = c->xs;
    unsigned char *b = x->stage + (size_t)j * x->slot_bytes;
    *idx = (uint8_t *)(b + x->off[which][0]); *mn = (uint16_t *)(b + x->off[which][1]);
    *sc = (uint16_t *)(b + x->off[which][2]); *book = (float *)(b + x->off[which][3]);
}

int sn_cuda_xs_stage_wait(sn_cuda *c, int j0, int n)
{
    xs_lane *x = c->xs;
    for (int j = j0; j < j0 + n && j < x->nstage; j++) if (c->cuEventSynchronize(x->ev_staged[j]) != 0) return -1;
    return 0;
}

void sn_cuda_xs_chunk_begin(sn_cuda *c) { c->xs->pin++; }

/* use counts: one per lookup; every age_every lookups all counts halve, so experts that were hot long ago can leave */
static void xs_count(xs_lane *x, int key)
{
    if (x->key_cnt[key] < 0xffffffffu) x->key_cnt[key]++;
    if (++x->uses % x->age_every == 0) for (int k = 0; k < x->nkeys; k++) x->key_cnt[k] >>= 1;
}

int sn_cuda_xs_find(sn_cuda *c, int key)
{
    xs_lane *x = c->xs;
    if (key < 0 || key >= x->nkeys) return -1;
    int s = x->key_slot[key];
    xs_count(x, key);
    if (s < 0) return -1;
    x->slot_tick[s] = ++x->tick; x->slot_pin[s] = x->pin;
    x->info.hits++;
    return s;
}

int sn_cuda_xs_claim(sn_cuda *c, int key)
{
    xs_lane *x = c->xs;
    if (key < 0 || key >= x->nkeys) return -1;
    int v = -1;
    for (int s = 0; s < x->nslots; s++) {
        if (x->slot_pin[s] == x->pin) continue;          /* staged in this chunk: its kernels are not queued yet */
        if (x->slot_key[s] < 0) { v = s; break; }
        if (v < 0) { v = s; continue; }
        if (x->policy) {          /* fewest uses goes first; among equals the least recently used */
            uint32_t cs = x->key_cnt[x->slot_key[s]], cv = x->key_cnt[x->slot_key[v]];
            if (cs < cv || (cs == cv && x->slot_tick[s] < x->slot_tick[v])) v = s;
        } else if (x->slot_tick[s] < x->slot_tick[v]) v = s;
    }
    if (v < 0) return -1;
    if (x->slot_key[v] >= 0) x->key_slot[x->slot_key[v]] = -1;
    x->slot_key[v] = -1;                                 /* filled in by a successful upload */
    x->slot_tick[v] = ++x->tick; x->slot_pin[v] = x->pin;
    (void)key;
    return v;
}

int sn_cuda_xs_upload(sn_cuda *c, int j, int slot, int key)
{
    xs_lane *x = c->xs;
    CUdeviceptr dst = x->slots + (CUdeviceptr)slot * x->slot_bytes;
    if (c->cuStreamWaitEvent(x->s_copy, x->ev_free[slot], 0) != 0) return -1;
    if (c->cuMemcpyHtoDAsync(dst, x->stage + (size_t)j * x->slot_bytes, x->slot_bytes, x->s_copy) != 0) return -1;
    if (c->cuEventRecord(x->ev_ready[slot], x->s_copy) != 0 || c->cuEventRecord(x->ev_staged[j], x->s_copy) != 0) return -1;
    x->slot_key[slot] = key; x->key_slot[key] = slot;
    x->info.loads++; x->info.bytes_up += x->slot_bytes;
    return 0;
}

int sn_cuda_xs_layer_begin(sn_cuda *c, const float *lat, const double *plat, int n, const int *tok, const int *oidx, int nidx)
{
    xs_lane *x = c->xs;
    if (n > x->nmax || nidx > x->nmax * x->K) return -1;
    if (c->cuMemcpyHtoD(x->dLat, lat, (size_t)n * x->Lt * 4) != 0) return -1;
    if (plat && c->cuMemcpyHtoD(x->dPL, plat, (size_t)n * (x->Lt + 1) * 8) != 0) return -1;
    if (tok && nidx && c->cuMemcpyHtoD(x->dTok, tok, (size_t)nidx * 4) != 0) return -1;
    if (oidx && nidx && c->cuMemcpyHtoD(x->dOidx, oidx, (size_t)nidx * 4) != 0) return -1;
    return 0;
}

static CUdeviceptr xs_part(const xs_lane *x, int slot, int which, int part)
{
    return x->slots + (CUdeviceptr)slot * x->slot_bytes + (CUdeviceptr)x->off[which][part];
}

/* table build + lookups on one packed matrix of a slot (the decode kernels of sn_cuda_dtc_rows, on the comp stream) */
static int xs_dtc_mat(sn_cuda *c, int slot, int which, CUdeviceptr xin, CUdeviceptr P, CUdeviceptr y, int cols, int rows)
{
    xs_lane *x = c->xs;
    int nph = x->nph[which], ng = x->ng[which], gph = 256 / nph, c_ = cols, r_ = rows;
    CUdeviceptr book = xs_part(x, slot, which, 3), idx = xs_part(x, slot, which, 0), mn = xs_part(x, slot, which, 1), sc = xs_part(x, slot, which, 2);
    void *a1[] = { &book, &xin, &x->dT, &ng, &gph };
    if (c->cuLaunchKernel(c->f_dtcb, (unsigned)(nph * ng), 1, 1, 256, 1, 1, 0, x->s_comp, a1, NULL) != 0) return -1;
    void *a2[] = { &idx, &mn, &sc, &x->dT, &P, &y, &c_, &r_, &ng, &nph, &gph };
    return c->cuLaunchKernel(c->f_dtcr, (unsigned)((rows + 127) / 128), 1, 1, 128, 1, 1, 0, x->s_comp, a2, NULL) == 0 ? 0 : -1;
}

int sn_cuda_xs_dtc(sn_cuda *c, int slot, int tok, int orow)
{
    xs_lane *x = c->xs;
    if (c->cuStreamWaitEvent(x->s_comp, x->ev_ready[slot], 0) != 0) return -1;
    CUdeviceptr xin = x->dLat + (CUdeviceptr)tok * x->Lt * 4, P = x->dPL + (CUdeviceptr)tok * (x->Lt + 1) * 8;
    CUdeviceptr y = x->dEO + (CUdeviceptr)orow * x->Lt * 4;
    int I = x->I;
    if (xs_dtc_mat(c, slot, 0, xin, P, x->dU, x->Lt, I) != 0) return -1;
    void *ar[] = { &x->dU, &x->dPmid, &I };
    if (c->cuLaunchKernel(c->f_xrelu, 1, 1, 1, 256, 1, 1, (unsigned)I * 4, x->s_comp, ar, NULL) != 0) return -1;
    if (xs_dtc_mat(c, slot, 1, x->dU, x->dPmid, y, I, x->Lt) != 0) return -1;
    if (c->cuEventRecord(x->ev_free[slot], x->s_comp) != 0) return -1;
    x->info.experts_dtc++;
    return 0;
}

static int xs_expand_launch(sn_cuda *c, CUstream st, CUdeviceptr idx, CUdeviceptr mn, CUdeviceptr sc, CUdeviceptr book, uint64_t g0, uint64_t ng, CUdeviceptr out)
{
    if (ng == 0) return 0;
    void *a[] = { &idx, &mn, &sc, &book, &g0, &ng, &out };
    return c->cuLaunchKernel(c->f_xexp, (unsigned)((ng + 255) / 256), 1, 1, 256, 1, 1, 0, st, a, NULL) == 0 ? 0 : -1;
}

static int xs_gemm_launch(sn_cuda *c, CUstream st, CUdeviceptr W, int rows, int K, CUdeviceptr X, CUdeviceptr xi, int N,
                          CUdeviceptr Y, CUdeviceptr yi, int ystride, int r0, int act, int simd)
{
    int bx = N >= 32 ? 32 : N, by = 256 / bx;
    if (by > 64) by = 64;
    void *a[] = { &W, &rows, &K, &X, &xi, &N, &Y, &yi, &ystride, &r0, &act, &simd };
    return c->cuLaunchKernel(c->f_xgemm, (unsigned)((N + bx - 1) / bx), (unsigned)((rows + by - 1) / by), 1, (unsigned)bx, (unsigned)by, 1, 0, st, a, NULL) == 0 ? 0 : -1;
}

int sn_cuda_xs_gemm(sn_cuda *c, int slot, int start, int cnt)
{
    xs_lane *x = c->xs;
    if (cnt <= 0) return 0;
    if (c->cuStreamWaitEvent(x->s_comp, x->ev_ready[slot], 0) != 0) return -1;
    CUdeviceptr none = 0, tok = x->dTok + (CUdeviceptr)start * 4, oi = x->dOidx + (CUdeviceptr)start * 4;
    /* up: expand on the card, G = gathered latents, U = relu2(W G) */
    if (xs_expand_launch(c, x->s_comp, xs_part(x, slot, 0, 0), xs_part(x, slot, 0, 1), xs_part(x, slot, 0, 2), xs_part(x, slot, 0, 3), 0, (uint64_t)x->I * x->Lt / 4, x->dW) != 0) return -1;
    if (xs_gemm_launch(c, x->s_comp, x->dW, x->I, x->Lt, x->dLat, tok, cnt, x->dU, none, x->I, 0, 2, x->simd) != 0) return -1;
    /* down: expand over the same scratch (stream order), outputs scattered to their (token, slot) rows */
    if (xs_expand_launch(c, x->s_comp, xs_part(x, slot, 1, 0), xs_part(x, slot, 1, 1), xs_part(x, slot, 1, 2), xs_part(x, slot, 1, 3), 0, (uint64_t)x->I * x->Lt / 4, x->dW) != 0) return -1;
    if (xs_gemm_launch(c, x->s_comp, x->dW, x->Lt, x->I, x->dU, none, cnt, x->dEO, oi, x->Lt, 0, 0, x->simd) != 0) return -1;
    if (c->cuEventRecord(x->ev_free[slot], x->s_comp) != 0) return -1;
    x->info.experts_gemm++;
    return 0;
}

int sn_cuda_xs_layer_end(sn_cuda *c, float *eo, int nrows)
{
    xs_lane *x = c->xs;
    if (c->cuStreamSynchronize(x->s_comp) != 0) return -1;
    return c->cuMemcpyDtoH(eo, x->dEO, (size_t)nrows * x->Lt * 4) == 0 ? 0 : -1;
}

void sn_cuda_xs_abort(sn_cuda *c)
{
    xs_lane *x = c->xs;
    if (!x) return;
    c->cuStreamSynchronize(x->s_copy); c->cuStreamSynchronize(x->s_comp);
    for (int s = 0; s < x->nslots; s++) { if (x->slot_key[s] >= 0) x->key_slot[x->slot_key[s]] = -1; x->slot_key[s] = -1; }
}

/* Prefill product on a dense tensor already resident (packed or raw) on the card: expand row slabs into the scratch, then
 * the same GEMM kernel; Y[n * R + r] like sn_gemm_linear. */
int sn_cuda_xs_dense_gemm(sn_cuda *c, const sn_pack *p, const sn_tensor *t, const float *X, int N, int K, int R, float *Y, int act)
{
    xs_lane *x = c->xs;
    long ti = (long)(t - p->t);
    if (!x || ti < 0 || ti >= c->nres || !c->res[ti].kind || (K & 7) || N < 1 || (act != 0 && act != 2)) return -1;
    if ((uint64_t)K * (uint64_t)R != t->count) return -1;
    const dres *r = &c->res[ti];
    size_t nx = (size_t)N * K, ny = (size_t)N * R;
    if (nx > x->dX_n) { if (x->dX) c->cuMemFree(x->dX); x->dX = 0; x->dX_n = 0; if (xs_mem(c, &x->dX, nx * 4) != 0) return -1; x->dX_n = nx; }
    if (ny > x->dY_n) { if (x->dY) c->cuMemFree(x->dY); x->dY = 0; x->dY_n = 0; if (xs_mem(c, &x->dY, ny * 4) != 0) return -1; x->dY_n = ny; }
    if (c->cuStreamSynchronize(x->s_comp) != 0 || c->cuMemcpyHtoD(x->dX, X, nx * 4) != 0) return -1;
    int slab = (int)(x->dW_n / (size_t)K);
    if (slab < 1) return -1;
    CUdeviceptr none = 0;
    for (int r0 = 0; r0 < R; r0 += slab) {
        int rows = R - r0 < slab ? R - r0 : slab;
        uint64_t w0 = (uint64_t)r0 * (uint64_t)K, nw = (uint64_t)rows * (uint64_t)K;
        if (r->kind == 1) {
            if (xs_expand_launch(c, x->s_comp, r->idx, r->mn, r->sc, r->book, w0 / 4, nw / 4, x->dW) != 0) return -1;
        } else {
            CUdeviceptr src = r->idx + (CUdeviceptr)w0 * 2;
            void *a[] = { &src, &nw, &x->dW };
            if (c->cuLaunchKernel(c->f_xraw, (unsigned)((nw + 255) / 256), 1, 1, 256, 1, 1, 0, x->s_comp, a, NULL) != 0) return -1;
        }
        if (xs_gemm_launch(c, x->s_comp, x->dW, rows, K, x->dX, none, N, x->dY, none, R, r0, act, x->simd) != 0) return -1;
    }
    if (c->cuStreamSynchronize(x->s_comp) != 0 || c->cuMemcpyDtoH(Y, x->dY, ny * 4) != 0) return -1;
    x->info.dense_gemm++;
    return 0;
}

/* Test hook: expand weights [first, first + n) of a resident tensor (or of slot `slot`'s up/down matrix when t is NULL) on the
 * card and copy them back. first and n must be multiples of 4 for packed tensors. */
int sn_cuda_expand(sn_cuda *c, const sn_pack *p, const sn_tensor *t, int slot, int which, uint64_t first, uint64_t n, float *out)
{
    xs_lane *x = c->xs;
    if (!x || n == 0) return -1;
    CUdeviceptr dst = 0;
    if (xs_mem(c, &dst, (size_t)n * 4) != 0) return -1;
    int rc = -1;
    if (!t) {
        if (slot >= 0 && slot < x->nslots && !(first & 3) && !(n & 3) && first + n <= x->nb[which] * 256
            && c->cuStreamWaitEvent(x->s_comp, x->ev_ready[slot], 0) == 0)
            rc = xs_expand_launch(c, x->s_comp, xs_part(x, slot, which, 0), xs_part(x, slot, which, 1), xs_part(x, slot, which, 2), xs_part(x, slot, which, 3), first / 4, n / 4, dst);
    } else {
        long ti = (long)(t - p->t);
        if (ti >= 0 && ti < c->nres && c->res[ti].kind && first + n <= t->count) {
            const dres *r = &c->res[ti];
            if (r->kind == 1 && !(first & 3) && !(n & 3)) rc = xs_expand_launch(c, x->s_comp, r->idx, r->mn, r->sc, r->book, first / 4, n / 4, dst);
            else if (r->kind == 2) {
                CUdeviceptr src = r->idx + (CUdeviceptr)first * 2;
                void *a[] = { &src, &n, &dst };
                rc = c->cuLaunchKernel(c->f_xraw, (unsigned)((n + 255) / 256), 1, 1, 256, 1, 1, 0, x->s_comp, a, NULL) == 0 ? 0 : -1;
            }
        }
    }
    if (rc == 0) rc = c->cuStreamSynchronize(x->s_comp) == 0 && c->cuMemcpyDtoH(out, dst, (size_t)n * 4) == 0 ? 0 : -1;
    c->cuMemFree(dst);
    return rc;
}
