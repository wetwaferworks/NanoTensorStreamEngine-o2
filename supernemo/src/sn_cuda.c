/* sn_cuda.c - see sn_cuda.h. Pure C99; the device code is a string compiled by NVRTC at run time. */
#include "sn_cuda.h"
#include "sn_pqk.h"

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
typedef void *CUcontext, *CUmodule, *CUfunction, *CUstream;
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
    CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction f_pq, f_raw, f_dtcb, f_dtcr;
    char name[128];
    long long budget, used; int count;
    CUdeviceptr dx, dy; size_t dx_n, dy_n;
    CUdeviceptr dT, dP; size_t dT_n, dP_n;      /* DTC table and prefix-sum scratch */
    long long cache_budget, cache_used; int cache_count;
    dres *res; int nres;
};

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
    if (nvrtcCreateProgram(&prog, KERNELS, "sn_kernels.cu", 0, NULL, NULL) != 0) { snprintf(why, (size_t)whylen, "NVRTC cannot create the program"); goto fail; }
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
        || c->cuModuleGetFunction(&c->f_dtcb, c->mod, "dtc_build") != 0 || c->cuModuleGetFunction(&c->f_dtcr, c->mod, "dtc_rows") != 0) {
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
