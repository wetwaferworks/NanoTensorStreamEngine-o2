/* nemo_gpu.c - Little Nemo end-to-end GPU runtime, CUDA lane (accel/).
 *
 * Host side is plain C99: the CUDA driver API (nvcuda.dll) and NVRTC
 * (nvrtc64_*.dll) are loaded at run time, the kernels in
 * nemo_gpu_kernels.h are compiled for the device's own SM at open.
 * The loader/NVRTC pattern follows the accel/ CUDA lane of the
 * 2026-11-26 accel worktree (read, not modified). One thread, no
 * mutexes, every driver call checked.
 *
 * Residency: PQ index bytes + fp16 frames + fp32 books for all 2-D
 * layer projections, lm_head in bf16, all small parameters in fp32,
 * Mamba states, conv windows and the bf16 KV cache. Nothing streams
 * per token except the 4480-float embedding row up and the logits down.
 */
#define _POSIX_C_SOURCE 200809L
#include "nemo_gpu.h"
#include "pool.h"

#include <math.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif
#include <time.h>

#ifdef _WIN32
#include <windows.h>
static void *dl_open(const char *n) {
    /* Full paths load with their own directory on the search path
     * (NVRTC needs its builtins DLL beside it). Match ntse_dl_open(). */
    if (strchr(n, '\\') || strchr(n, '/'))
        return (void *)LoadLibraryExA(n, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    return (void *)LoadLibraryA(n);
}
static void *dl_sym(void *l, const char *s) { return l ? (void *)GetProcAddress((HMODULE)l, s) : NULL; }
static void dl_close(void *l) { if (l) FreeLibrary((HMODULE)l); }
#else
#include <dlfcn.h>
static void *dl_open(const char *n) { return dlopen(n, RTLD_NOW | RTLD_LOCAL); }
static void *dl_sym(void *l, const char *s) { return l ? dlsym(l, s) : NULL; }
static void dl_close(void *l) { if (l) dlclose(l); }
#endif

#include "loader.h"
#include "nemo_gpu_kernels.h"

/* ---- driver API / NVRTC surface (types only, no CUDA headers) ---- */
typedef int CUresult;
typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;
typedef int nvrtcResult;
typedef struct _nvrtcProgram *nvrtcProgram;

typedef struct {
    CUresult (*Init)(unsigned);
    CUresult (*DeviceGet)(CUdevice *, int);
    CUresult (*DeviceGetCount)(int *);
    CUresult (*DeviceGetName)(char *, int, CUdevice);
    CUresult (*DeviceGetAttribute)(int *, int, CUdevice);
    CUresult (*DeviceTotalMem)(size_t *, CUdevice);
    CUresult (*PrimaryCtxRetain)(CUcontext *, CUdevice);
    CUresult (*PrimaryCtxRelease)(CUdevice);
    CUresult (*CtxSetCurrent)(CUcontext);
    CUresult (*CtxSynchronize)(void);
    CUresult (*MemGetInfo)(size_t *, size_t *);
    CUresult (*MemAlloc)(CUdeviceptr *, size_t);
    CUresult (*MemFree)(CUdeviceptr);
    CUresult (*MemcpyHtoD)(CUdeviceptr, const void *, size_t);
    CUresult (*MemcpyDtoH)(void *, CUdeviceptr, size_t);
    CUresult (*MemcpyDtoD)(CUdeviceptr, CUdeviceptr, size_t);
    CUresult (*DriverGetVersion)(int *);
    CUresult (*MemsetD8)(CUdeviceptr, unsigned char, size_t);
    CUresult (*ModuleLoadData)(CUmodule *, const void *);
    CUresult (*ModuleUnload)(CUmodule);
    CUresult (*ModuleGetFunction)(CUfunction *, CUmodule, const char *);
    CUresult (*FuncSetAttribute)(CUfunction, int, int);
    CUresult (*LaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned,
                             unsigned, unsigned, unsigned, CUstream, void **, void **);
    CUresult (*GetErrorString)(CUresult, const char **);
} cu_api;

typedef struct {
    nvrtcResult (*Create)(nvrtcProgram *, const char *, const char *, int,
                          const char *const *, const char *const *);
    nvrtcResult (*Compile)(nvrtcProgram, int, const char *const *);
    nvrtcResult (*LogSize)(nvrtcProgram, size_t *);
    nvrtcResult (*Log)(nvrtcProgram, char *);
    nvrtcResult (*CubinSize)(nvrtcProgram, size_t *);
    nvrtcResult (*Cubin)(nvrtcProgram, char *);
    nvrtcResult (*PtxSize)(nvrtcProgram, size_t *);
    nvrtcResult (*Ptx)(nvrtcProgram, char *);
    nvrtcResult (*Destroy)(nvrtcProgram *);
    nvrtcResult (*Version)(int *, int *);
} rtc_api;

enum { K_PQ, K_BF16, K_RMS, K_ADD, K_RELU2, K_CONV, K_SSM, K_GNORM, K_KVPUSH,
       K_ASCORE, K_AOUT, K_I8, K_XSUM, K_PQF, K_PQH, K_PQS, K_PQU, K_PQV, K_PQW, K_PQT, K_BW,
       K_GT8, K_GT16, K_GR8T8, K_GR8T4, K_GR2T32, K_GT8K, K_PQT1K, K_PQX4, K_GX8, K_GX16, K_RMSB, K_CONVB, K_SSMB, K_GNORMB, K_KVPB, K_PQFX, K_GF16, K_N };
static const char *const kname[K_N] = {
    "k_pq_gemv", "k_bf16_gemv", "k_rmsnorm", "k_add", "k_relu2", "k_conv",
    "k_ssm", "k_gated_norm", "k_kv_push", "k_attn_scores", "k_attn_out", "k_i8_gemv", "k_xsum16", "k_pq_gemv_fast", "k_pq_gemv_h", "k_pq_gemv_s", "k_pq_gemv_u", "k_pq_gemv_v", "k_pq_gemv_w", "k_pq_gemv_t", "k_bw",
    "k_pq_gemm_t8", "k_pq_gemm_t16", "k_pq_gemm_r8t8", "k_pq_gemm_r8t4", "k_pq_gemm_r2t32", "k_pq_gemm_t8k", "k_pq_gemv_t1k", "k_pq_gemv_x4", "k_pq_gemm_x8", "k_pq_gemm_x16", "k_rmsnorm_b", "k_conv_b", "k_ssm_b", "k_gated_norm_b", "k_kv_push_b", "k_pq_gemv_f", "k_pq_gemm_f16" };

#define H 4480
#define VOCAB 131072
#define NLAYERS 56
#define INNER 10240
#define CONV 12288
#define PROJ 22656
#define NHEAD 128
#define HDIM 80
#define NSTATE 128
#define NGROUP 8
#define MLP 15680
#define NQ 40
#define NKV 8
#define AD 128
#define EPSF 1e-5f

static const char LAYER_PATTERN[NLAYERS + 1] =
    "M-M-M-MM-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM-M-M-M-M-M-";

/* mc: the block frame as one 32-bit word {mn fp16 | sc fp16 << 16}, so a
 * block costs one 4-byte load instead of two 2-byte loads (the t/t1k/GEMM
 * and FULL kernels); same bytes as the separate arrays, which are kept
 * only for the legacy kernels. */
typedef struct { CUdeviceptr idx, mn, sc, mc, book, bookh; int rows, cols, pidx; } gpq;

typedef struct {
    char type;
    CUdeviceptr norm_w;
    gpq p0, p1, p2, p3;                     /* M: in,out  -: up,down  *: q,k,v,o */
    gpq r0, r1, r2, r3;                     /* expanded pack: refinement of p0..p3 (rows 0 = none) */
    CUdeviceptr conv_w, conv_b, a_log, d, dt_bias, gnorm_w;
    CUdeviceptr ssm, win;                   /* M state */
    CUdeviceptr kc, vc;                     /* * cache */
    CUdeviceptr ssm_s, win_s;               /* M: snapshot for a step redo */
} glayer;

struct nemo_gpu {
    void *lcu, *lrtc;
    cu_api cu;
    rtc_api rtc;
    CUdevice dev;
    CUcontext ctx;
    int ctx_ok;
    CUmodule mod;
    CUfunction fn[K_N];
    nemo_gpu_info info;
    int layer_limit, max_pos, len, valid;
    glayer L[NLAYERS];
    CUdeviceptr norm_f, lm_head, head_scale;
    int head_i8;
    CUdeviceptr x, h, proj, v, y, inner, mid, q, k, vv, logits;
    CUdeviceptr scores, xs;
    /* batched prefill workspace: [chunk][width] per buffer (0 = per-token prefill) */
    int chunk, ptb, gemm_k, gemm_r, gemv_k, gemv_x;  /* tuned kernel choices (NEMO_GPU_GEMM / NEMO_GPU_GEMV) */
    CUdeviceptr bx, bh, bproj, bv, by, binner, bmid, bq, bk, bvv;
    float *emb_host;
    int exact, half_book;
    int full_ok, snap_ok, snap_len;          /* EXPAND resources (fp32 books, state snapshot) */
    char full_why[160];
    FILE *emb_f;
    uint64_t emb_off;
    uint64_t used;
    void (*hook)(void *, int, const float *);
    void *hook_user;
    CUdeviceptr rt, brt;                     /* refinement GEMV / GEMM outputs (expanded packs only) */
    int nref;                                /* refinement tensors resident */
    uint64_t ref_bytes;
    int head_host;                           /* 1: int8 lm_head and its GEMV live on the host */
    int8_t *head_q;                          /* VOCAB * H, host */
    float *head_sc, *h_host;
    nemo_pool head_pool;
    int head_pool_ok;
    char err[256];
};

/* ------------------------------------------------------------------ */

static int sym(void *lib, const char *n, void *slot) {
    void *p = dl_sym(lib, n);
    if (!p) return -1;
    memcpy(slot, &p, sizeof p);
    return 0;
}

static int load_cuda(nemo_gpu *g, char *why, size_t wl) {
#ifdef _WIN32
    g->lcu = dl_open("nvcuda.dll");
#else
    g->lcu = dl_open("libcuda.so.1");
#endif
    if (!g->lcu) { snprintf(why, wl, "CUDA driver library not found"); return -1; }
    struct { const char *n; void *s; } t[] = {
        {"cuInit", &g->cu.Init}, {"cuDeviceGet", &g->cu.DeviceGet},
        {"cuDeviceGetCount", &g->cu.DeviceGetCount},
        {"cuDeviceGetName", &g->cu.DeviceGetName},
        {"cuDeviceGetAttribute", &g->cu.DeviceGetAttribute},
        {"cuDeviceTotalMem_v2", &g->cu.DeviceTotalMem},
        {"cuDevicePrimaryCtxRetain", &g->cu.PrimaryCtxRetain},
        {"cuDevicePrimaryCtxRelease_v2", &g->cu.PrimaryCtxRelease},
        {"cuCtxSetCurrent", &g->cu.CtxSetCurrent},
        {"cuCtxSynchronize", &g->cu.CtxSynchronize},
        {"cuMemGetInfo_v2", &g->cu.MemGetInfo}, {"cuMemAlloc_v2", &g->cu.MemAlloc},
        {"cuMemFree_v2", &g->cu.MemFree}, {"cuMemcpyHtoD_v2", &g->cu.MemcpyHtoD},
        {"cuMemcpyDtoH_v2", &g->cu.MemcpyDtoH}, {"cuMemsetD8_v2", &g->cu.MemsetD8},
        {"cuMemcpyDtoD_v2", &g->cu.MemcpyDtoD}, {"cuDriverGetVersion", &g->cu.DriverGetVersion},
        {"cuModuleLoadData", &g->cu.ModuleLoadData},
        {"cuModuleUnload", &g->cu.ModuleUnload},
        {"cuModuleGetFunction", &g->cu.ModuleGetFunction},
        {"cuLaunchKernel", &g->cu.LaunchKernel},
        {"cuFuncSetAttribute", &g->cu.FuncSetAttribute},
        {"cuGetErrorString", &g->cu.GetErrorString},
    };
    for (size_t i = 0; i < sizeof t / sizeof t[0]; i++)
        if (sym(g->lcu, t[i].n, t[i].s) != 0) { snprintf(why, wl, "CUDA driver lacks %s", t[i].n); return -1; }
    const char *env = getenv("NTSE_NVRTC_DLL");
    if (env && *env) g->lrtc = dl_open(env);
#ifdef _WIN32
    static const char *names[] = {"nvrtc64_130_0.dll", "nvrtc64_120_0.dll", "nvrtc64_112_0.dll", "nvrtc64_111_0.dll"};
    const char *cp = getenv("CUDA_PATH");
    for (size_t i = 0; !g->lrtc && i < sizeof names / sizeof names[0]; i++) {
        char path[1024];
        if (cp && *cp) {
            snprintf(path, sizeof path, "%s\\bin\\%s", cp, names[i]);
            g->lrtc = dl_open(path);
            if (!g->lrtc) { snprintf(path, sizeof path, "%s\\bin\\x64\\%s", cp, names[i]); g->lrtc = dl_open(path); }
        }
        if (!g->lrtc) g->lrtc = dl_open(names[i]);
    }
#else
    if (!g->lrtc) g->lrtc = dl_open("libnvrtc.so");
#endif
    if (!g->lrtc) { snprintf(why, wl, "NVRTC library not found (set NTSE_NVRTC_DLL or CUDA_PATH)"); return -1; }
    struct { const char *n; void *s; int need; } r[] = {
        {"nvrtcCreateProgram", &g->rtc.Create, 1}, {"nvrtcCompileProgram", &g->rtc.Compile, 1},
        {"nvrtcGetProgramLogSize", &g->rtc.LogSize, 1}, {"nvrtcGetProgramLog", &g->rtc.Log, 1},
        {"nvrtcGetCUBINSize", &g->rtc.CubinSize, 0}, {"nvrtcGetCUBIN", &g->rtc.Cubin, 0},
        {"nvrtcGetPTXSize", &g->rtc.PtxSize, 1}, {"nvrtcGetPTX", &g->rtc.Ptx, 1},
        {"nvrtcDestroyProgram", &g->rtc.Destroy, 1}, {"nvrtcVersion", &g->rtc.Version, 0},
    };
    for (size_t i = 0; i < sizeof r / sizeof r[0]; i++)
        if (sym(g->lrtc, r[i].n, r[i].s) != 0 && r[i].need) { snprintf(why, wl, "NVRTC lacks %s", r[i].n); return -1; }
    return 0;
}

static int ck(nemo_gpu *g, CUresult r, const char *what) {
    if (r == 0) return 0;
    const char *s = NULL;
    if (g->cu.GetErrorString) g->cu.GetErrorString(r, &s);
    snprintf(g->err, sizeof g->err, "%s: %s (%d)", what, s ? s : "?", r);
    return -1;
}

/* NVRTC options every compile uses, after the architecture flag. They
 * live at file scope so the kernel digest below covers them too. */
static const char *const nvrtc_fixed_opts[] = {"--fmad=false", "--std=c++11", "--ftz=false", "--prec-div=true", "--prec-sqrt=true"};
#define NVRTC_FIXED_OPTS ((int)(sizeof nvrtc_fixed_opts / sizeof nvrtc_fixed_opts[0]))

static uint64_t fnv1a64(uint64_t h, const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 0x100000001b3ull; }
    return h;
}
/* Digest of what the GPU will actually run: the kernel source text and
 * the fixed NVRTC options. The calibration cache keys on it, so editing
 * a kernel invalidates a cached pass even if nothing else changed. */
uint64_t nemo_gpu_kernel_digest(void) {
    uint64_t h = fnv1a64(0xcbf29ce484222325ull, nemo_gpu_cuda_src, sizeof nemo_gpu_cuda_src - 1);
    for (int i = 0; i < NVRTC_FIXED_OPTS; i++) {
        h = fnv1a64(h, "\0", 1);
        h = fnv1a64(h, nvrtc_fixed_opts[i], strlen(nvrtc_fixed_opts[i]));
    }
    return h;
}
/* When this translation unit was compiled. Any rebuild of the lane
 * changes it, which covers host-side changes to the self-test that the
 * kernel digest cannot see. */
const char *nemo_gpu_build_id(void) { return __DATE__ " " __TIME__; }

static char *compile(nemo_gpu *g, int maj, int min, int real, char *why, size_t wl) {
    nvrtcProgram p = NULL;
    if (g->rtc.Create(&p, nemo_gpu_cuda_src, "nemo_gpu.cu", 0, NULL, NULL) != 0) {
        snprintf(why, wl, "nvrtcCreateProgram failed"); return NULL;
    }
    char arch[64];
    snprintf(arch, sizeof arch, "--gpu-architecture=%s_%d%d", real ? "sm" : "compute", maj, min);
    const char *opts[1 + NVRTC_FIXED_OPTS];
    opts[0] = arch;
    for (int i = 0; i < NVRTC_FIXED_OPTS; i++) opts[1 + i] = nvrtc_fixed_opts[i];
    if (g->rtc.Compile(p, 1 + NVRTC_FIXED_OPTS, opts) != 0) {
        size_t ls = 0; char log[400] = "";
        g->rtc.LogSize(p, &ls);
        if (ls > 1 && ls < (1u << 20)) { char *f = (char *)malloc(ls); if (f && g->rtc.Log(p, f) == 0) snprintf(log, sizeof log, "%s", f); free(f); }
        snprintf(why, wl, "NVRTC compile failed: %.300s", log);
        g->rtc.Destroy(&p);
        return NULL;
    }
    size_t sz = 0; char *img = NULL; int rc;
    if (real) {
        if (!g->rtc.CubinSize || !g->rtc.Cubin) { g->rtc.Destroy(&p); return NULL; }
        rc = g->rtc.CubinSize(p, &sz);
        if (rc == 0 && sz && (img = (char *)malloc(sz))) rc = g->rtc.Cubin(p, img);
    } else {
        rc = g->rtc.PtxSize(p, &sz);
        if (rc == 0 && sz && (img = (char *)malloc(sz))) rc = g->rtc.Ptx(p, img);
    }
    g->rtc.Destroy(&p);
    if (rc != 0 || !img) { free(img); snprintf(why, wl, "NVRTC output retrieval failed"); return NULL; }
    { const char *dp = getenv("NEMO_GPU_DUMP"); /* developer aid: write the compiled image */
      if (dp && *dp) { FILE *f = fopen(dp, "wb"); if (f) { fwrite(img, 1, sz, f); fclose(f); } } }
    return img;
}

static int dalloc(nemo_gpu *g, CUdeviceptr *p, size_t n) {
    if (ck(g, g->cu.MemAlloc(p, n ? n : 4), "cuMemAlloc")) return -1;
    g->used += n;
    return 0;
}

static int up(nemo_gpu *g, CUdeviceptr *p, const void *src, size_t n) {
    if (dalloc(g, p, n)) return -1;
    return ck(g, g->cu.MemcpyHtoD(*p, src, n), "cuMemcpyHtoD");
}

/* fp32 parameter from the checkpoint, uploaded; count checked. */
static int up_param(nemo_gpu *g, const nemo_checkpoint *cp, const char *name,
                    uint64_t want, CUdeviceptr *p) {
    const nemo_tensor_info *t = nemo_find(cp, name);
    if (!t || t->count != want) { snprintf(g->err, sizeof g->err, "tensor %s missing or misshaped", name); return -1; }
    float *f = nemo_load_f32(cp, t);
    if (!f) { snprintf(g->err, sizeof g->err, "cannot load %s", name); return -1; }
    int rc = up(g, p, f, want * sizeof(float));
    free(f);
    return rc;
}

/* ---- the .nq2 v3 PQ pack (stream + book + manifest) ---- */
typedef struct {
    FILE *s;
    uint64_t header, nblocks;
    uint32_t pq_d, idxw, entries, block;
    int nt;
    char (*name)[256];
    uint64_t *b0, *nb, *book_off, *seg_digest, *book_digest;
    float *book;
    uint64_t book_floats;
} pack;

static int rd32(FILE *f, uint32_t *v) { uint8_t b[4]; if (fread(b, 1, 4, f) != 4) return -1; *v = b[0] | (b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24); return 0; }
static int rd64(FILE *f, uint64_t *v) { uint32_t lo, hi; if (rd32(f, &lo) || rd32(f, &hi)) return -1; *v = lo | ((uint64_t)hi << 32); return 0; }
static int seek64(FILE *f, uint64_t off) {
#ifdef _WIN32
    return _fseeki64(f, (long long)off, SEEK_SET);
#else
    return fseeko(f, (off_t)off, SEEK_SET);
#endif
}
static uint64_t size64(FILE *f) {
#ifdef _WIN32
    if (_fseeki64(f, 0, SEEK_END)) return 0;
    return (uint64_t)_ftelli64(f);
#else
    if (fseeko(f, 0, SEEK_END)) return 0;
    return (uint64_t)ftello(f);
#endif
}

#define PACK_FNV_SEED 14695981039346656037ull
static uint64_t pack_fnv(const void *data, size_t len, uint64_t h) {
    const uint8_t *p = (const uint8_t *)data;
    for (size_t i = 0; i < len; i++) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ull;
    }
    return h;
}
static uint64_t pack_fnv_stream(FILE *f, uint64_t off, uint64_t len, uint64_t h) {
    uint8_t buf[65536];
    if (seek64(f, off) != 0) return 0;
    while (len) {
        size_t want = len > sizeof(buf) ? sizeof(buf) : (size_t)len;
        if (fread(buf, 1, want, f) != want) return 0;
        h = pack_fnv(buf, want, h);
        len -= want;
    }
    return h;
}
static void trim_nl(char *s) {
    size_t n = strlen(s);
    while (n && (s[n - 1] == '\n' || s[n - 1] == '\r')) s[--n] = 0;
}

static void pack_free(pack *k) {
    if (k->s) fclose(k->s);
    free(k->name); free(k->b0); free(k->nb); free(k->book_off);
    free(k->seg_digest); free(k->book_digest); free(k->book);
    memset(k, 0, sizeof *k);
}

static int pack_open(pack *k, const char *path, char *why, size_t wl) {
    memset(k, 0, sizeof *k);
    k->s = fopen(path, "rb");
    if (!k->s) { snprintf(why, wl, "cannot open pack %s", path); return -1; }
    char mg[4]; uint32_t ver, nseg, flags; uint64_t total, gen;
    if (fread(mg, 1, 4, k->s) != 4 || memcmp(mg, "NQ2\0", 4) || rd32(k->s, &ver) || ver != 3
        || rd32(k->s, &k->block) || rd32(k->s, &k->entries) || rd64(k->s, &total)
        || rd64(k->s, &k->nblocks) || rd32(k->s, &nseg) || nseg == 0 || nseg > 4096) {
        snprintf(why, wl, "pack is not an nq2 format-v3 stream"); return -1;
    }
    uint64_t *seg = (uint64_t *)malloc(sizeof(uint64_t) * nseg);
    if (!seg) return -1;
    for (uint32_t i = 0; i < nseg; i++) if (rd64(k->s, &seg[i])) { free(seg); return -1; }
    if (rd64(k->s, &gen) || rd32(k->s, &flags)) { free(seg); return -1; }
    if (flags != 1u || rd32(k->s, &k->pq_d) || k->pq_d != 4 || k->block != 256 || k->entries != 256) {
        snprintf(why, wl, "pack layout unsupported by the GPU runtime (need PQ d=4, K=256, block 256, no raw/per-tensor-block sections; flags=%u)", flags);
        free(seg); return -1;
    }
    k->idxw = k->block / k->pq_d;
    long hp = ftell(k->s);
    k->header = (uint64_t)(hp > 0 ? hp : 0);
    if (size64(k->s) != k->header + (uint64_t)(k->idxw + 4) * k->nblocks) {
        snprintf(why, wl, "pack stream size does not match its header"); free(seg); return -1;
    }
    /* manifest: names + segments, in packing order */
    char mp[1200];
    snprintf(mp, sizeof mp, "%s", path);
    size_t L = strlen(mp);
    if (L < 5 || strcmp(mp + L - 4, ".nq2")) { free(seg); snprintf(why, wl, "pack name must end in .nq2"); return -1; }
    strcpy(mp + L - 4, ".nq2manifest");
    FILE *m = fopen(mp, "r");
    if (!m) { free(seg); snprintf(why, wl, "manifest %s missing", mp); return -1; }
    char line[1200]; unsigned long long mgen = 0; int nt = -1;
    if (!fgets(line, sizeof line, m) || strcmp(line, "NQ2MANIFEST 1\n")
        || !fgets(line, sizeof line, m) || sscanf(line, "generation %llx", &mgen) != 1 || mgen != gen
        || !fgets(line, sizeof line, m) || sscanf(line, "tensors %d", &nt) != 1 || nt != (int)nseg) {
        fclose(m); free(seg); snprintf(why, wl, "manifest does not describe this pack"); return -1;
    }
    k->nt = nt;
    k->name = malloc(sizeof(*k->name) * (size_t)nt);
    k->b0 = malloc(sizeof(uint64_t) * (size_t)nt);
    k->nb = malloc(sizeof(uint64_t) * (size_t)nt);
    k->book_off = malloc(sizeof(uint64_t) * (size_t)nt);
    k->seg_digest = malloc(sizeof(uint64_t) * (size_t)nt);
    k->book_digest = malloc(sizeof(uint64_t) * (size_t)nt);
    if (!k->name || !k->b0 || !k->nb || !k->book_off || !k->seg_digest || !k->book_digest) { fclose(m); free(seg); return -1; }
    uint64_t cur = 0;
    for (int i = 0; i < nt; i++) {
        unsigned long long b0, nb, sd, bd; unsigned blk; char kind[16];
        if (!fgets(line, sizeof line, m)
            || sscanf(line, "tensor %255s %llu %llu %u %llx %llx %15s", k->name[i], &b0, &nb, &blk, &sd, &bd, kind) != 7
            || b0 != cur || nb != seg[i] || blk != k->block || strcmp(kind, "packed") != 0) {
            fclose(m); free(seg); snprintf(why, wl, "manifest line %d disagrees with the stream", i); return -1;
        }
        k->b0[i] = b0; k->nb[i] = nb; k->book_off[i] = (uint64_t)i * k->entries * k->block;
        k->seg_digest[i] = (uint64_t)sd; k->book_digest[i] = (uint64_t)bd;
        cur += nb;
    }
    {
        int saw_end = 0;
        while (fgets(line, sizeof line, m)) {
            trim_nl(line);
            if (!line[0]) continue;
            if (strcmp(line, "end") == 0) { saw_end = 1; break; }
        }
        if (!saw_end) {
            fclose(m); free(seg);
            snprintf(why, wl, "manifest is missing its trailing end line");
            return -1;
        }
    }
    fclose(m);
    free(seg);
    if (cur != k->nblocks) { snprintf(why, wl, "pack block walk mismatch"); return -1; }
    /* book */
    char bp[1200];
    snprintf(bp, sizeof bp, "%s", path);
    strcpy(bp + L - 4, ".nq2book");
    FILE *b = fopen(bp, "rb");
    if (!b) { snprintf(why, wl, "book %s missing", bp); return -1; }
    uint32_t bver, bblk, bent, bbooks, bpq; uint64_t bgen;
    if (fread(mg, 1, 4, b) != 4 || memcmp(mg, "NQ2B", 4) || rd32(b, &bver) || bver != 3 || rd32(b, &bblk)
        || rd32(b, &bent) || rd32(b, &bbooks) || rd64(b, &bgen) || rd32(b, &bpq)
        || bblk != k->block || bent != k->entries || bgen != gen || bpq != k->pq_d || bbooks != (uint32_t)nt) {
        fclose(b); snprintf(why, wl, "book does not pair with the stream (generation/geometry)"); return -1;
    }
    k->book_floats = (uint64_t)nt * k->entries * k->block;
    k->book = (float *)malloc(k->book_floats * sizeof(float));
    if (!k->book || fread(k->book, sizeof(float), k->book_floats, b) != k->book_floats) {
        fclose(b); snprintf(why, wl, "book read failed"); return -1;
    }
    fclose(b);
    for (int i = 0; i < nt; i++) {
        uint64_t bh = pack_fnv(k->book + k->book_off[i],
                               (size_t)k->entries * k->block * sizeof(float), PACK_FNV_SEED);
        if (bh != k->book_digest[i]) {
            snprintf(why, wl, "tensor %s fails its manifest book digest", k->name[i]);
            return -1;
        }
        uint64_t h = PACK_FNV_SEED;
        h = pack_fnv_stream(k->s, k->header + k->b0[i] * k->idxw, k->nb[i] * k->idxw, h);
        h = pack_fnv_stream(k->s, k->header + k->nblocks * k->idxw + 2 * k->b0[i], 2 * k->nb[i], h);
        h = pack_fnv_stream(k->s, k->header + k->nblocks * (k->idxw + 2) + 2 * k->b0[i], 2 * k->nb[i], h);
        if (h != k->seg_digest[i]) {
            snprintf(why, wl, "tensor %s fails its manifest segment digest", k->name[i]);
            return -1;
        }
    }
    return 0;
}

int nemo_gpu_pack_check(const char *pack_path, char *why, size_t wl) {
    pack k;
    int rc = pack_open(&k, pack_path, why, wl);
    pack_free(&k);
    return rc;
}

static int pack_find(const pack *k, const char *name) {
    for (int i = 0; i < k->nt; i++) if (!strcmp(k->name[i], name)) return i;
    return -1;
}

/* fp32 -> fp16, round to nearest even (normals and subnormals; books are finite). */
static uint16_t f32_to_f16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    uint32_t sgn = (u >> 16) & 0x8000u, a = u & 0x7fffffffu;
    if (a >= 0x47800000u) return (uint16_t)(sgn | 0x7c00u); /* overflow -> inf (not expected) */
    if (a < 0x38800000u) { /* subnormal half: exact scale by 2^24, round half even */
        float m; memcpy(&m, &a, 4);
        return (uint16_t)(sgn | (uint32_t)lrintf(m * 16777216.0f));
    }
    uint32_t v = ((a - 0x38000000u) >> 13), rem = a & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (v & 1u))) v++;
    return (uint16_t)(sgn | v);
}

static float f16_to_f32(uint16_t h) {
    uint32_t sgn = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 31u, m = h & 1023u, u;
    if (e == 0) { float f = (float)m * 5.9604644775390625e-8f; memcpy(&u, &f, 4); u |= sgn; }
    else if (e == 31) u = sgn | 0x7f800000u | (m << 13);
    else u = sgn | ((e + 112u) << 23) | (m << 13);
    float f; memcpy(&f, &u, 4); return f;
}
/* Block scales: fp16 as stored, or widened (exactly) to fp32 for k_pq_gemv_w. */
static int up_scale(nemo_gpu *g, CUdeviceptr *d, const uint8_t *src, uint64_t nb) {
    if (1) return up(g, d, src, (size_t)nb * 2); /* fp16 kept on device (VRAM); kernels widen exactly */
    float *f = (float *)malloc((size_t)nb * 4);
    if (!f) return -1;
    for (uint64_t j = 0; j < nb; j++) { uint16_t h; memcpy(&h, src + 2 * j, 2); f[j] = f16_to_f32(h); }
    int rc = up(g, d, f, (size_t)nb * 4);
    free(f);
    return rc;
}

static int t_tiling_ok(int rows, int cols) {
    int P = 1;
    while (P <= 8 && ((uint64_t)P * (uint64_t)cols) % 256u) P *= 2;
    return P <= 8 && rows % (4 * P) == 0;
}
static int up_mc(nemo_gpu *g, CUdeviceptr *d, const uint8_t *mn, const uint8_t *sc, uint64_t nb) {
    uint32_t *f = (uint32_t *)malloc((size_t)nb * 4);
    if (!f) return -1;
    for (uint64_t j = 0; j < nb; j++) {
        uint16_t a, b; memcpy(&a, mn + 2 * j, 2); memcpy(&b, sc + 2 * j, 2);
        f[j] = (uint32_t)a | ((uint32_t)b << 16);
    }
    int rc = up(g, d, f, (size_t)nb * 4);
    free(f);
    return rc;
}
static int up_pq2(nemo_gpu *g, pack *k, const nemo_checkpoint *cp, const char *name, const char *pname, int rows, int cols, gpq *o) {
    int i = pack_find(k, pname);
    const nemo_tensor_info *t = nemo_find(cp, name);
    if (i < 0 || !t) { snprintf(g->err, sizeof g->err, "projection %s not in pack/checkpoint", name); return -1; }
    uint64_t cnt = (uint64_t)rows * cols;
    if (t->count != cnt || k->nb[i] * k->block != cnt || cols % 16) {
        snprintf(g->err, sizeof g->err, "projection %s geometry mismatch", name); return -1;
    }
    uint64_t nb = k->nb[i];
    size_t isz = (size_t)(nb * k->idxw), fsz = (size_t)(nb * 2);
    uint8_t *buf = (uint8_t *)malloc(isz), *mnb = (uint8_t *)malloc(fsz);
    if (!buf || !mnb) { free(buf); free(mnb); return -1; }
    int legacy = g->half_book != 6 || !t_tiling_ok(rows, cols);   /* fp16 frames needed too */
    int rc = -1;
    uint64_t h = PACK_FNV_SEED;
    if (seek64(k->s, k->header + k->b0[i] * k->idxw) || fread(buf, 1, isz, k->s) != isz) goto done;
    h = pack_fnv(buf, isz, h);
    if (up(g, &o->idx, buf, isz)) goto done;
    if (seek64(k->s, k->header + k->nblocks * k->idxw + 2 * k->b0[i]) || fread(buf, 1, fsz, k->s) != fsz) goto done;
    h = pack_fnv(buf, fsz, h);
    memcpy(mnb, buf, fsz);
    if (legacy && up_scale(g, &o->mn, buf, nb)) goto done;
    if (seek64(k->s, k->header + k->nblocks * (k->idxw + 2) + 2 * k->b0[i]) || fread(buf, 1, fsz, k->s) != fsz) goto done;
    h = pack_fnv(buf, fsz, h);
    if (h != k->seg_digest[i]) {
        snprintf(g->err, sizeof g->err, "tensor %s fails its manifest segment digest", pname);
        goto done;
    }
    if (legacy && up_scale(g, &o->sc, buf, nb)) goto done;
    if (up_mc(g, &o->mc, mnb, buf, nb)) goto done;
    {
        size_t nf = (size_t)k->entries * k->block;
        if (g->half_book) {
            uint16_t *hb = (uint16_t *)malloc(nf * 2);
            if (!hb) goto done;
            for (size_t j = 0; j < nf; j++) hb[j] = f32_to_f16(k->book[k->book_off[i] + j]);
            int r2 = up(g, &o->bookh, hb, nf * 2);
            free(hb);
            if (r2) goto done;
        } else if (up(g, &o->book, k->book + k->book_off[i], nf * sizeof(float))) goto done;
    }
    o->rows = rows; o->cols = cols; o->pidx = i;
    {
        size_t bb = (size_t)k->entries * k->block * (g->half_book ? 2 : sizeof(float));
        size_t fr = (size_t)nb * 4 + (legacy ? 2 * fsz : 0);   /* frames on the device */
        g->info.pack_bytes += isz + fr + bb;
        g->info.bytes_per_token += isz + (size_t)nb * 4 + bb;
    }
    rc = 0;
done:
    if (rc && !g->err[0]) snprintf(g->err, sizeof g->err, "reading %s from the pack failed", pname);
    free(buf); free(mnb);
    return rc;
}
static int up_pq(nemo_gpu *g, pack *k, const nemo_checkpoint *cp, const char *name, int rows, int cols, gpq *o) {
    return up_pq2(g, k, cp, name, name, rows, cols, o);
}
/* Expanded pack (2026-11-27): a projection may carry a refinement tensor
 * "<name>.refine", the PQ-packed residual W_original - W_decoded, stored in
 * the same pack with the same geometry. Its GEMV is added to the base
 * GEMV's output (y = W x + R x, fixed order), so the layer computes with
 * W + R. A pack without refinements loads exactly as before. */
typedef struct { char name[300], base[256]; int rows, cols, layer, pidx; gpq *r; } rpend;

static int up_pqr(nemo_gpu *g, pack *k, const nemo_checkpoint *cp, const char *name, int rows, int cols, gpq *o, gpq *r, rpend *pend, int *npend) {
    if (up_pq(g, k, cp, name, rows, cols, o)) return -1;
    char rn[300];
    snprintf(rn, sizeof rn, "%s.refine", name);
    int i = pack_find(k, rn);
    if (i < 0) return 0;
    rpend *p = &pend[(*npend)++];
    memset(p, 0, sizeof *p);
    snprintf(p->name, sizeof p->name, "%s", rn);
    snprintf(p->base, sizeof p->base, "%s", name);
    p->rows = rows; p->cols = cols; p->r = r; p->pidx = i;
    const char *ls = strstr(name, "layers.");
    p->layer = ls ? atoi(ls + 7) : 999;
    return 0;
}

/* Bytes a refine tensor will occupy on the device (conservative: include
 * the compact frame and the half book). */
static size_t refine_bytes_of(const pack *k, int i, int half_book) {
    uint64_t nb = k->nb[i];
    size_t isz = (size_t)(nb * k->idxw), fsz = (size_t)(nb * 2);
    size_t bb = (size_t)k->entries * k->block * (half_book ? 2 : 4);
    return isz + (size_t)nb * 4 + 2 * fsz + bb;
}

/* Default layer order, from the 27-28 Nov 2026 probe (best pair R8+R11 first).
 * Overridden by NEMO_GPU_REFINE=all|off|auto|11,10,8,... */
static int refine_pri(int layer, const char *spec) {
    if (spec && !strcmp(spec, "off")) return 10000;
    if (spec && strcmp(spec, "all") && strcmp(spec, "auto") && spec[0]) {
        char buf[256]; snprintf(buf, sizeof buf, "%s", spec);
        int p = 0;
        for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ","), p++) {
            while (*tok == ' ') tok++;
            if (isdigit((unsigned char)*tok) && atoi(tok) == layer) return p;
        }
        return 10000;
    }
    static const int pref[] = {11, 8, 10, 21, 14, 9, 20, 15, 12, 13, 22, 23, -1};
    for (int i = 0; pref[i] >= 0; i++) if (pref[i] == layer) return i;
    return 100 + layer;
}

/* Upload whole layers in priority order while they fit in `budget`. A
 * layer is kept or skipped as a unit, so a projection is never half-refined. */
static int up_refine_fit(nemo_gpu *g, pack *k, const nemo_checkpoint *cp, rpend *pend, int npend, size_t budget) {
    if (npend <= 0) return 0;
    int order[224];
    for (int i = 0; i < npend; i++) order[i] = i;
    for (int a = 0; a < npend; a++) for (int b = a + 1; b < npend; b++) {
        int pa = refine_pri(pend[order[a]].layer, getenv("NEMO_GPU_REFINE"));
        int pb = refine_pri(pend[order[b]].layer, getenv("NEMO_GPU_REFINE"));
        if (pb < pa || (pb == pa && pend[order[b]].layer < pend[order[a]].layer)) {
            int t = order[a]; order[a] = order[b]; order[b] = t;
        }
    }
    size_t left = budget;
    int skipped = 0;
    for (int i = 0; i < npend; ) {
        int layer = pend[order[i]].layer, j = i, take = 1;
        size_t need = 0;
        while (j < npend && pend[order[j]].layer == layer) {
            need += refine_bytes_of(k, pend[order[j]].pidx, g->half_book);
            j++;
        }
        if (need > left || refine_pri(layer, getenv("NEMO_GPU_REFINE")) >= 10000) take = 0;
        for (int t = i; t < j; t++) {
            rpend *p = &pend[order[t]];
            if (!take) { skipped++; continue; }
            uint64_t before = g->info.pack_bytes;
            if (up_pq2(g, k, cp, p->base, p->name, p->rows, p->cols, p->r)) return -1;
            g->nref++;
            g->ref_bytes += g->info.pack_bytes - before;
        }
        if (take) left -= need;
        i = j;
    }
    g->info.refine_skipped = skipped;
    g->info.refine_budget = budget;
    return 0;
}

static float host_dot(const int8_t *q, const float *x) {
#ifdef __AVX2__
    __m256 acc = _mm256_setzero_ps();
    int c = 0;
    for (; c + 16 <= H; c += 16) {
        __m128i v = _mm_loadu_si128((const __m128i *)(q + c));
        __m256i i16 = _mm256_cvtepi8_epi16(v);
        __m256i lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(i16));
        __m256i hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(i16, 1));
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(lo), _mm256_loadu_ps(x + c), acc);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(hi), _mm256_loadu_ps(x + c + 8), acc);
    }
    float t[8]; _mm256_storeu_ps(t, acc);
    float s = t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
    for (; c < H; c++) s += (float)q[c] * x[c];
    return s;
#else
    float s = 0.0f;
    for (int c = 0; c < H; c++) s += (float)q[c] * x[c];
    return s;
#endif
}
typedef struct { const int8_t *q; const float *sc, *x; float *y; } host_job;
static void host_rows(void *ctx, int begin, int end) {
    host_job *j = ctx;
    for (int r = begin; r < end; r++) j->y[r] = j->sc[r] * host_dot(j->q + (size_t)r * H, j->x);
}

/* ---- launches ---- */
static double now_s(void) {
#ifdef _WIN32
    LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
#else
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}
/* NEMO_GPU_PROFILE=1: synchronize around every launch and add its wall
 * time to a per-kernel total, printed to stderr at close. Diagnostic only. */
static int g_prof = -1;
static double g_prof_t[K_N];
static int g_shape[32][2], g_shape_k[32]; static double g_shape_t[32]; static long g_shape_n[32];
static long g_prof_n[K_N];
static int launch_sm(nemo_gpu *g, int k, unsigned gx, unsigned gy, unsigned bx, unsigned smem, void **args);
static int launch(nemo_gpu *g, int k, unsigned gx, unsigned gy, unsigned bx, void **args) {
    return launch_sm(g, k, gx, gy, bx, 0, args);
}
static int launch_sm(nemo_gpu *g, int k, unsigned gx, unsigned gy, unsigned bx, unsigned smem, void **args) {
    if (g_prof < 0) { const char *p = getenv("NEMO_GPU_PROFILE"); g_prof = p && *p == '1'; }
    double t0 = 0;
    if (g_prof) { g->cu.CtxSynchronize(); t0 = now_s(); }
    int rc = ck(g, g->cu.LaunchKernel(g->fn[k], gx, gy, 1, bx, 1, 1, smem, NULL, args, NULL), kname[k]);
    if (g_prof) { g->cu.CtxSynchronize(); double dt = now_s() - t0; g_prof_t[k] += dt; g_prof_n[k]++;
        if ((k >= K_PQF && k <= K_PQT) || (k >= K_GT8 && k <= K_GX16) || k == K_PQ || k == K_PQFX || k == K_GF16) { int ai = (k == K_PQ || k == K_PQT || (k >= K_GT8 && k <= K_GX16) || k == K_PQFX || k == K_GF16) ? 5 : k >= K_PQV ? 6 : 7; int rr = *(int *)args[ai], cc = *(int *)args[ai + 1];
            for (int q = 0; q < 32; q++) { if (!g_shape[q][0] || (g_shape[q][0] == rr && g_shape[q][1] == cc && g_shape_k[q] == k)) { g_shape[q][0] = rr; g_shape[q][1] = cc; g_shape_k[q] = k; g_shape_t[q] += dt; g_shape_n[q]++; break; } } } }
    return rc;
}
/* Decode GEMV choices: DTC t1k (default) or t (NEMO_GPU_GEMV), same
 * per-row order, bit-identical. FULL (NEMO_GPU_GEMV_FULL): "f" (default,
 * the fast path: smem-staged fp32 book, row-tiled, its own summation order,
 * within tolerance of the old FULL) or the old exact path "exact"/"x4"
 * (same as "x1", bit-identical to the C reference order). Prefill follows
 * the same choice, so batched == stepped on either path. */
static void kernel_choice(nemo_gpu *g) {
    const char *v = getenv("NEMO_GPU_GEMV"); g->gemv_k = v && !strcmp(v, "t") ? K_PQT : K_PQT1K;
    v = getenv("NEMO_GPU_GEMV_FULL");
    g->gemv_x = v && !strcmp(v, "x1") ? K_PQ : v && (!strcmp(v, "x4") || !strcmp(v, "exact")) ? K_PQX4 : K_PQFX;
}
/* Row tiling of the FULL fast path (as DTC): P rows apart share a block
 * offset; 0 when the shape does not tile (then the exact kernels run). */
static int full_tile(nemo_gpu *g, int rows, int cols) {
    if (g->gemv_x != K_PQFX) return 0;
    int P = 1;
    while (P <= 8 && ((uint64_t)P * (uint64_t)cols) % 256u) P *= 2;
    return P <= 8 && rows % (4 * P) == 0 ? P : 0;
}
static int pq_gemv(nemo_gpu *g, const gpq *w, CUdeviceptr x, CUdeviceptr y) {
    if (g->exact && w->book) {
        int P = full_tile(g, w->rows, w->cols);
        if (P) {
            int nwarps = w->rows / 4;
            void *a[] = {(void *)&w->idx, (void *)&w->mc, (void *)&w->book, &x, &y, (void *)&w->rows, (void *)&w->cols, &P};
            return launch_sm(g, K_PQFX, (unsigned)((nwarps + 31) / 32), 1, 1024, 65536u, a);
        }
        int k = g->gemv_x == K_PQFX ? K_PQX4 : g->gemv_x;
        void *a[] = {(void *)&w->idx, (void *)&w->mc, (void *)&w->book, &x, &y, (void *)&w->rows, (void *)&w->cols};
        return launch(g, k, (unsigned)((w->rows + 7) / 8), 1, 256, a);
    }
    if (g->half_book == 6) {
        int P = 1;
        while (P <= 8 && ((uint64_t)P * (uint64_t)w->cols) % 256u) P *= 2;
        if (P <= 8 && w->rows % (4 * P) == 0) {
            int nwarps = w->rows / 4;
            void *a[] = {(void *)&w->idx, (void *)&w->mc, (void *)&w->bookh, &x, &y, (void *)&w->rows, (void *)&w->cols, &P};
            if (g->gemv_k == K_PQT1K) return launch_sm(g, K_PQT1K, (unsigned)((nwarps + 31) / 32), 1, 1024, 65536u, a);
            return launch_sm(g, K_PQT, (unsigned)((nwarps + 15) / 16), 1, 512, 65536u, a);
        }
    }
    if (g->half_book >= 5) {
        int rpw = w->rows >= 64 * 60 ? 4 : w->rows >= 32 * 60 ? 2 : 1;
        void *a[] = {(void *)&w->idx, (void *)&w->mn, (void *)&w->sc, (void *)&w->bookh, &x, &y, (void *)&w->rows, (void *)&w->cols, &rpw};
        return launch_sm(g, K_PQW, (unsigned)((w->rows + 16 * rpw - 1) / (16 * rpw)), 1, 512, 65536u, a);
    }
    if (g->half_book == 4 && (w->cols & 63) == 0) {
        int rpw = w->rows >= 64 * 60 ? 4 : w->rows >= 32 * 60 ? 2 : 1;
        static int dbg = -1;
        if (dbg < 0) { const char *d = getenv("NEMO_GPU_DBG"); dbg = d ? atoi(d) : 0; }
        void *a[] = {(void *)&w->idx, (void *)&w->mn, (void *)&w->sc, (void *)&w->bookh, &x, &y, (void *)&w->rows, (void *)&w->cols, &rpw, &dbg};
        return launch_sm(g, K_PQV, (unsigned)((w->rows + 16 * rpw - 1) / (16 * rpw)), 1, 512, 65536u, a);
    }
    int n16 = w->cols / 16;
    void *s[] = {&x, &g->xs, &n16};
    if (launch(g, K_XSUM, (unsigned)((n16 + 255) / 256), 1, 256, s)) return -1;
    if (g->half_book == 3 && (w->cols & 63) == 0) {
        int rpw = w->rows >= 128 * 60 ? 4 : w->rows >= 64 * 60 ? 2 : 1;
        void *a[] = {(void *)&w->idx, (void *)&w->mn, (void *)&w->sc, (void *)&w->bookh, &x, &g->xs, &y, (void *)&w->rows, (void *)&w->cols, &rpw};
        return launch_sm(g, K_PQU, (unsigned)((w->rows + 32 * rpw - 1) / (32 * rpw)), 1, 1024, 65536u, a);
    }
    if (g->half_book >= 2) {
        int rpw = w->rows >= 128 * 60 ? 4 : w->rows >= 64 * 60 ? 2 : 1;
        void *a[] = {(void *)&w->idx, (void *)&w->mn, (void *)&w->sc, (void *)&w->bookh, &x, &g->xs, &y, (void *)&w->rows, (void *)&w->cols, &rpw};
        return launch_sm(g, K_PQS, (unsigned)((w->rows + 32 * rpw - 1) / (32 * rpw)), 1, 1024, 65536u, a);
    }
    if (g->half_book) {
        void *a[] = {(void *)&w->idx, (void *)&w->mn, (void *)&w->sc, (void *)&w->bookh, &x, &g->xs, &y, (void *)&w->rows, (void *)&w->cols};
        return launch(g, K_PQH, (unsigned)((w->rows + 7) / 8), 1, 256, a);
    }
    void *a[] = {(void *)&w->idx, (void *)&w->mn, (void *)&w->sc, (void *)&w->book, &x, &g->xs, &y, (void *)&w->rows, (void *)&w->cols};
    return launch(g, K_PQF, (unsigned)((w->rows + 7) / 8), 1, 256, a);
}
static int rmsnorm(nemo_gpu *g, CUdeviceptr x, CUdeviceptr w, CUdeviceptr o) {
    int n = H; float eps = EPSF;
    void *a[] = {&x, &w, &o, &n, &eps};
    return launch(g, K_RMS, 1, 1, 1024, a);
}
static int add(nemo_gpu *g, CUdeviceptr x, CUdeviceptr y, int n) {
    void *a[] = {&x, &y, &n};
    return launch(g, K_ADD, (unsigned)((n + 255) / 256), 1, 256, a);
}
/* Projection with its optional refinement: y = W x, then y += R x. */
static int pqr_gemv(nemo_gpu *g, const gpq *w, const gpq *r, CUdeviceptr x, CUdeviceptr y) {
    if (pq_gemv(g, w, x, y)) return -1;
    if (!r->rows) return 0;
    if (pq_gemv(g, r, x, g->rt)) return -1;
    return add(g, y, g->rt, r->rows);
}
static int pq_gemm(nemo_gpu *g, const gpq *w, CUdeviceptr X, CUdeviceptr Y, int n);
static int pqr_gemm(nemo_gpu *g, const gpq *w, const gpq *r, CUdeviceptr X, CUdeviceptr Y, int n) {
    if (pq_gemm(g, w, X, Y, n)) return -1;
    if (!r->rows) return 0;
    if (pq_gemm(g, r, X, g->brt, n)) return -1;
    return add(g, Y, g->brt, r->rows * n);
}

static int run_layer(nemo_gpu *g, int l) {
    glayer *L = &g->L[l];
    if (rmsnorm(g, g->x, L->norm_w, g->h)) return -1;
    if (L->type == 'M') {
        if (pqr_gemv(g, &L->p0, &L->r0, g->h, g->proj)) return -1;
        CUdeviceptr u = g->proj + (CUdeviceptr)INNER * 4, dt = g->proj + (CUdeviceptr)(INNER + CONV) * 4;
        int ch = CONV;
        void *ac[] = {&L->win, &u, &L->conv_w, &L->conv_b, &g->v, &ch};
        if (launch(g, K_CONV, (CONV + 255) / 256, 1, 256, ac)) return -1;
        CUdeviceptr Bp = g->v + (CUdeviceptr)INNER * 4, Cp = Bp + (CUdeviceptr)NGROUP * NSTATE * 4;
        int heads = NHEAD, hd = HDIM, ns = NSTATE, gr = NGROUP;
        void *as[] = {&L->ssm, &g->v, &Bp, &Cp, &dt, &L->a_log, &L->d, &L->dt_bias, &g->y, &heads, &hd, &ns, &gr};
        if (launch(g, K_SSM, (NHEAD * HDIM + 127) / 128, 1, 128, as)) return -1;
        int gs = INNER / 8; float eps = EPSF;
        void *an[] = {&g->y, &g->proj, &L->gnorm_w, &g->inner, &gs, &eps};
        if (launch(g, K_GNORM, 8, 1, 1024, an)) return -1;
        if (pqr_gemv(g, &L->p1, &L->r1, g->inner, g->y)) return -1;
    } else if (L->type == '-') {
        if (pqr_gemv(g, &L->p0, &L->r0, g->h, g->mid)) return -1;
        int n = MLP;
        void *ar[] = {&g->mid, &n};
        if (launch(g, K_RELU2, (MLP + 255) / 256, 1, 256, ar)) return -1;
        if (pqr_gemv(g, &L->p1, &L->r1, g->mid, g->y)) return -1;
    } else {
        if (pqr_gemv(g, &L->p0, &L->r0, g->h, g->q) || pqr_gemv(g, &L->p1, &L->r1, g->h, g->k) || pqr_gemv(g, &L->p2, &L->r2, g->h, g->vv)) return -1;
        int pos = g->len, row = NKV * AD;
        void *ak[] = {&L->kc, &L->vc, &g->k, &g->vv, &pos, &row};
        if (launch(g, K_KVPUSH, (row + 255) / 256, 1, 256, ak)) return -1;
        int len = g->len + 1, nq = NQ, nkv = NKV, d = AD, mp = g->max_pos;
        void *a1[] = {&g->q, &L->kc, &g->scores, &len, &nq, &nkv, &d, &mp};
        if (launch(g, K_ASCORE, (unsigned)((len + 127) / 128), NQ, 128, a1)) return -1;
        void *a2[] = {&g->scores, &L->vc, &g->inner, &len, &nq, &nkv, &d, &mp};
        if (launch(g, K_AOUT, NQ, 1, AD, a2)) return -1;
        if (pqr_gemv(g, &L->p3, &L->r3, g->inner, g->y)) return -1;
    }
    return add(g, g->x, g->y, H);
}

static int hook_call(nemo_gpu *g, int layer) {
    static float hx[H];
    if (ck(g, g->cu.MemcpyDtoH(hx, g->x, sizeof hx), "cuMemcpyDtoH")) return -1;
    g->hook(g->hook_user, layer, hx);
    return 0;
}

int nemo_gpu_step(nemo_gpu *g, int token, float *logits) {
    if (!g->valid) return -2;
    if (token < 0 || token >= VOCAB || g->len >= g->max_pos) return -1;
    uint16_t row[H]; float xf[H];
    if (seek64(g->emb_f, g->emb_off + (uint64_t)token * H * 2) || fread(row, 2, H, g->emb_f) != H) {
        snprintf(g->err, sizeof g->err, "embedding read failed"); return -1;
    }
    for (int i = 0; i < H; i++) { uint32_t u = (uint32_t)row[i] << 16; memcpy(&xf[i], &u, 4); }
    g->valid = 0;
    if (ck(g, g->cu.MemcpyHtoD(g->x, xf, sizeof xf), "cuMemcpyHtoD")) return -1;
    for (int l = 0; l < g->layer_limit; l++) {
        if (g->hook && hook_call(g, l)) return -1;
        if (run_layer(g, l)) return -1;
    }
    if (g->hook && hook_call(g, g->layer_limit)) return -1;
    g->len++;
    g->valid = 1;
    if (logits) return nemo_gpu_logits(g, logits);
    return ck(g, g->cu.CtxSynchronize(), "step sync");
}

int nemo_gpu_logits(nemo_gpu *g, float *logits) {
    if (!g->valid) return -2;
    if (rmsnorm(g, g->x, g->norm_f, g->h)) return -1;
    if (g->head_host) {
        if (ck(g, g->cu.MemcpyDtoH(g->h_host, g->h, (size_t)H * 4), "hidden copy")) return -1;
        host_job j = { g->head_q, g->head_sc, g->h_host, logits };
        if (g->head_pool_ok) nemo_pool_for(&g->head_pool, VOCAB, host_rows, &j);
        else host_rows(&j, 0, VOCAB);
        return 0;
    }
    int rows = VOCAB, cols = H;
    if (g->head_i8) {
        void *a[] = {&g->lm_head, &g->head_scale, &g->h, &g->logits, &rows, &cols};
        if (launch(g, K_I8, VOCAB / 8, 1, 256, a)) return -1;
    } else {
        void *a[] = {&g->lm_head, &g->h, &g->logits, &rows, &cols};
        if (launch(g, K_BF16, VOCAB / 8, 1, 256, a)) return -1;
    }
    return ck(g, g->cu.MemcpyDtoH(logits, g->logits, (size_t)VOCAB * 4), "logits copy");
}

/* ---- batched prefill (tokens in) ----
 * nemo_gpu_prefill runs up to `chunk` prompt tokens through each layer
 * together. Projections are PQ GEMMs (k_pq_gemm_t* for DTC, k_pq_gemm_x*
 * for FULL) that keep every output's per-row, per-lane summation order,
 * the recurrences (conv, SSM) loop over the tokens inside the thread, and
 * attention runs the decode kernels once per token at its own length, so
 * the state and residual after a prefill are bit-identical to stepping
 * the same tokens one by one with nemo_gpu_step. */
static int pq_gemm(nemo_gpu *g, const gpq *w, CUdeviceptr X, CUdeviceptr Y, int n) {
    int tb = g->ptb;
    unsigned gy = (unsigned)((n + tb - 1) / tb);
    if (g->exact && w->book) {
        int P = full_tile(g, w->rows, w->cols);
        if (P) {
            int nwarps = w->rows / 4;
            void *a[] = {(void *)&w->idx, (void *)&w->mc, (void *)&w->book, &X, &Y, (void *)&w->rows, (void *)&w->cols, &P, &n};
            return launch_sm(g, K_GF16, (unsigned)((nwarps + 15) / 16), (unsigned)((n + 15) / 16), 512, 65536u, a);
        }
        void *a[] = {(void *)&w->idx, (void *)&w->mc, (void *)&w->book, &X, &Y, (void *)&w->rows, (void *)&w->cols, &n};
        return launch_sm(g, K_GX16, (unsigned)((w->rows + 7) / 8), (unsigned)((n + 15) / 16), 256, 0, a);
    }
    if (g->half_book == 6) {
        int P = 1;
        while (P <= 8 && ((uint64_t)P * (uint64_t)w->cols) % 256u) P *= 2;
        int R = g->gemm_r;
        if (P <= 8 && w->rows % (R * P) == 0) {
            int nwarps = w->rows / R;
            void *a[] = {(void *)&w->idx, (void *)&w->mc, (void *)&w->bookh, &X, &Y, (void *)&w->rows, (void *)&w->cols, &P, &n};
            if (g->gemm_k == K_GT8K) return launch_sm(g, K_GT8K, (unsigned)((nwarps + 31) / 32), gy, 1024, 65536u, a);
            return launch_sm(g, g->gemm_k, (unsigned)((nwarps + 15) / 16), gy, 512, 65536u, a);
        }
    }
    /* any other kernel choice: the single-token GEMV per token (same result by construction) */
    for (int t = 0; t < n; t++)
        if (pq_gemv(g, w, X + (CUdeviceptr)t * w->cols * 4, Y + (CUdeviceptr)t * w->rows * 4)) return -1;
    return 0;
}

static int run_layer_b(nemo_gpu *g, int l, int n, int pos0) {
    glayer *L = &g->L[l];
    int hn = H; float eps = EPSF;
    { void *a[] = {&g->bx, &L->norm_w, &g->bh, &hn, &eps};
      if (launch_sm(g, K_RMSB, (unsigned)n, 1, 1024, 0, a)) return -1; }
    if (L->type == 'M') {
        if (pqr_gemm(g, &L->p0, &L->r0, g->bh, g->bproj, n)) return -1;
        CUdeviceptr u = g->bproj + (CUdeviceptr)INNER * 4, dt = g->bproj + (CUdeviceptr)(INNER + CONV) * 4;
        int us = PROJ, os = CONV, ch = CONV;
        void *ac[] = {&L->win, &u, &us, &L->conv_w, &L->conv_b, &g->bv, &os, &ch, &n};
        if (launch_sm(g, K_CONVB, (CONV + 255) / 256, 1, 256, 0, ac)) return -1;
        int xs = CONV, ds = PROJ, ys = INNER, heads = NHEAD, hd = HDIM, ns = NSTATE, gr = NGROUP, inner = INNER;
        void *as[] = {&L->ssm, &g->bv, &xs, &dt, &ds, &L->a_log, &L->d, &L->dt_bias, &g->by, &ys, &heads, &hd, &ns, &gr, &inner, &n};
        if (launch_sm(g, K_SSMB, (NHEAD * HDIM + 127) / 128, 1, 128, 0, as)) return -1;
        int gs = INNER / 8, zs = PROJ, o2 = INNER;
        void *an[] = {&g->by, &ys, &g->bproj, &zs, &L->gnorm_w, &g->binner, &o2, &gs, &eps};
        if (launch_sm(g, K_GNORMB, 8, (unsigned)n, 1024, 0, an)) return -1;
        if (pqr_gemm(g, &L->p1, &L->r1, g->binner, g->by, n)) return -1;
    } else if (L->type == '-') {
        if (pqr_gemm(g, &L->p0, &L->r0, g->bh, g->bmid, n)) return -1;
        int m = MLP * n;
        void *ar[] = {&g->bmid, &m};
        if (launch(g, K_RELU2, (unsigned)((m + 255) / 256), 1, 256, ar)) return -1;
        if (pqr_gemm(g, &L->p1, &L->r1, g->bmid, g->by, n)) return -1;
    } else {
        if (pqr_gemm(g, &L->p0, &L->r0, g->bh, g->bq, n) || pqr_gemm(g, &L->p1, &L->r1, g->bh, g->bk, n) || pqr_gemm(g, &L->p2, &L->r2, g->bh, g->bvv, n)) return -1;
        int row = NKV * AD, pos = pos0;
        void *ak[] = {&L->kc, &L->vc, &g->bk, &g->bvv, &pos, &row};
        if (launch_sm(g, K_KVPB, (unsigned)((row + 255) / 256), (unsigned)n, 256, 0, ak)) return -1;
        for (int t = 0; t < n; t++) {
            int len = pos0 + t + 1, nq = NQ, nkv = NKV, d = AD, mp = g->max_pos;
            CUdeviceptr qt = g->bq + (CUdeviceptr)t * NQ * AD * 4, ot = g->binner + (CUdeviceptr)t * NQ * AD * 4;
            void *a1[] = {&qt, &L->kc, &g->scores, &len, &nq, &nkv, &d, &mp};
            if (launch(g, K_ASCORE, (unsigned)((len + 127) / 128), NQ, 128, a1)) return -1;
            void *a2[] = {&g->scores, &L->vc, &ot, &len, &nq, &nkv, &d, &mp};
            if (launch(g, K_AOUT, NQ, 1, AD, a2)) return -1;
        }
        if (pqr_gemm(g, &L->p3, &L->r3, g->binner, g->by, n)) return -1;
    }
    return add(g, g->bx, g->by, H * n);
}

int nemo_gpu_prefill_chunk(const nemo_gpu *g) { return g->chunk; }
int nemo_gpu_full_fast(const nemo_gpu *g) { return g->gemv_x == K_PQFX; }

int nemo_gpu_prefill(nemo_gpu *g, const int *ids, int n) {
    if (!g->valid) return -2;
    if (n < 0 || g->len + n > g->max_pos) return -1;
    for (int i = 0; i < n; i++) if (ids[i] < 0 || ids[i] >= VOCAB) return -1;
    if (!g->chunk || g->hook) {
        for (int i = 0; i < n; i++) if (nemo_gpu_step(g, ids[i], NULL)) return -1;
        return 0;
    }
    for (int done = 0; done < n;) {
        int c = n - done < g->chunk ? n - done : g->chunk;
        uint16_t row[H];
        for (int t = 0; t < c; t++) {
            if (seek64(g->emb_f, g->emb_off + (uint64_t)ids[done + t] * H * 2) || fread(row, 2, H, g->emb_f) != H) {
                snprintf(g->err, sizeof g->err, "embedding read failed"); return -1;
            }
            float *xf = g->emb_host + (size_t)t * H;
            for (int i = 0; i < H; i++) { uint32_t u = (uint32_t)row[i] << 16; memcpy(&xf[i], &u, 4); }
        }
        g->valid = 0;
        if (ck(g, g->cu.MemcpyHtoD(g->bx, g->emb_host, (size_t)c * H * 4), "cuMemcpyHtoD")) return -1;
        for (int l = 0; l < g->layer_limit; l++)
            if (run_layer_b(g, l, c, g->len)) return -1;
        /* the residual of the chunk's last token is the stream nemo_gpu_logits reads */
        if (ck(g, g->cu.MemcpyDtoD(g->x, g->bx + (CUdeviceptr)(c - 1) * H * 4, H * 4), "cuMemcpyDtoD")) return -1;
        g->len += c;
        g->valid = 1;
        done += c;
    }
    return ck(g, g->cu.CtxSynchronize(), "prefill sync");
}

/* Workspace for the batched prefill; optional (0 = per-token prefill). */
static void prefill_resources(nemo_gpu *g) {
    const char *e = getenv("NEMO_GPU_PREFILL_CHUNK");
    int ch = e && *e ? atoi(e) : 64;
    /* tile choice for the DTC GEMM: t16 (4 rows x 16 tokens, default), t8, r8t8, r8t4 */
    const char *t = getenv("NEMO_GPU_GEMM");
    g->gemm_k = K_GT16; g->gemm_r = 4; g->ptb = 16;
    if (t && !strcmp(t, "t8")) { g->gemm_k = K_GT8; g->ptb = 8; }
    else if (t && !strcmp(t, "r8t8")) { g->gemm_k = K_GR8T8; g->gemm_r = 8; g->ptb = 8; }
    else if (t && !strcmp(t, "r8t4")) { g->gemm_k = K_GR8T4; g->gemm_r = 8; g->ptb = 4; }
    else if (t && !strcmp(t, "r2t32")) { g->gemm_k = K_GR2T32; g->gemm_r = 2; g->ptb = 32; }
    else if (t && !strcmp(t, "t8k")) { g->gemm_k = K_GT8K; g->ptb = 8; }
    if (ch <= 1) { g->chunk = 0; return; }
    if (ch > 512) ch = 512;
    ch = (ch + 31) / 32 * 32;   /* GEMM tiles read whole token groups (32 covers every tile) */
    size_t c = (size_t)ch;
    size_t need = c * 4 * ((size_t)H * 2 + PROJ + CONV + INNER * 2 + MLP + NQ * AD + NKV * AD * 2 + (g->nref ? PROJ : 0));
    size_t fr = 0, tt = 0; g->cu.MemGetInfo(&fr, &tt);
    if (fr < need + (48u << 20)) { g->chunk = 0; return; }
    g->emb_host = (float *)malloc(c * H * 4);
    if (!g->emb_host
        || dalloc(g, &g->bx, c * H * 4) || dalloc(g, &g->bh, c * H * 4) || dalloc(g, &g->bproj, c * PROJ * 4)
        || dalloc(g, &g->bv, c * CONV * 4) || dalloc(g, &g->by, c * INNER * 4) || dalloc(g, &g->binner, c * INNER * 4)
        || dalloc(g, &g->bmid, c * MLP * 4) || dalloc(g, &g->bq, c * NQ * AD * 4) || dalloc(g, &g->bk, c * NKV * AD * 4)
        || dalloc(g, &g->bvv, c * NKV * AD * 4) || (g->nref && dalloc(g, &g->brt, c * PROJ * 4))) {
        CUdeviceptr *b[] = {&g->bx, &g->bh, &g->bproj, &g->bv, &g->by, &g->binner, &g->bmid, &g->bq, &g->bk, &g->bvv, &g->brt};
        for (size_t i = 0; i < sizeof b / sizeof b[0]; i++) if (*b[i]) { g->cu.MemFree(*b[i]); *b[i] = 0; }
        free(g->emb_host); g->emb_host = NULL;
        g->chunk = 0; return;
    }
    /* zero the padding rows the GEMM tiles read past the last token */
    g->cu.MemsetD8(g->bx, 0, c * H * 4); g->cu.MemsetD8(g->bh, 0, c * H * 4);
    g->cu.MemsetD8(g->binner, 0, c * INNER * 4); g->cu.MemsetD8(g->bmid, 0, c * MLP * 4);
    g->chunk = ch;
}

void nemo_gpu_reset(nemo_gpu *g) {
    for (int l = 0; l < g->layer_limit; l++) {
        glayer *L = &g->L[l];
        if (L->type == 'M') {
            g->cu.MemsetD8(L->ssm, 0, (size_t)NHEAD * HDIM * NSTATE * 4);
            g->cu.MemsetD8(L->win, 0, (size_t)CONV * 4 * 4);
        }
    }
    g->len = 0;
    g->valid = g->cu.CtxSynchronize() == 0;
}

int nemo_gpu_pos(const nemo_gpu *g) { return g->len; }

void nemo_gpu_set_layer_hook(nemo_gpu *g, void (*hook)(void *, int, const float *), void *user) {
    g->hook = hook; g->hook_user = user;
}

void nemo_gpu_info_get(const nemo_gpu *g, nemo_gpu_info *info) { *info = g->info; }

void nemo_gpu_close(nemo_gpu *g) {
    if (!g) return;
    if (g_prof > 0) {
        for (int k = 0; k < K_N; k++)
            if (g_prof_n[k]) fprintf(stderr, "profile %-16s %8ld launches %9.1f ms total %8.1f us each\n", kname[k], g_prof_n[k], g_prof_t[k] * 1e3, g_prof_t[k] * 1e6 / g_prof_n[k]);
        for (int q = 0; q < 32 && g_shape[q][0]; q++)
            fprintf(stderr, "profile %-14s %5dx%-5d %6ld launches %9.1f ms total %8.1f us each %6.1f GB/s idx\n", kname[g_shape_k[q]], g_shape[q][0], g_shape[q][1], g_shape_n[q], g_shape_t[q] * 1e3, g_shape_t[q] * 1e6 / g_shape_n[q], (double)g_shape[q][0] * g_shape[q][1] / 4.0 * g_shape_n[q] / g_shape_t[q] / 1e9);
    }
    if (g->ctx_ok) {
        g->cu.CtxSetCurrent(g->ctx);
        g->cu.CtxSynchronize();
        if (g->mod) g->cu.ModuleUnload(g->mod);
        g->cu.PrimaryCtxRelease(g->dev); /* frees every allocation of the primary context */
    }
    if (g->emb_f) fclose(g->emb_f);
    free(g->emb_host);
    free(g->head_q); free(g->head_sc); free(g->h_host);
    if (g->head_pool_ok) nemo_pool_shutdown(&g->head_pool);
    dl_close(g->lrtc);
    dl_close(g->lcu);
    free(g);
}

/* ---- EXPAND resources: fp32 books for the bf16-faithful (full) kernel and
 * a copy of the recurrent state so one step can be redone at full
 * precision. Allocated after everything else; optional. */
static void expand_resources(nemo_gpu *g, const pack *k) {
    const char *off = getenv("NEMO_GPU_NO_EXPAND");
    if (g->exact || !g->half_book) { g->full_ok = 1; snprintf(g->full_why, sizeof g->full_why, "full kernel is the only kernel"); }
    else if (off && *off == '1') { snprintf(g->full_why, sizeof g->full_why, "disabled by NEMO_GPU_NO_EXPAND"); return; }
    else {
        size_t nf = (size_t)k->entries * k->block * sizeof(float);
        for (int l = 0; l < g->layer_limit; l++) {
            gpq *w[8] = {&g->L[l].p0, &g->L[l].p1, &g->L[l].p2, &g->L[l].p3, &g->L[l].r0, &g->L[l].r1, &g->L[l].r2, &g->L[l].r3};
            for (int j = 0; j < 8; j++) {
                if (!w[j]->rows || w[j]->book) continue;
                size_t fr = 0, tt = 0; g->cu.MemGetInfo(&fr, &tt);
                if (fr < nf + (48u << 20) || up(g, &w[j]->book, k->book + k->book_off[w[j]->pidx], nf)) {
                    snprintf(g->full_why, sizeof g->full_why, "VRAM: no room for the fp32 books (%.0f MiB free)", fr / 1048576.0);
                    for (int l2 = 0; l2 <= l; l2++) {
                        gpq *v[8] = {&g->L[l2].p0, &g->L[l2].p1, &g->L[l2].p2, &g->L[l2].p3, &g->L[l2].r0, &g->L[l2].r1, &g->L[l2].r2, &g->L[l2].r3};
                        for (int j2 = 0; j2 < 8; j2++) if (v[j2]->book) { g->cu.MemFree(v[j2]->book); v[j2]->book = 0; g->used -= nf; }
                    }
                    return;
                }
            }
        }
        g->full_ok = 1;
        snprintf(g->full_why, sizeof g->full_why, "fp32 books resident");
    }
    size_t ss = (size_t)NHEAD * HDIM * NSTATE * 4, ws = (size_t)CONV * 16;
    for (int l = 0; l < g->layer_limit; l++) {
        glayer *L = &g->L[l];
        if (L->type != 'M') continue;
        size_t fr = 0, tt = 0; g->cu.MemGetInfo(&fr, &tt);
        if (fr < ss + ws + (32u << 20) || dalloc(g, &L->ssm_s, ss) || dalloc(g, &L->win_s, ws)) {
            for (int l2 = 0; l2 <= l; l2++) {
                if (g->L[l2].ssm_s) { g->cu.MemFree(g->L[l2].ssm_s); g->L[l2].ssm_s = 0; g->used -= ss; }
                if (g->L[l2].win_s) { g->cu.MemFree(g->L[l2].win_s); g->L[l2].win_s = 0; g->used -= ws; }
            }
            snprintf(g->full_why + strlen(g->full_why), sizeof g->full_why - strlen(g->full_why), "; no room for the step snapshot (per-token redo off)");
            return;
        }
    }
    g->snap_ok = 1;
}

int nemo_gpu_full_available(const nemo_gpu *g, int *per_step, char *why, size_t wl) {
    if (per_step) *per_step = g->full_ok && g->snap_ok;
    if (why && wl) snprintf(why, wl, "%s", g->full_why);
    return g->full_ok;
}
int nemo_gpu_set_full(nemo_gpu *g, int on) {
    if (on && !g->full_ok) return -1;
    if (!g->half_book) return 0; /* already the full kernel */
    g->exact = on ? 1 : 0;
    return 0;
}
int nemo_gpu_get_full(const nemo_gpu *g) { return g->exact || !g->half_book; }
int nemo_gpu_snapshot(nemo_gpu *g) {
    if (!g->snap_ok || !g->valid) return -1;
    for (int l = 0; l < g->layer_limit; l++) {
        glayer *L = &g->L[l];
        if (L->type != 'M') continue;
        if (ck(g, g->cu.MemcpyDtoD(L->ssm_s, L->ssm, (size_t)NHEAD * HDIM * NSTATE * 4), "snapshot")
            || ck(g, g->cu.MemcpyDtoD(L->win_s, L->win, (size_t)CONV * 16), "snapshot")) return -1;
    }
    g->snap_len = g->len;
    return 0;
}
/* Back to the snapshot: the recurrent state is copied back and the
 * position rewinds; the attention K/V rows at positions >= snap_len are
 * simply overwritten by the redo. */
int nemo_gpu_restore(nemo_gpu *g) {
    if (!g->snap_ok) return -1;
    for (int l = 0; l < g->layer_limit; l++) {
        glayer *L = &g->L[l];
        if (L->type != 'M') continue;
        if (ck(g, g->cu.MemcpyDtoD(L->ssm, L->ssm_s, (size_t)NHEAD * HDIM * NSTATE * 4), "restore")
            || ck(g, g->cu.MemcpyDtoD(L->win, L->win_s, (size_t)CONV * 16), "restore")) { g->valid = 0; return -1; }
    }
    g->len = g->snap_len;
    g->valid = 1;
    return 0;
}

static int open_device(nemo_gpu *g, char *why, size_t wl) {
    int n = 0;
    if (g->cu.Init(0) || g->cu.DeviceGetCount(&n) || n < 1 || g->cu.DeviceGet(&g->dev, 0)) {
        snprintf(why, wl, "no CUDA device"); return -1;
    }
    g->cu.DeviceGetName(g->info.device, sizeof g->info.device, g->dev);
    g->cu.DeviceGetAttribute(&g->info.cc_major, 75, g->dev);
    g->cu.DeviceGetAttribute(&g->info.cc_minor, 76, g->dev);
    size_t tot = 0; g->cu.DeviceTotalMem(&tot, g->dev); g->info.vram_total = tot;
    if (g->cu.PrimaryCtxRetain(&g->ctx, g->dev) || g->cu.CtxSetCurrent(g->ctx)) {
        snprintf(why, wl, "cannot create a CUDA context"); return -1;
    }
    g->ctx_ok = 1;
    size_t fr = 0, t2 = 0; g->cu.MemGetInfo(&fr, &t2); g->info.vram_free_at_open = fr;
    char *img = compile(g, g->info.cc_major, g->info.cc_minor, 1, why, wl);
    if (!img) img = compile(g, g->info.cc_major, g->info.cc_minor, 0, why, wl);
    if (!img) return -1;
    int rc = g->cu.ModuleLoadData(&g->mod, img);
    free(img);
    if (rc) { snprintf(why, wl, "cuModuleLoadData failed (%d)", rc); return -1; }
    for (int k = 0; k < K_N; k++)
        if (g->cu.ModuleGetFunction(&g->fn[k], g->mod, kname[k])) { snprintf(why, wl, "kernel %s missing", kname[k]); return -1; }
    /* 8 = CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES; 64 KiB for the staged book. */
    if (g->half_book >= 2 && (g->cu.FuncSetAttribute(g->fn[K_PQS], 8, 65536) || g->cu.FuncSetAttribute(g->fn[K_PQU], 8, 65536) || g->cu.FuncSetAttribute(g->fn[K_PQV], 8, 65536) || g->cu.FuncSetAttribute(g->fn[K_PQW], 8, 65536) || g->cu.FuncSetAttribute(g->fn[K_PQT], 8, 65536)
        || g->cu.FuncSetAttribute(g->fn[K_GT8], 8, 65536) || g->cu.FuncSetAttribute(g->fn[K_GT16], 8, 65536)
        || g->cu.FuncSetAttribute(g->fn[K_GR8T8], 8, 65536) || g->cu.FuncSetAttribute(g->fn[K_GR8T4], 8, 65536)
        || g->cu.FuncSetAttribute(g->fn[K_GR2T32], 8, 65536) || g->cu.FuncSetAttribute(g->fn[K_GT8K], 8, 65536)
        || g->cu.FuncSetAttribute(g->fn[K_PQT1K], 8, 65536))) {
        if (g->half_book >= 5) { snprintf(why, wl, "cannot enable 64 KiB shared memory for k_pq_gemv_w"); return -1; }
        g->half_book = 1;
    }
    /* the FULL fast path needs the same 64 KiB; without it FULL runs the exact kernels */
    if (g->cu.FuncSetAttribute(g->fn[K_PQFX], 8, 65536) || g->cu.FuncSetAttribute(g->fn[K_GF16], 8, 65536))
        g->gemv_x = K_PQX4;
    return 0;
}

int nemo_gpu_open(nemo_gpu **out, const char *ckpt_dir, const char *pack_path,
                  int layer_limit, int max_pos, char *why, size_t wl) {
    *out = NULL;
    nemo_gpu *g = (nemo_gpu *)calloc(1, sizeof *g);
    if (!g) { snprintf(why, wl, "out of memory"); return -1; }
    { const char *ex = getenv("NEMO_GPU_EXACT"); g->exact = ex && *ex == '1'; }
    { const char *hb = getenv("NEMO_GPU_BOOK"); g->half_book = g->exact ? 0 : (hb && !strcmp(hb, "fp32")) ? 0 : (hb && !strcmp(hb, "l1")) ? 1 : (hb && !strcmp(hb, "smem")) ? 2 : (hb && !strcmp(hb, "u")) ? 3 : (hb && !strcmp(hb, "v")) ? 4 : (hb && !strcmp(hb, "w")) ? 5 : 6; }
    kernel_choice(g);
    g->layer_limit = layer_limit > 0 && layer_limit <= NLAYERS ? layer_limit : NLAYERS;
    g->max_pos = max_pos > 0 ? max_pos : 4096;
    g->info.layers = g->layer_limit; g->info.max_pos = g->max_pos;
    nemo_checkpoint cp; int cp_ok = 0;
    pack k; memset(&k, 0, sizeof k);
    if (load_cuda(g, why, wl) || open_device(g, why, wl)) goto fail;
    if (nemo_checkpoint_open(&cp, ckpt_dir) != 0) { snprintf(why, wl, "cannot open checkpoint %s", ckpt_dir); goto fail; }
    cp_ok = 1;
    if (pack_open(&k, pack_path, why, wl)) goto fail;
    char nm[256];
    rpend pend[224]; int npend = 0;
#define NM(leaf) (snprintf(nm, sizeof nm, "backbone.layers.%d.%s", l, leaf), nm)
    for (int l = 0; l < g->layer_limit; l++) {
        glayer *L = &g->L[l];
        L->type = LAYER_PATTERN[l];
        if (up_param(g, &cp, NM("norm.weight"), H, &L->norm_w)) goto ferr;
        if (L->type == 'M') {
            if (up_pqr(g, &k, &cp, NM("mixer.in_proj.weight"), PROJ, H, &L->p0, &L->r0, pend, &npend)) goto ferr;
            if (up_pqr(g, &k, &cp, NM("mixer.out_proj.weight"), H, INNER, &L->p1, &L->r1, pend, &npend)) goto ferr;
            if (up_param(g, &cp, NM("mixer.conv1d.weight"), (uint64_t)CONV * 4, &L->conv_w)
                || up_param(g, &cp, NM("mixer.conv1d.bias"), CONV, &L->conv_b)
                || up_param(g, &cp, NM("mixer.A_log"), NHEAD, &L->a_log)
                || up_param(g, &cp, NM("mixer.D"), NHEAD, &L->d)
                || up_param(g, &cp, NM("mixer.dt_bias"), NHEAD, &L->dt_bias)
                || up_param(g, &cp, NM("mixer.norm.weight"), INNER, &L->gnorm_w)) goto ferr;
            if (dalloc(g, &L->ssm, (size_t)NHEAD * HDIM * NSTATE * 4) || dalloc(g, &L->win, (size_t)CONV * 16)) goto ferr;
        } else if (L->type == '-') {
            if (up_pqr(g, &k, &cp, NM("mixer.up_proj.weight"), MLP, H, &L->p0, &L->r0, pend, &npend)) goto ferr;
            if (up_pqr(g, &k, &cp, NM("mixer.down_proj.weight"), H, MLP, &L->p1, &L->r1, pend, &npend)) goto ferr;
        } else {
            if (up_pqr(g, &k, &cp, NM("mixer.q_proj.weight"), NQ * AD, H, &L->p0, &L->r0, pend, &npend)
                || up_pqr(g, &k, &cp, NM("mixer.k_proj.weight"), NKV * AD, H, &L->p1, &L->r1, pend, &npend)
                || up_pqr(g, &k, &cp, NM("mixer.v_proj.weight"), NKV * AD, H, &L->p2, &L->r2, pend, &npend)
                || up_pqr(g, &k, &cp, NM("mixer.o_proj.weight"), H, NQ * AD, &L->p3, &L->r3, pend, &npend)) goto ferr;
            size_t kv = (size_t)g->max_pos * NKV * AD * 2;
            if (dalloc(g, &L->kc, kv) || dalloc(g, &L->vc, kv)) goto ferr;
        }
    }
#undef NM
    if (up_param(g, &cp, "backbone.norm_f.weight", H, &g->norm_f)) goto ferr;
    if (dalloc(g, &g->x, H * 4) || dalloc(g, &g->h, H * 4) || dalloc(g, &g->proj, PROJ * 4)
        || dalloc(g, &g->v, CONV * 4) || dalloc(g, &g->y, INNER * 4) || dalloc(g, &g->inner, INNER * 4)
        || dalloc(g, &g->mid, MLP * 4) || dalloc(g, &g->q, NQ * AD * 4) || dalloc(g, &g->k, NKV * AD * 4)
        || dalloc(g, &g->vv, NKV * AD * 4) || dalloc(g, &g->logits, (size_t)VOCAB * 4)
        || dalloc(g, &g->scores, (size_t)NQ * g->max_pos * 8) || dalloc(g, &g->xs, (MLP / 16 + 64) * 4)) goto ferr;
    {
        /* VRAM plan: measure free memory after the base pack and the
         * workspaces, then keep as many whole refine layers as still leave
         * room for the int8 lm_head, a 64 MiB margin, and (when we can)
         * the EXPAND fp32 books. NEMO_GPU_REFINE=auto|off|all|11,10,8 */
        size_t fr = 0, tt = 0; g->cu.MemGetInfo(&fr, &tt);
        g->info.vram_free_after_pack = fr;
        const char *force = getenv("NEMO_GPU_HEAD");
        const char *rspec = getenv("NEMO_GPU_REFINE");
        if (!rspec || !rspec[0]) rspec = "auto";
        g->head_host = force && !strcmp(force, "host");
        size_t head_need = g->head_host ? 0 : (size_t)VOCAB * H; /* int8 */
        size_t head_mgn = g->head_host ? (16u << 20) : (64u << 20);
        size_t exp_rsv = (88u << 20) + (48u << 20);
        if (getenv("NEMO_GPU_NO_EXPAND") && getenv("NEMO_GPU_NO_EXPAND")[0] == '1') exp_rsv = (16u << 20);
        size_t budget = 0, reserve = head_need + head_mgn + exp_rsv;
        if (!strcmp(rspec, "all")) budget = fr;           /* try every refine tensor */
        else if (fr > reserve) budget = fr - reserve;
        else if (fr > head_need + head_mgn + (16u << 20)) {
            /* not enough for EXPAND books as well; keep a small safety margin */
            budget = fr - head_need - head_mgn - (16u << 20);
            exp_rsv = (16u << 20);
        }
        if (up_refine_fit(g, &k, &cp, pend, npend, budget)) goto ferr;
        fprintf(stderr, "nemo_gpu: VRAM plan: %.0f MiB free after the base pack; head %s %.0f + %.0f margin; EXPAND reserve %.0f; refine budget %.0f MiB; kept %d tensors (%.1f MiB), skipped %d\n",
                fr / 1048576.0, g->head_host ? "host" : "int8", head_need / 1048576.0, head_mgn / 1048576.0,
                exp_rsv / 1048576.0, budget / 1048576.0, g->nref, g->ref_bytes / 1048576.0, g->info.refine_skipped);
    }
    if (g->nref && dalloc(g, &g->rt, PROJ * 4)) goto ferr;
    {
        /* lm_head: bf16, resident, uploaded in 64 MiB chunks; or int8 on
         * the host when NEMO_GPU_HEAD=host (the GEMV runs on the CPU). */
        const nemo_tensor_info *t = nemo_find(&cp, "lm_head.weight");
        if (!t || !t->is_bf16 || t->count != (uint64_t)VOCAB * H) { snprintf(g->err, sizeof g->err, "lm_head missing or not bf16"); goto ferr; }
        size_t bytes = (size_t)t->count * 2;
        size_t fr = 0, tt = 0; g->cu.MemGetInfo(&fr, &tt);
        const char *force = getenv("NEMO_GPU_HEAD");
        g->head_i8 = g->head_host || (force && !strcmp(force, "int8")) || fr < bytes + (96u << 20);
        if (force && !strcmp(force, "bf16")) { g->head_i8 = 0; g->head_host = 0; }
        size_t dbytes = g->head_i8 ? (size_t)t->count : bytes;
        if (!g->head_host && fr < dbytes + (64u << 20)) {
            snprintf(g->err, sizeof g->err, "VRAM: lm_head needs %.0f MiB, %.0f MiB free after the pack", dbytes / 1048576.0, fr / 1048576.0);
            goto ferr;
        }
        if (g->head_host) {
            g->head_q = (int8_t *)malloc(dbytes);
            g->head_sc = (float *)malloc(sizeof(float) * VOCAB);
            g->h_host = (float *)malloc(sizeof(float) * H);
            if (!g->head_q || !g->head_sc || !g->h_host) { snprintf(g->err, sizeof g->err, "lm_head host alloc failed"); goto ferr; }
        } else
        if (!g->head_host && dalloc(g, &g->lm_head, dbytes)) goto ferr;
        if (g->head_host) {
            int ht = 4;
            const char *hts = getenv("NEMO_GPU_HEAD_THREADS");
            if (hts && *hts) ht = atoi(hts);
            if (ht < 1) ht = 1;
            if (ht > 8) ht = 8;
            memset(&g->head_pool, 0, sizeof g->head_pool);
            if (nemo_pool_init(&g->head_pool, ht) == 0) g->head_pool_ok = 1;
        }
        float *scales = g->head_i8 ? (float *)malloc(sizeof(float) * VOCAB) : NULL;
        FILE *f = fopen(t->shard_path, "rb");
        size_t rows_per = 4096, chunk = rows_per * H * 2;
        uint8_t *buf = (uint8_t *)malloc(chunk);
        int8_t *qb = g->head_i8 ? (int8_t *)malloc(rows_per * H) : NULL;
        if (!f || !buf || (g->head_i8 && (!scales || !qb)) || seek64(f, t->data_off)) {
            if (f) { fclose(f); }
            free(buf); free(qb); free(scales);
            snprintf(g->err, sizeof g->err, "lm_head read failed"); goto ferr;
        }
        for (size_t r0 = 0; r0 < VOCAB; r0 += rows_per) {
            size_t n = chunk;
            if (fread(buf, 1, n, f) != n) { fclose(f); free(buf); free(qb); free(scales); snprintf(g->err, sizeof g->err, "lm_head read failed"); goto ferr; }
            int rcu;
            if (g->head_i8) {
                const uint16_t *w = (const uint16_t *)buf;
                for (size_t r = 0; r < rows_per; r++) {
                    float mx = 0.0f;
                    for (int c = 0; c < H; c++) { uint32_t u = (uint32_t)w[r * H + c] << 16; float v; memcpy(&v, &u, 4); if (fabsf(v) > mx) mx = fabsf(v); }
                    float s = mx > 0.0f ? mx / 127.0f : 1.0f;
                    scales[r0 + r] = s;
                    for (int c = 0; c < H; c++) { uint32_t u = (uint32_t)w[r * H + c] << 16; float v; memcpy(&v, &u, 4);
                        long q = lrintf(v / s); if (q > 127) q = 127; if (q < -127) q = -127; qb[r * H + c] = (int8_t)q; }
                }
                if (g->head_host) {
                    memcpy(g->head_q + r0 * H, qb, rows_per * H);
                    rcu = 0;
                } else {
                    rcu = ck(g, g->cu.MemcpyHtoD(g->lm_head + r0 * H, qb, rows_per * H), "lm_head upload");
                }
            } else {
                rcu = ck(g, g->cu.MemcpyHtoD(g->lm_head + r0 * H * 2, buf, n), "lm_head upload");
            }
            if (rcu) { fclose(f); free(buf); free(qb); free(scales); goto ferr; }
        }
        fclose(f); free(buf); free(qb);
        if (g->head_i8) {
            if (g->head_host) {
                memcpy(g->head_sc, scales, sizeof(float) * VOCAB);
                free(scales);
            } else {
                int rcs = up(g, &g->head_scale, scales, sizeof(float) * VOCAB);
                free(scales);
                if (rcs) goto ferr;
            }
        }
        bytes = dbytes + (g->head_i8 ? sizeof(float) * VOCAB : 0);
        g->info.head_bytes = bytes;
        g->info.bytes_per_token += bytes;
        if (g->head_host)
            fprintf(stderr, "nemo_gpu: lm_head on the host (int8 per-row, %d CPU lanes), %.0f MiB RAM\n",
                    g->head_pool_ok ? nemo_pool_threads(&g->head_pool) : 1, bytes / 1048576.0);
    }
    {
        const nemo_tensor_info *e = nemo_find(&cp, "backbone.embeddings.weight");
        if (!e || !e->is_bf16 || e->count != (uint64_t)VOCAB * H) { snprintf(g->err, sizeof g->err, "embeddings missing or not bf16"); goto ferr; }
        g->emb_f = fopen(e->shard_path, "rb");
        g->emb_off = e->data_off;
        if (!g->emb_f) { snprintf(g->err, sizeof g->err, "cannot open embedding shard"); goto ferr; }
    }
    expand_resources(g, &k);
    prefill_resources(g);
    g->info.vram_used_by_us = g->used;
    g->info.refine_tensors = g->nref;
    g->info.refine_bytes = g->ref_bytes;
    if (g->nref)
        fprintf(stderr, "nemo_gpu: expanded pack: %d refinement tensors, %.1f MiB on the device\n",
                g->nref, g->ref_bytes / 1048576.0);
    g->info.head_int8 = g->head_i8;
    g->info.head_host = g->head_host;
    g->info.full_ok = g->full_ok;
    g->info.snap_ok = g->snap_ok;
    g->info.exact = g->exact;
    g->info.half_book = g->half_book;
    pack_free(&k);
    nemo_checkpoint_close(&cp);
    g->valid = 1;
    nemo_gpu_reset(g);
    if (!g->valid) { snprintf(why, wl, "initial reset failed"); nemo_gpu_close(g); return -1; }
    *out = g;
    return 0;
ferr:
    snprintf(why, wl, "%s", g->err[0] ? g->err : "upload failed");
fail:
    pack_free(&k);
    if (cp_ok) nemo_checkpoint_close(&cp);
    nemo_gpu_close(g);
    return -1;
}

/* ------------------------------------------------------------------ */
/* Calibration probe (see nemo_gpu.h).                                 */

/* fmaf, correctly rounded, without trusting the C library: the product
 * is exact in double, TwoSum gives the exact remainder, and the one
 * double-rounding case (a float midpoint) is settled by its sign. */
static float fma_ref(float a, float b, float c) {
    double p = (double)a * (double)b, s = p + (double)c;
    double bb = s - p, err = (p - (s - bb)) + ((double)c - bb);
    float f = (float)s;
    if (err != 0.0 && (double)f != s) {
        float n = nextafterf(f, (s > (double)f) ? INFINITY : -INFINITY);
        if (s == ((double)f + (double)n) * 0.5) {
            float hi = f > n ? f : n, lo = f > n ? n : f;
            f = err > 0.0 ? hi : lo;
        }
    }
    return f;
}
static float raw_lo_ref(uint32_t v) { uint32_t u = ((v << 13) & 0x0FFFE000u) | ((v << 16) & 0x80000000u); float f; memcpy(&f, &u, 4); return f; }
static float raw_hi_ref(uint32_t v) { uint32_t u = ((v >> 3) & 0x0FFFE000u) | (v & 0x80000000u); float f; memcpy(&f, &u, 4); return f; }
static float bfr_ref(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) u = 0x7fc00000u;
    else { u += 0x7fffu + ((u >> 16) & 1u); u &= 0xffff0000u; }
    memcpy(&f, &u, 4); return f;
}
static float wsum_ref(float *v) {
    for (int o = 16; o > 0; o >>= 1) { float t[32]; for (int l = 0; l < 32; l++) t[l] = v[l] + v[l ^ o]; memcpy(v, t, sizeof t); }
    return v[0];
}
/* C emulation of k_pq_gemv_t (DTC) for one row. */
static float dtc_ref(const uint8_t *idx, const uint16_t *mn, const uint16_t *sc, const uint16_t *bookh,
                     const float *x, int r, int cols) {
    const float s112 = 5.192296858534828e+33f;
    uint64_t base = (uint64_t)r * cols, bf = base >> 8; int off = (int)(base & 255);
    int nblk = (int)(((base + cols - 1) >> 8) - bf) + 1;
    float v[32];
    for (int lane = 0; lane < 32; lane++) {
        float a = 0.0f;
        for (int p = 0; p < 2; p++)
            for (int blk = 0; blk < nblk; blk++) {
                int c = 128 * p + 4 * lane - off + 256 * blk;
                if (c < 0 || c >= cols) continue;
                uint32_t id = idx[(bf + blk) * 64 + 32 * p + lane];
                float m = raw_lo_ref(mn[bf + blk]) * s112, cc = raw_lo_ref(sc[bf + blk]) * s112 * s112;
                const uint16_t *q = bookh + ((size_t)(32 * p + lane) * 256 + id) * 4;
                uint32_t qx = q[0] | ((uint32_t)q[1] << 16), qy = q[2] | ((uint32_t)q[3] << 16);
                a = fma_ref(fma_ref(cc, raw_lo_ref(qx), m), x[c], a);
                a = fma_ref(fma_ref(cc, raw_hi_ref(qx), m), x[c + 1], a);
                a = fma_ref(fma_ref(cc, raw_lo_ref(qy), m), x[c + 2], a);
                a = fma_ref(fma_ref(cc, raw_hi_ref(qy), m), x[c + 3], a);
            }
        v[lane] = a;
    }
    return wsum_ref(v);
}
/* C emulation of k_pq_gemv (FULL) for one row. */
static float full_ref(const uint8_t *idx, const uint16_t *mn, const uint16_t *sc, const float *book,
                      const float *x, int r, int cols) {
    uint64_t base = (uint64_t)r * cols; int nsv = cols >> 2;
    float v[32];
    for (int lane = 0; lane < 32; lane++) {
        float acc = 0.0f;
        for (int s = lane; s < nsv; s += 32) {
            uint64_t e = base + 4ull * s, b = e >> 8; int sub = (int)((e & 255) >> 2);
            int ix = idx[b * 64 + sub];
            const float *q = book + ((size_t)sub * 256 + ix) * 4;
            float m = f16_to_f32(mn[b]), c = f16_to_f32(sc[b]);
            for (int i = 0; i < 4; i++) acc = acc + bfr_ref(m + c * q[i]) * x[4 * s + i];
        }
        v[lane] = acc;
    }
    return wsum_ref(v);
}
/* C emulation of k_pq_gemv_f (FULL fast path) for row r of a P-tiled shape. */
static float full_f_ref(const uint8_t *idx, const uint16_t *mn, const uint16_t *sc, const float *book,
                        const float *x, int r, int cols) {
    uint64_t base = (uint64_t)r * cols, bf = base >> 8; int off = (int)(base & 255);
    int nblk = (int)(((base + cols - 1) >> 8) - bf) + 1;
    float v[32];
    for (int lane = 0; lane < 32; lane++) {
        int sl = lane & 15, par = lane >> 4; float a = 0.0f;
        for (int p = 0; p < 4; p++)
            for (int blk = par; blk < nblk; blk += 2) {
                int c = 64 * p + 4 * sl - off + 256 * blk;
                if (c < 0 || c >= cols) continue;
                int sub = 16 * p + sl; int ix = idx[(bf + blk) * 64 + sub];
                const float *q = book + ((size_t)sub * 256 + ix) * 4;
                float m = f16_to_f32(mn[bf + blk]), cc = f16_to_f32(sc[bf + blk]);
                for (int i = 0; i < 4; i++) { volatile float pr = cc * q[i]; volatile float s = m + pr; a = fma_ref(bfr_ref(s), x[c + i], a); }
            }
        v[lane] = a;
    }
    return wsum_ref(v);
}

static uint64_t lcg_state = 0x2545F4914F6CDD1Dull;
static uint32_t lcg(void) { lcg_state = lcg_state * 6364136223846793005ull + 1442695040888963407ull; return (uint32_t)(lcg_state >> 33); }
static float frand(void) { return (float)((lcg() & 0xffffff) / 16777216.0) * 2.0f - 1.0f; }

typedef struct { gpq w; uint8_t *idx; uint16_t *mn, *sc, *bookh; float *book; } synth;
static int synth_make(nemo_gpu *g, synth *t, int rows, int cols, int host_ref) {
    memset(t, 0, sizeof *t);
    uint64_t n = (uint64_t)rows * cols, nb = n / 256;
    t->idx = (uint8_t *)malloc(nb * 64); t->mn = (uint16_t *)malloc(nb * 2); t->sc = (uint16_t *)malloc(nb * 2);
    t->book = (float *)malloc(256 * 256 * 4); t->bookh = (uint16_t *)malloc(256 * 256 * 2);
    if (!t->idx || !t->mn || !t->sc || !t->book || !t->bookh) return -1;
    for (uint64_t i = 0; i < nb * 64; i++) t->idx[i] = (uint8_t)lcg();
    for (uint64_t i = 0; i < nb; i++) { t->mn[i] = f32_to_f16(frand() * 0.02f); t->sc[i] = f32_to_f16(0.005f + 0.05f * (frand() + 1.0f)); }
    for (int i = 0; i < 256 * 256; i++) { t->book[i] = frand() * 2.5f; t->bookh[i] = f32_to_f16(t->book[i]); }
    (void)host_ref;
    t->w.rows = rows; t->w.cols = cols;
    if (up(g, &t->w.idx, t->idx, nb * 64) || up(g, &t->w.mn, t->mn, nb * 2) || up(g, &t->w.sc, t->sc, nb * 2)
        || up(g, &t->w.book, t->book, 256 * 256 * 4) || up(g, &t->w.bookh, t->bookh, 256 * 256 * 2)
        || up_mc(g, &t->w.mc, (const uint8_t *)t->mn, (const uint8_t *)t->sc, nb)) return -1;
    return 0;
}
/* The self-test makes exactly this many synthetic tensors: two
 * correctness shapes and one throughput shape. synth_slot hands out
 * zeroed slots and refuses past the end, so an added shape fails the
 * self-test instead of writing past the stack array; nt counts every
 * slot handed out, so cleanup frees each one exactly once, partial
 * uploads included (synth_free skips anything never allocated). */
#define SELFTEST_SYNTHS 3
static synth *synth_slot(synth *t, int *nt, int cap) {
    if (*nt < 0 || *nt >= cap) return NULL;
    synth *s = &t[(*nt)++];
    memset(s, 0, sizeof *s);
    return s;
}
static void synth_free(nemo_gpu *g, synth *t) {
    CUdeviceptr d[6] = {t->w.idx, t->w.mn, t->w.sc, t->w.mc, t->w.book, t->w.bookh};
    for (int i = 0; i < 6; i++) if (d[i]) g->cu.MemFree(d[i]);
    free(t->idx); free(t->mn); free(t->sc); free(t->book); free(t->bookh);
}

int nemo_gpu_identify(char *device, size_t dl, int *driver, int *nvrtc_major, int *nvrtc_minor) {
    nemo_gpu *g = (nemo_gpu *)calloc(1, sizeof *g);
    char why[256];
    int rc = -1, n = 0;
    if (!g) return -1;
    *driver = 0; *nvrtc_major = *nvrtc_minor = 0; if (dl) device[0] = 0;
    if (load_cuda(g, why, sizeof why) == 0 && g->cu.Init(0) == 0 && g->cu.DeviceGetCount(&n) == 0 && n > 0
        && g->cu.DeviceGet(&g->dev, 0) == 0) {
        g->cu.DeviceGetName(device, (int)dl, g->dev);
        g->cu.DriverGetVersion(driver);
        if (g->rtc.Version) g->rtc.Version(nvrtc_major, nvrtc_minor);
        rc = 0;
    }
    dl_close(g->lrtc); dl_close(g->lcu); free(g);
    return rc;
}

int nemo_gpu_selftest(nemo_gpu_probe *r, char *why, size_t wl) {
    memset(r, 0, sizeof *r);
    nemo_gpu *g = (nemo_gpu *)calloc(1, sizeof *g);
    if (!g) { snprintf(why, wl, "out of memory"); return -1; }
    g->half_book = 6;
    kernel_choice(g);   /* the self-test checks the kernels the engine will run */
    int rc = -1;
    synth t[SELFTEST_SYNTHS]; int nt = 0;
    memset(t, 0, sizeof t);
    float *x = NULL, *y = NULL;
    CUdeviceptr dx = 0, dy = 0, dout = 0;
    if (load_cuda(g, why, wl)) goto out;
    g->cu.DriverGetVersion(&r->driver);
    if (g->rtc.Version) g->rtc.Version(&r->nvrtc_major, &r->nvrtc_minor);
    if (open_device(g, why, wl)) goto out;
    snprintf(r->device, sizeof r->device, "%s", g->info.device);
    r->cc_major = g->info.cc_major; r->cc_minor = g->info.cc_minor;
    r->vram_total = g->info.vram_total; r->vram_free = g->info.vram_free_at_open;
    x = (float *)malloc(4480 * 4); y = (float *)malloc(22656 * 4);
    if (!x || !y) { snprintf(why, wl, "out of memory"); goto out; }
    for (int i = 0; i < 4480; i++) x[i] = frand();
    if (up(g, &dx, x, 4480 * 4) || dalloc(g, &dy, 22656 * 4) || dalloc(g, &dout, 16)) { snprintf(why, wl, "%s", g->err); goto out; }
    /* correctness: two shapes (aligned rows; rows starting mid-block) */
    {
        int shp[2][2] = {{64, 512}, {32, 4480}};
        r->dtc_exact = r->full_exact = 1;
        for (int si = 0; si < 2; si++) {
            synth *T = synth_slot(t, &nt, SELFTEST_SYNTHS);
            if (!T) { snprintf(why, wl, "selftest: no synthetic tensor slot left"); goto out; }
            if (synth_make(g, T, shp[si][0], shp[si][1], 1)) { snprintf(why, wl, "synthetic upload failed: %s", g->err); goto out; }
            for (int mode = 0; mode < 2; mode++) {
                g->exact = mode;
                if (pq_gemv(g, &T->w, dx, dy) || ck(g, g->cu.CtxSynchronize(), "selftest")
                    || ck(g, g->cu.MemcpyDtoH(y, dy, (size_t)T->w.rows * 4), "selftest copy")) { snprintf(why, wl, "%s", g->err); goto out; }
                for (int row = 0; row < T->w.rows; row++) {
                    float ref = mode ? (full_tile(g, T->w.rows, T->w.cols) ? full_f_ref(T->idx, T->mn, T->sc, T->book, x, row, T->w.cols)
                                                                            : full_ref(T->idx, T->mn, T->sc, T->book, x, row, T->w.cols))
                                     : dtc_ref(T->idx, T->mn, T->sc, T->bookh, x, row, T->w.cols);
                    uint32_t a, b; memcpy(&a, &ref, 4); memcpy(&b, &y[row], 4);
                    if (a != b) { if (mode) { r->full_exact = 0; r->full_mismatch++; } else { r->dtc_exact = 0; r->dtc_mismatch++; } }
                }
            }
        }
        g->exact = 0;
    }
    /* throughput: one in_proj-sized PQ GEMV (22656 x 4480) and a raw read */
    {
        synth *T = synth_slot(t, &nt, SELFTEST_SYNTHS);
        if (!T) { snprintf(why, wl, "selftest: no synthetic tensor slot left"); goto out; }
        if (synth_make(g, T, 22656, 4480, 0)) { snprintf(why, wl, "bench upload failed: %s", g->err); goto out; }
        double wts = 22656.0 * 4480.0;
        for (int mode = 0; mode < 2; mode++) {
            g->exact = mode;
            int reps = mode ? 3 : 10;
            if (pq_gemv(g, &T->w, dx, dy) || ck(g, g->cu.CtxSynchronize(), "bench")) { snprintf(why, wl, "%s", g->err); goto out; }
            double t0 = now_s();
            for (int i = 0; i < reps; i++) if (pq_gemv(g, &T->w, dx, dy)) { snprintf(why, wl, "%s", g->err); goto out; }
            if (ck(g, g->cu.CtxSynchronize(), "bench")) { snprintf(why, wl, "%s", g->err); goto out; }
            double dt = (now_s() - t0) / reps;
            if (!(dt > 0.0) || !isfinite(dt)) { snprintf(why, wl, "selftest timer produced a non-positive interval"); goto out; }
            if (mode) r->full_gwps = wts / dt / 1e9; else r->dtc_gwps = wts / dt / 1e9;
        }
        g->exact = 0;
        unsigned long long n16 = (unsigned long long)(wts / 4.0) / 16ull;
        void *a[] = {&T->w.idx, &n16, &dout};
        if (launch(g, K_BW, 160, 1, 256, a) || ck(g, g->cu.CtxSynchronize(), "bw")) { snprintf(why, wl, "%s", g->err); goto out; }
        double t0 = now_s();
        for (int i = 0; i < 10; i++) if (launch(g, K_BW, 160, 1, 256, a)) { snprintf(why, wl, "%s", g->err); goto out; }
        if (ck(g, g->cu.CtxSynchronize(), "bw")) { snprintf(why, wl, "%s", g->err); goto out; }
        {
            double bdt = now_s() - t0;
            if (!(bdt > 0.0) || !isfinite(bdt)) { snprintf(why, wl, "selftest timer produced a non-positive interval"); goto out; }
            r->read_gbs = n16 * 16.0 * 10 / bdt / 1e9;
        }
    }
    {
        /* weights per token in the 9B stack's projections */
        double W = 27.0 * (22656.0 * 4480 + 4480.0 * 10240) + 25.0 * (2.0 * 15680 * 4480)
                 + 4.0 * (2.0 * 5120 * 4480 + 2.0 * 1024 * 4480);
        double head_s = r->read_gbs > 0 ? 0.56e9 / (r->read_gbs * 1e9) : 0.01;
        r->est_tps = 1.0 / (W / (r->dtc_gwps * 1e9) + head_s + 0.006);
    }
    r->pass = r->dtc_exact && r->full_exact;
    if (!r->pass)
        snprintf(why, wl, "kernel output differs from the C emulation (dtc %ld, full %ld rows)", r->dtc_mismatch, r->full_mismatch);
    rc = r->pass ? 0 : -1;
out:
    for (int i = 0; i < nt; i++) synth_free(g, &t[i]);
    if (dx) g->cu.MemFree(dx);
    if (dy) g->cu.MemFree(dy);
    if (dout) g->cu.MemFree(dout);
    free(x); free(y);
    nemo_gpu_close(g);
    return rc;
}
