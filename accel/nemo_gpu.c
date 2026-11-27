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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
       K_ASCORE, K_AOUT, K_I8, K_XSUM, K_PQF, K_PQH, K_PQS, K_PQU, K_PQV, K_PQW, K_PQT, K_BW, K_N };
static const char *const kname[K_N] = {
    "k_pq_gemv", "k_bf16_gemv", "k_rmsnorm", "k_add", "k_relu2", "k_conv",
    "k_ssm", "k_gated_norm", "k_kv_push", "k_attn_scores", "k_attn_out", "k_i8_gemv", "k_xsum16", "k_pq_gemv_fast", "k_pq_gemv_h", "k_pq_gemv_s", "k_pq_gemv_u", "k_pq_gemv_v", "k_pq_gemv_w", "k_pq_gemv_t", "k_bw" };

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

typedef struct { CUdeviceptr idx, mn, sc, book, bookh; int rows, cols, pidx; } gpq;

typedef struct {
    char type;
    CUdeviceptr norm_w;
    gpq p0, p1, p2, p3;                     /* M: in,out  -: up,down  *: q,k,v,o */
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
    int exact, half_book;
    int full_ok, snap_ok, snap_len;          /* EXPAND resources (fp32 books, state snapshot) */
    char full_why[160];
    FILE *emb_f;
    uint64_t emb_off;
    uint64_t used;
    void (*hook)(void *, int, const float *);
    void *hook_user;
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

static char *compile(nemo_gpu *g, int maj, int min, int real, char *why, size_t wl) {
    nvrtcProgram p = NULL;
    if (g->rtc.Create(&p, nemo_gpu_cuda_src, "nemo_gpu.cu", 0, NULL, NULL) != 0) {
        snprintf(why, wl, "nvrtcCreateProgram failed"); return NULL;
    }
    char arch[64];
    snprintf(arch, sizeof arch, "--gpu-architecture=%s_%d%d", real ? "sm" : "compute", maj, min);
    const char *opts[] = {arch, "--fmad=false", "--std=c++11", "--ftz=false", "--prec-div=true", "--prec-sqrt=true"};
    if (g->rtc.Compile(p, 6, opts) != 0) {
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
    uint64_t *b0, *nb, *book_off;
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

static void pack_free(pack *k) {
    if (k->s) fclose(k->s);
    free(k->name); free(k->b0); free(k->nb); free(k->book_off); free(k->book);
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
    if (!k->name || !k->b0 || !k->nb || !k->book_off) { fclose(m); free(seg); return -1; }
    uint64_t cur = 0;
    for (int i = 0; i < nt; i++) {
        unsigned long long b0, nb, sd, bd; unsigned blk; char kind[16];
        if (!fgets(line, sizeof line, m)
            || sscanf(line, "tensor %255s %llu %llu %u %llx %llx %15s", k->name[i], &b0, &nb, &blk, &sd, &bd, kind) != 7
            || b0 != cur || nb != seg[i] || blk != k->block) {
            fclose(m); free(seg); snprintf(why, wl, "manifest line %d disagrees with the stream", i); return -1;
        }
        k->b0[i] = b0; k->nb[i] = nb; k->book_off[i] = (uint64_t)i * k->entries * k->block;
        cur += nb;
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
    return 0;
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

static int up_pq(nemo_gpu *g, pack *k, const nemo_checkpoint *cp, const char *name, int rows, int cols, gpq *o) {
    int i = pack_find(k, name);
    const nemo_tensor_info *t = nemo_find(cp, name);
    if (i < 0 || !t) { snprintf(g->err, sizeof g->err, "projection %s not in pack/checkpoint", name); return -1; }
    uint64_t cnt = (uint64_t)rows * cols;
    if (t->count != cnt || k->nb[i] * k->block != cnt || cols % 16) {
        snprintf(g->err, sizeof g->err, "projection %s geometry mismatch", name); return -1;
    }
    uint64_t nb = k->nb[i];
    size_t isz = (size_t)(nb * k->idxw), fsz = (size_t)(nb * 2);
    uint8_t *buf = (uint8_t *)malloc(isz);
    if (!buf) return -1;
    int rc = -1;
    if (seek64(k->s, k->header + k->b0[i] * k->idxw) || fread(buf, 1, isz, k->s) != isz) goto done;
    if (up(g, &o->idx, buf, isz)) goto done;
    if (seek64(k->s, k->header + k->nblocks * k->idxw + 2 * k->b0[i]) || fread(buf, 1, fsz, k->s) != fsz) goto done;
    if (up_scale(g, &o->mn, buf, nb)) goto done;
    if (seek64(k->s, k->header + k->nblocks * (k->idxw + 2) + 2 * k->b0[i]) || fread(buf, 1, fsz, k->s) != fsz) goto done;
    if (up_scale(g, &o->sc, buf, nb)) goto done;
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
        g->info.pack_bytes += isz + 2 * fsz + bb;
        g->info.bytes_per_token += isz + 2 * fsz + bb;
    }
    rc = 0;
done:
    if (rc && !g->err[0]) snprintf(g->err, sizeof g->err, "reading %s from the pack failed", name);
    free(buf);
    return rc;
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
static int g_shape[16][2]; static double g_shape_t[16]; static long g_shape_n[16];
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
        if (k >= K_PQF) { int ai = k >= K_PQV ? 6 : 7; int rr = *(int *)args[ai], cc = *(int *)args[ai + 1];
            for (int q = 0; q < 16; q++) { if (!g_shape[q][0] || (g_shape[q][0] == rr && g_shape[q][1] == cc)) { g_shape[q][0] = rr; g_shape[q][1] = cc; g_shape_t[q] += dt; g_shape_n[q]++; break; } } } }
    return rc;
}
static int pq_gemv(nemo_gpu *g, const gpq *w, CUdeviceptr x, CUdeviceptr y) {
    if (g->exact && w->book) {
        void *a[] = {(void *)&w->idx, (void *)&w->mn, (void *)&w->sc, (void *)&w->book, &x, &y, (void *)&w->rows, (void *)&w->cols};
        return launch(g, K_PQ, (unsigned)((w->rows + 7) / 8), 1, 256, a);
    }
    if (g->half_book == 6) {
        int P = 1;
        while (P <= 8 && ((uint64_t)P * (uint64_t)w->cols) % 256u) P *= 2;
        if (P <= 8 && w->rows % (4 * P) == 0) {
            int nwarps = w->rows / 4;
            void *a[] = {(void *)&w->idx, (void *)&w->mn, (void *)&w->sc, (void *)&w->bookh, &x, &y, (void *)&w->rows, (void *)&w->cols, &P};
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

static int run_layer(nemo_gpu *g, int l) {
    glayer *L = &g->L[l];
    if (rmsnorm(g, g->x, L->norm_w, g->h)) return -1;
    if (L->type == 'M') {
        if (pq_gemv(g, &L->p0, g->h, g->proj)) return -1;
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
        if (pq_gemv(g, &L->p1, g->inner, g->y)) return -1;
    } else if (L->type == '-') {
        if (pq_gemv(g, &L->p0, g->h, g->mid)) return -1;
        int n = MLP;
        void *ar[] = {&g->mid, &n};
        if (launch(g, K_RELU2, (MLP + 255) / 256, 1, 256, ar)) return -1;
        if (pq_gemv(g, &L->p1, g->mid, g->y)) return -1;
    } else {
        if (pq_gemv(g, &L->p0, g->h, g->q) || pq_gemv(g, &L->p1, g->h, g->k) || pq_gemv(g, &L->p2, g->h, g->vv)) return -1;
        int pos = g->len, row = NKV * AD;
        void *ak[] = {&L->kc, &L->vc, &g->k, &g->vv, &pos, &row};
        if (launch(g, K_KVPUSH, (row + 255) / 256, 1, 256, ak)) return -1;
        int len = g->len + 1, nq = NQ, nkv = NKV, d = AD, mp = g->max_pos;
        void *a1[] = {&g->q, &L->kc, &g->scores, &len, &nq, &nkv, &d, &mp};
        if (launch(g, K_ASCORE, (unsigned)((len + 127) / 128), NQ, 128, a1)) return -1;
        void *a2[] = {&g->scores, &L->vc, &g->inner, &len, &nq, &nkv, &d, &mp};
        if (launch(g, K_AOUT, NQ, 1, AD, a2)) return -1;
        if (pq_gemv(g, &L->p3, g->inner, g->y)) return -1;
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
        for (int q = 0; q < 16 && g_shape[q][0]; q++)
            fprintf(stderr, "profile pq %5dx%-5d %6ld launches %9.1f ms total %8.1f us each %6.1f GB/s idx\n", g_shape[q][0], g_shape[q][1], g_shape_n[q], g_shape_t[q] * 1e3, g_shape_t[q] * 1e6 / g_shape_n[q], (double)g_shape[q][0] * g_shape[q][1] / 4.0 * g_shape_n[q] / g_shape_t[q] / 1e9);
    }
    if (g->ctx_ok) {
        g->cu.CtxSetCurrent(g->ctx);
        g->cu.CtxSynchronize();
        if (g->mod) g->cu.ModuleUnload(g->mod);
        g->cu.PrimaryCtxRelease(g->dev); /* frees every allocation of the primary context */
    }
    if (g->emb_f) fclose(g->emb_f);
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
            gpq *w[4] = {&g->L[l].p0, &g->L[l].p1, &g->L[l].p2, &g->L[l].p3};
            for (int j = 0; j < 4; j++) {
                if (!w[j]->rows || w[j]->book) continue;
                size_t fr = 0, tt = 0; g->cu.MemGetInfo(&fr, &tt);
                if (fr < nf + (48u << 20) || up(g, &w[j]->book, k->book + k->book_off[w[j]->pidx], nf)) {
                    snprintf(g->full_why, sizeof g->full_why, "VRAM: no room for the fp32 books (%.0f MiB free)", fr / 1048576.0);
                    for (int l2 = 0; l2 <= l; l2++) {
                        gpq *v[4] = {&g->L[l2].p0, &g->L[l2].p1, &g->L[l2].p2, &g->L[l2].p3};
                        for (int j2 = 0; j2 < 4; j2++) if (v[j2]->book) { g->cu.MemFree(v[j2]->book); v[j2]->book = 0; g->used -= nf; }
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
    if (g->half_book >= 2 && (g->cu.FuncSetAttribute(g->fn[K_PQS], 8, 65536) || g->cu.FuncSetAttribute(g->fn[K_PQU], 8, 65536) || g->cu.FuncSetAttribute(g->fn[K_PQV], 8, 65536) || g->cu.FuncSetAttribute(g->fn[K_PQW], 8, 65536) || g->cu.FuncSetAttribute(g->fn[K_PQT], 8, 65536))) {
        if (g->half_book >= 5) { snprintf(why, wl, "cannot enable 64 KiB shared memory for k_pq_gemv_w"); return -1; }
        g->half_book = 1;
    }
    return 0;
}

int nemo_gpu_open(nemo_gpu **out, const char *ckpt_dir, const char *pack_path,
                  int layer_limit, int max_pos, char *why, size_t wl) {
    *out = NULL;
    nemo_gpu *g = (nemo_gpu *)calloc(1, sizeof *g);
    if (!g) { snprintf(why, wl, "out of memory"); return -1; }
    { const char *ex = getenv("NEMO_GPU_EXACT"); g->exact = ex && *ex == '1'; }
    { const char *hb = getenv("NEMO_GPU_BOOK"); g->half_book = g->exact ? 0 : (hb && !strcmp(hb, "fp32")) ? 0 : (hb && !strcmp(hb, "l1")) ? 1 : (hb && !strcmp(hb, "smem")) ? 2 : (hb && !strcmp(hb, "u")) ? 3 : (hb && !strcmp(hb, "v")) ? 4 : (hb && !strcmp(hb, "w")) ? 5 : 6; }
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
#define NM(leaf) (snprintf(nm, sizeof nm, "backbone.layers.%d.%s", l, leaf), nm)
    for (int l = 0; l < g->layer_limit; l++) {
        glayer *L = &g->L[l];
        L->type = LAYER_PATTERN[l];
        if (up_param(g, &cp, NM("norm.weight"), H, &L->norm_w)) goto ferr;
        if (L->type == 'M') {
            if (up_pq(g, &k, &cp, NM("mixer.in_proj.weight"), PROJ, H, &L->p0)) goto ferr;
            if (up_pq(g, &k, &cp, NM("mixer.out_proj.weight"), H, INNER, &L->p1)) goto ferr;
            if (up_param(g, &cp, NM("mixer.conv1d.weight"), (uint64_t)CONV * 4, &L->conv_w)
                || up_param(g, &cp, NM("mixer.conv1d.bias"), CONV, &L->conv_b)
                || up_param(g, &cp, NM("mixer.A_log"), NHEAD, &L->a_log)
                || up_param(g, &cp, NM("mixer.D"), NHEAD, &L->d)
                || up_param(g, &cp, NM("mixer.dt_bias"), NHEAD, &L->dt_bias)
                || up_param(g, &cp, NM("mixer.norm.weight"), INNER, &L->gnorm_w)) goto ferr;
            if (dalloc(g, &L->ssm, (size_t)NHEAD * HDIM * NSTATE * 4) || dalloc(g, &L->win, (size_t)CONV * 16)) goto ferr;
        } else if (L->type == '-') {
            if (up_pq(g, &k, &cp, NM("mixer.up_proj.weight"), MLP, H, &L->p0)) goto ferr;
            if (up_pq(g, &k, &cp, NM("mixer.down_proj.weight"), H, MLP, &L->p1)) goto ferr;
        } else {
            if (up_pq(g, &k, &cp, NM("mixer.q_proj.weight"), NQ * AD, H, &L->p0)
                || up_pq(g, &k, &cp, NM("mixer.k_proj.weight"), NKV * AD, H, &L->p1)
                || up_pq(g, &k, &cp, NM("mixer.v_proj.weight"), NKV * AD, H, &L->p2)
                || up_pq(g, &k, &cp, NM("mixer.o_proj.weight"), H, NQ * AD, &L->p3)) goto ferr;
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
        /* lm_head: bf16, resident, uploaded in 64 MiB chunks */
        const nemo_tensor_info *t = nemo_find(&cp, "lm_head.weight");
        if (!t || !t->is_bf16 || t->count != (uint64_t)VOCAB * H) { snprintf(g->err, sizeof g->err, "lm_head missing or not bf16"); goto ferr; }
        size_t bytes = (size_t)t->count * 2;
        size_t fr = 0, tt = 0; g->cu.MemGetInfo(&fr, &tt);
        const char *force = getenv("NEMO_GPU_HEAD");
        g->head_i8 = (force && !strcmp(force, "int8")) || fr < bytes + (96u << 20);
        if (force && !strcmp(force, "bf16")) g->head_i8 = 0;
        size_t dbytes = g->head_i8 ? (size_t)t->count : bytes;
        if (fr < dbytes + (64u << 20)) {
            snprintf(g->err, sizeof g->err, "VRAM: lm_head needs %.0f MiB, %.0f MiB free after the pack", dbytes / 1048576.0, fr / 1048576.0);
            goto ferr;
        }
        if (dalloc(g, &g->lm_head, dbytes)) goto ferr;
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
                rcu = ck(g, g->cu.MemcpyHtoD(g->lm_head + r0 * H, qb, rows_per * H), "lm_head upload");
            } else {
                rcu = ck(g, g->cu.MemcpyHtoD(g->lm_head + r0 * H * 2, buf, n), "lm_head upload");
            }
            if (rcu) { fclose(f); free(buf); free(qb); free(scales); goto ferr; }
        }
        fclose(f); free(buf); free(qb);
        if (g->head_i8) {
            int rcs = up(g, &g->head_scale, scales, sizeof(float) * VOCAB);
            free(scales);
            if (rcs) goto ferr;
        }
        bytes = dbytes + (g->head_i8 ? sizeof(float) * VOCAB : 0);
        g->info.head_bytes = bytes;
        g->info.bytes_per_token += bytes;
    }
    {
        const nemo_tensor_info *e = nemo_find(&cp, "backbone.embeddings.weight");
        if (!e || !e->is_bf16 || e->count != (uint64_t)VOCAB * H) { snprintf(g->err, sizeof g->err, "embeddings missing or not bf16"); goto ferr; }
        g->emb_f = fopen(e->shard_path, "rb");
        g->emb_off = e->data_off;
        if (!g->emb_f) { snprintf(g->err, sizeof g->err, "cannot open embedding shard"); goto ferr; }
    }
    expand_resources(g, &k);
    g->info.vram_used_by_us = g->used;
    g->info.head_int8 = g->head_i8;
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
        || up(g, &t->w.book, t->book, 256 * 256 * 4) || up(g, &t->w.bookh, t->bookh, 256 * 256 * 2)) return -1;
    return 0;
}
static void synth_free(nemo_gpu *g, synth *t) {
    CUdeviceptr d[5] = {t->w.idx, t->w.mn, t->w.sc, t->w.book, t->w.bookh};
    for (int i = 0; i < 5; i++) if (d[i]) g->cu.MemFree(d[i]);
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
    int rc = -1;
    synth t[2]; int nt = 0;
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
            synth *T = &t[nt];
            if (synth_make(g, T, shp[si][0], shp[si][1], 1)) { snprintf(why, wl, "synthetic upload failed: %s", g->err); goto out; }
            nt++;
            for (int mode = 0; mode < 2; mode++) {
                g->exact = mode;
                if (pq_gemv(g, &T->w, dx, dy) || ck(g, g->cu.CtxSynchronize(), "selftest")
                    || ck(g, g->cu.MemcpyDtoH(y, dy, (size_t)T->w.rows * 4), "selftest copy")) { snprintf(why, wl, "%s", g->err); goto out; }
                for (int row = 0; row < T->w.rows; row++) {
                    float ref = mode ? full_ref(T->idx, T->mn, T->sc, T->book, x, row, T->w.cols)
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
        synth *T = &t[nt];
        if (synth_make(g, T, 22656, 4480, 0)) { snprintf(why, wl, "bench upload failed: %s", g->err); goto out; }
        nt++;
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
