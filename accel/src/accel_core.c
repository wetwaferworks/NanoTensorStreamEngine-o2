/* accel_core.c - NTSE-o2 acceleration core (research, scratch).
 *
 * Direction: JD. Lanes proposed by Gizmo. NTSE, nq2 and the shared
 * DTC kernel: Elfie. Co-implemented by Cortana.
 *
 * What lives here: input validation, the CPU reference (which IS
 * tools/nq2-dc-kernel.h, included, not copied), the device layout
 * of a dictionary-encoded tensor, the device-memory budget, lane
 * selection and the fallback policy, and the GPU side of the
 * Tensor Handshake (streamed DTC with explicit slot ownership).
 *
 * Zero mutexes: the only concurrency is the streaming stager
 * thread, coordinated with release/acquire atomics, CAS claim
 * words and the handshake door from tools/nq2-handshake.h.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "accel_internal.h"
#include "../../tools/nq2-dc-kernel.h"
#include "../../tools/nq2-handshake.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <pthread.h>
#endif

/* ------------------------------------------------------------ */
/* small utilities                                               */
/* ------------------------------------------------------------ */

const char *ntse_accel_strerror(int c)
{
    switch (c) {
    case NTSE_OK: return "ok";
    case NTSE_E_INVAL: return "invalid argument or malformed tensor";
    case NTSE_E_UNSUPPORTED: return "not supported on this lane";
    case NTSE_E_NOMEM: return "host allocation failed";
    case NTSE_E_BUDGET: return "device memory budget exceeded";
    case NTSE_E_DEVICE: return "device API failure";
    case NTSE_E_NODEV: return "lane not present";
    case NTSE_E_STATE: return "buffer ownership violation refused";
    case NTSE_E_NONFINITE: return "non-finite book or frame value";
    default: return "unknown status";
    }
}

const char *ntse_lane_name(ntse_lane l)
{
    switch (l) {
    case NTSE_LANE_AUTO: return "auto";
    case NTSE_LANE_CPU: return "cpu";
    case NTSE_LANE_CUDA: return "cuda";
    case NTSE_LANE_CL_IGPU: return "opencl-igpu";
    case NTSE_LANE_CL_GPU: return "opencl-gpu";
    default: return "?";
    }
}

const char *ntse_op_name(ntse_op op)
{
    switch (op) {
    case NTSE_OP_DTC_EAGER: return "dtc-eager";
    case NTSE_OP_DTC_ACTIVE: return "dtc-active";
    case NTSE_OP_DTC_DIRECT: return "dtc-direct";
    case NTSE_OP_EXPANDED: return "expanded";
    case NTSE_OP_DTC_LAZY: return "dtc-lazy";
    default: return "?";
    }
}

float ntse_half_to_float(uint16_t h)
{
    /* Bit-for-bit the conversion in tools/nq2-infer.c f16_to_float. */
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while (!(mant & 0x400u)) { mant <<= 1; exp--; }
            mant &= 0x3ffu;
            bits = sign | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

void *ntse_dl_open(const char *name)
{
#ifdef _WIN32
    /* Full paths load with their own directory on the search path
     * (NVRTC needs its builtins DLL beside it). */
    if (strchr(name, '\\') || strchr(name, '/'))
        return (void *)LoadLibraryExA(name, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    return (void *)LoadLibraryA(name);
#else
    return dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
}

void *ntse_dl_sym(void *lib, const char *sym)
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

void ntse_dl_close(void *lib)
{
    if (!lib) return;
#ifdef _WIN32
    FreeLibrary((HMODULE)lib);
#else
    dlclose(lib);
#endif
}

static double now_ms(void)
{
#ifdef _WIN32
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)f.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
#endif
}

static uint64_t al(uint64_t v)
{
    return (v + (NTSE_ALIGN - 1)) & ~(uint64_t)(NTSE_ALIGN - 1);
}

static void say(char *why, size_t n, const char *msg)
{
    if (why && n) {
        strncpy(why, msg, n - 1);
        why[n - 1] = '\0';
    }
}

/* ------------------------------------------------------------ */
/* validation                                                    */
/* ------------------------------------------------------------ */

/* rows*cols is capped at 2^40 weights so every derived byte count
 * below fits in 64 bits with a wide margin. */
#define NTSE_MAX_WEIGHTS (1ull << 40)

static int geom_ok(const ntse_dtc_tensor *t, char *why, size_t wl)
{
    char msg[200];
    if (!t) { say(why, wl, "null tensor"); return NTSE_E_INVAL; }
    if (!t->book || !t->idx || !t->mn_h || !t->sc_h) {
        say(why, wl, "null book, index or frame array");
        return NTSE_E_INVAL;
    }
    if (t->block == 0 || t->block > NTSE_MAX_BLOCK
        || t->entries == 0 || t->entries > NTSE_MAX_ENTRIES) {
        snprintf(msg, sizeof msg, "unsupported geometry: block %u, entries %u",
                 t->block, t->entries);
        say(why, wl, msg);
        return NTSE_E_INVAL;
    }
    if (t->rows == 0 || t->cols == 0 || t->rows > NTSE_MAX_DIM
        || t->cols > NTSE_MAX_DIM) {
        say(why, wl, "rows/cols out of range");
        return NTSE_E_INVAL;
    }
    if (t->rows > NTSE_MAX_WEIGHTS / t->cols) {
        say(why, wl, "rows*cols exceeds the 2^40 weight cap");
        return NTSE_E_INVAL;
    }
    if (t->cols % t->block != 0) {
        snprintf(msg, sizeof msg, "cols %llu not a multiple of block %u: "
                 "blocks straddle rows, DTC does not apply (expand on the CPU)",
                 (unsigned long long)t->cols, t->block);
        say(why, wl, msg);
        return NTSE_E_UNSUPPORTED;
    }
    return NTSE_OK;
}

int ntse_dtc_validate(const ntse_dtc_tensor *t, char *why, size_t wl)
{
    int rc = geom_ok(t, why, wl);
    if (rc != NTSE_OK) return rc;
    char msg[200];
    uint64_t nseg = t->cols / t->block, nblk = t->rows * nseg;
    for (uint64_t b = 0; b < nblk; b++) {
        if (t->idx[b] >= t->entries) {
            snprintf(msg, sizeof msg, "index %u at block %llu exceeds entries %u",
                     t->idx[b], (unsigned long long)b, t->entries);
            say(why, wl, msg);
            return NTSE_E_INVAL;
        }
        if ((t->mn_h[b] & 0x7c00u) == 0x7c00u
            || (t->sc_h[b] & 0x7c00u) == 0x7c00u) {
            snprintf(msg, sizeof msg, "frame at block %llu is not finite",
                     (unsigned long long)b);
            say(why, wl, msg);
            return NTSE_E_NONFINITE;
        }
    }
    uint64_t nb = (uint64_t)t->entries * t->block;
    for (uint64_t i = 0; i < nb; i++) {
        if (!isfinite(t->book[i])) {
            snprintf(msg, sizeof msg, "book value %llu is not finite",
                     (unsigned long long)i);
            say(why, wl, msg);
            return NTSE_E_NONFINITE;
        }
    }
    return NTSE_OK;
}

/* ------------------------------------------------------------ */
/* CPU reference                                                 */
/* ------------------------------------------------------------ */

int ntse_cpu_matvec(const ntse_dtc_tensor *t, ntse_op op, const float *x,
                    uint32_t batch, float *y)
{
    if (geom_ok(t, NULL, 0) != NTSE_OK || !x || !y || batch == 0
        || batch > NTSE_MAX_BATCH || op < NTSE_OP_DTC_EAGER
        || op > NTSE_OP_DTC_LAZY)
        return NTSE_E_INVAL;
    uint64_t nseg = t->cols / t->block, nblk = t->rows * nseg;
    float *mn = (float *)malloc(nblk * sizeof(float));
    float *sc = (float *)malloc(nblk * sizeof(float));
    if (!mn || !sc) { free(mn); free(sc); return NTSE_E_NOMEM; }
    for (uint64_t b = 0; b < nblk; b++) {
        mn[b] = ntse_half_to_float(t->mn_h[b]);
        sc[b] = ntse_half_to_float(t->sc_h[b]);
    }
    int rc = NTSE_OK;
    if (op == NTSE_OP_EXPANDED) {
        /* materialize one row at a time, sequential fp32 dot */
        float *w = (float *)malloc(t->cols * sizeof(float));
        if (!w) { rc = NTSE_E_NOMEM; goto out; }
        for (uint64_t r = 0; r < t->rows; r++) {
            for (uint64_t j = 0; j < nseg; j++) {
                uint64_t b = r * nseg + j;
                const float *proto = t->book + (size_t)t->idx[b] * t->block;
                for (uint32_t i = 0; i < t->block; i++)
                    w[j * t->block + i] = mn[b] + sc[b] * proto[i];
            }
            for (uint32_t bb = 0; bb < batch; bb++) {
                const float *xb = x + (size_t)bb * t->cols;
                float acc = 0;
                for (uint64_t c = 0; c < t->cols; c++) acc += w[c] * xb[c];
                y[(size_t)bb * t->rows + r] = acc;
            }
        }
        free(w);
    } else {
        /* All four DTC ops have one reference: the shared kernel.
         * Eager, active, direct and lazy produce the same bits by
         * construction (same entry order, same apply order). */
        float *T = (float *)malloc(nseg * t->entries * sizeof(float));
        float *A = (float *)malloc(nseg * sizeof(float));
        if (!T || !A) { free(T); free(A); rc = NTSE_E_NOMEM; goto out; }
        for (uint32_t bb = 0; bb < batch; bb++) {
            nqdc_build(t->book, t->entries, t->block,
                       x + (size_t)bb * t->cols, nseg, T, A);
            nqdc_apply(t->idx, sc, mn, T, A, t->entries, nseg, t->rows,
                       y + (size_t)bb * t->rows);
        }
        free(T); free(A);
    }
out:
    free(mn); free(sc);
    return rc;
}

int ntse_cpu_oracle(const ntse_dtc_tensor *t, const float *x,
                    uint32_t batch, double *y)
{
    if (geom_ok(t, NULL, 0) != NTSE_OK || !x || !y || batch == 0
        || batch > NTSE_MAX_BATCH)
        return NTSE_E_INVAL;
    uint64_t nseg = t->cols / t->block;
    for (uint32_t bb = 0; bb < batch; bb++) {
        const float *xb = x + (size_t)bb * t->cols;
        for (uint64_t r = 0; r < t->rows; r++) {
            double acc = 0;
            for (uint64_t j = 0; j < nseg; j++) {
                uint64_t b = r * nseg + j;
                float m = ntse_half_to_float(t->mn_h[b]);
                float s = ntse_half_to_float(t->sc_h[b]);
                const float *proto = t->book + (size_t)t->idx[b] * t->block;
                for (uint32_t i = 0; i < t->block; i++) {
                    float w = m + s * proto[i];   /* the decoder's weight */
                    acc += (double)w * (double)xb[j * t->block + i];
                }
            }
            y[(size_t)bb * t->rows + r] = acc;
        }
    }
    return NTSE_OK;
}

/* ------------------------------------------------------------ */
/* device layout of one tensor                                   */
/* ------------------------------------------------------------ */
/* One allocation, sections aligned to NTSE_ALIGN:
 *   bookT  [block][entries] fp32  (coalesced table build)
 *   book   [entries][block] fp32  (direct op, expansion)
 *   idxT   [nseg][rows] u8        (thread r reads idxT[j*rows+r],
 *   mnT    [nseg][rows] fp16 bits   consecutive across threads)
 *   scT    [nseg][rows] fp16 bits
 *   act    active (j,k) codes j*entries+k, ascending (j, k)
 * The transposition is host work, done once at upload or by the
 * stager while the device runs earlier items. */
typedef struct {
    uint64_t nseg, nblk;
    uint64_t o_bookT, o_book, o_idxT, o_mnT, o_scT, o_act, act_cap;
    uint64_t total;
} ntse_layout;

static int act_encodable(const ntse_dtc_tensor *t)
{
    uint64_t nseg = t->cols / t->block;
    return nseg <= (0xffffffffull / t->entries);
}

static void layout_of(const ntse_dtc_tensor *t, ntse_layout *L)
{
    uint64_t book = (uint64_t)t->entries * t->block * 4;
    L->nseg = t->cols / t->block;
    L->nblk = t->rows * L->nseg;
    L->o_bookT = 0;
    L->o_book = al(L->o_bookT + book);
    L->o_idxT = al(L->o_book + book);
    L->o_mnT = al(L->o_idxT + L->nblk);
    L->o_scT = al(L->o_mnT + L->nblk * 2);
    L->o_act = al(L->o_scT + L->nblk * 2);
    /* the active set of a segment has at most min(entries, rows) codes */
    uint64_t per = t->entries < t->rows ? t->entries : t->rows;
    L->act_cap = act_encodable(t) ? L->nseg * per : 0;
    L->total = al(L->o_act + L->act_cap * 4 + 4);
}

/* Fill the layout into dst (L->total bytes, NTSE_ALIGN aligned or
 * at least 4-aligned). Returns the active count. */
static uint64_t layout_fill(const ntse_dtc_tensor *t, const ntse_layout *L,
                            uint8_t *dst)
{
    const uint32_t K = t->entries, B = t->block;
    float *bookT = (float *)(dst + L->o_bookT);
    for (uint32_t k = 0; k < K; k++)
        for (uint32_t i = 0; i < B; i++)
            bookT[(size_t)i * K + k] = t->book[(size_t)k * B + i];
    memcpy(dst + L->o_book, t->book, (size_t)K * B * 4);
    uint8_t *idxT = dst + L->o_idxT;
    uint16_t *mnT = (uint16_t *)(dst + L->o_mnT);
    uint16_t *scT = (uint16_t *)(dst + L->o_scT);
    for (uint64_t r = 0; r < t->rows; r++) {
        for (uint64_t j = 0; j < L->nseg; j++) {
            uint64_t s = r * L->nseg + j, d = j * t->rows + r;
            idxT[d] = t->idx[s];
            mnT[d] = t->mn_h[s];
            scT[d] = t->sc_h[s];
        }
    }
    uint64_t nact = 0;
    if (L->act_cap) {
        uint32_t *act = (uint32_t *)(dst + L->o_act);
        for (uint64_t j = 0; j < L->nseg; j++) {
            uint8_t used[NTSE_MAX_ENTRIES];
            memset(used, 0, sizeof used);
            const uint8_t *col = idxT + j * t->rows;
            for (uint64_t r = 0; r < t->rows; r++) used[col[r]] = 1;
            for (uint32_t k = 0; k < K; k++)
                if (used[k]) act[nact++] = (uint32_t)(j * K + k);
        }
    }
    return nact;
}

/* ------------------------------------------------------------ */
/* context                                                       */
/* ------------------------------------------------------------ */

struct ntse_dtensor {
    ntse_dtc_tensor h;     /* borrowed host arrays (CPU fallback) */
    int on_device;
    ntse_dmem mem;
    ntse_layout L;
    uint64_t nact;
};

struct ntse_accel {
    ntse_accel_config cfg;
    ntse_lane lane;
    int failed;
    char device[320];
    ntse_be *be;
    const ntse_be_ops *ops;
    ntse_be_info info;
    uint64_t budget, charged, peak, h2d;
    ntse_dmem ws;      /* matvec workspace: x, A, T, y */
    ntse_dmem wexp;    /* expanded-weights scratch      */
    ntse_dmem lin;     /* bf16 projection scratch       */
    ntse_dmem wflag;   /* lazy-op flags                 */
    int force_early_recycle;
};

void ntse_accel_config_default(ntse_accel_config *c)
{
    c->lane = NTSE_LANE_AUTO;
    c->vram_budget = 0;
    c->allow_fallback = 1;
    c->verbose = 0;
}

ntse_lane ntse_accel_lane(const ntse_accel *a)
{
    return (!a || a->failed) ? NTSE_LANE_CPU : a->lane;
}
const char *ntse_accel_device(const ntse_accel *a) { return a ? a->device : ""; }
int ntse_accel_failed(const ntse_accel *a) { return a ? a->failed : 1; }
uint64_t ntse_accel_budget(const ntse_accel *a) { return a ? a->budget : 0; }
uint64_t ntse_accel_charged(const ntse_accel *a) { return a ? a->charged : 0; }
uint64_t ntse_accel_peak(const ntse_accel *a) { return a ? a->peak : 0; }
uint64_t ntse_accel_h2d_bytes(const ntse_accel *a) { return a ? a->h2d : 0; }
void ntse_accel_test_force_early_recycle(ntse_accel *a, int on)
{
    if (a) a->force_early_recycle = on ? 1 : 0;
}

static int on_gpu(const ntse_accel *a)
{
    return a && a->be && !a->failed && a->lane != NTSE_LANE_CPU;
}

/* A device failure is sticky: the context drops to the CPU lane. */
static void fault(ntse_accel *a, int rc, const char *what)
{
    if (rc == NTSE_E_DEVICE && !a->failed) {
        a->failed = 1;
        fprintf(stderr, "ntse-accel: FAULT on %s during %s: %s; this "
                "context now runs on the CPU reference\n",
                ntse_lane_name(a->lane), what, ntse_accel_strerror(rc));
    }
}

static int charge_alloc(ntse_accel *a, uint64_t bytes, ntse_dmem *m)
{
    memset(m, 0, sizeof *m);
    if (bytes == 0) bytes = NTSE_ALIGN;
    if (bytes > a->budget || a->charged > a->budget - bytes)
        return NTSE_E_BUDGET;
    if (a->info.max_alloc && bytes > a->info.max_alloc)
        return NTSE_E_BUDGET;
    int rc = a->ops->dalloc(a->be, bytes, m);
    if (rc != NTSE_OK) {
        memset(m, 0, sizeof *m);
        return rc;
    }
    a->charged += m->bytes;
    if (a->charged > a->peak) a->peak = a->charged;
    return NTSE_OK;
}

static void charge_free(ntse_accel *a, ntse_dmem *m)
{
    if (!m->bytes) return;
    a->ops->dfree(a->be, m);
    a->charged -= m->bytes;
    memset(m, 0, sizeof *m);
}

static int ensure(ntse_accel *a, ntse_dmem *m, uint64_t bytes)
{
    if (m->bytes >= bytes) return NTSE_OK;
    charge_free(a, m);
    int rc = charge_alloc(a, bytes, m);
    if (rc != NTSE_OK) return rc;
    return a->ops->zero(a->be, NTSE_Q_COMPUTE, m);
}

static uint64_t env_mb(const char *name)
{
    const char *v = getenv(name);
    if (!v || !*v) return 0;
    char *end = NULL;
    unsigned long long mb = strtoull(v, &end, 10);
    if (end == v || mb == 0 || mb > (1ull << 20)) return 0;
    return (uint64_t)mb << 20;
}

static int lane_from_env(ntse_lane *out)
{
    const char *v = getenv("NTSE_ACCEL_LANE");
    if (!v || !*v) return 0;
    if (!strcmp(v, "cpu")) *out = NTSE_LANE_CPU;
    else if (!strcmp(v, "cuda")) *out = NTSE_LANE_CUDA;
    else if (!strcmp(v, "igpu") || !strcmp(v, "opencl-igpu")) *out = NTSE_LANE_CL_IGPU;
    else if (!strcmp(v, "gpu-cl") || !strcmp(v, "opencl-gpu")) *out = NTSE_LANE_CL_GPU;
    else if (!strcmp(v, "auto")) *out = NTSE_LANE_AUTO;
    else return -1;
    return 1;
}

static int try_lane(ntse_accel *a, ntse_lane l, char *why, size_t wl)
{
    int rc;
    memset(&a->info, 0, sizeof a->info);
    if (l == NTSE_LANE_CUDA)
        rc = ntse_cuda_create(&a->be, &a->ops, &a->info, why, wl);
    else
        rc = ntse_cl_create(l, &a->be, &a->ops, &a->info, why, wl);
    if (rc == NTSE_OK) a->lane = l;
    return rc;
}

int ntse_accel_open(ntse_accel **out, const ntse_accel_config *cfg)
{
    if (!out) return NTSE_E_INVAL;
    *out = NULL;
    ntse_accel *a = (ntse_accel *)calloc(1, sizeof *a);
    if (!a) return NTSE_E_NOMEM;
    if (cfg) a->cfg = *cfg; else ntse_accel_config_default(&a->cfg);
    a->lane = NTSE_LANE_CPU;
    snprintf(a->device, sizeof a->device, "cpu reference");

    ntse_lane want = a->cfg.lane;
    int er = lane_from_env(&want);
    if (er < 0) {
        fprintf(stderr, "ntse-accel: NTSE_ACCEL_LANE has an unknown value; "
                "using the configured lane\n");
        want = a->cfg.lane;
    }
    const char *dis = getenv("NTSE_ACCEL_DISABLE");
    if (dis && *dis && strcmp(dis, "0") != 0) want = NTSE_LANE_CPU;

    char why[256] = "", last[512] = "";
    int rc = NTSE_E_NODEV;
    if (want == NTSE_LANE_AUTO) {
        const ntse_lane order[3] = { NTSE_LANE_CUDA, NTSE_LANE_CL_GPU,
                                     NTSE_LANE_CL_IGPU };
        for (int i = 0; i < 3 && rc != NTSE_OK; i++) {
            why[0] = '\0';
            rc = try_lane(a, order[i], why, sizeof why);
            if (rc != NTSE_OK) {
                size_t n = strlen(last);
                snprintf(last + n, sizeof last - n, "%s%s: %s",
                         n ? "; " : "", ntse_lane_name(order[i]), why);
            }
        }
    } else if (want != NTSE_LANE_CPU) {
        rc = try_lane(a, want, why, sizeof why);
        if (rc != NTSE_OK)
            snprintf(last, sizeof last, "%s: %s", ntse_lane_name(want), why);
    }

    if (rc != NTSE_OK) {
        a->be = NULL; a->ops = NULL; a->lane = NTSE_LANE_CPU;
        if (want != NTSE_LANE_CPU && !a->cfg.allow_fallback) {
            if (a->cfg.verbose)
                fprintf(stderr, "ntse-accel: no lane (%s) and fallback "
                        "is off\n", last);
            free(a);
            return rc;
        }
        if (a->cfg.verbose)
            fprintf(stderr, "ntse-accel: lane cpu (reference)%s%s\n",
                    last[0] ? "; GPU lanes unavailable: " : "", last);
        *out = a;
        return NTSE_OK;
    }

    /* Budget: hardware-derived ceiling, overrides clamped loudly. */
    uint64_t hard = a->info.integrated ? (512ull << 20) : (3ull << 30);
    uint64_t basis = a->info.free_mem ? a->info.free_mem / 4 * 3
                                      : a->info.total_mem / 2;
    uint64_t ceiling = basis < hard ? basis : hard;
    uint64_t want_b = a->cfg.vram_budget;
    const char *src = "derived";
    uint64_t eb = env_mb("NTSE_ACCEL_VRAM_MB");
    if (eb) { want_b = eb; src = "NTSE_ACCEL_VRAM_MB"; }
    else if (want_b) src = "config";
    if (want_b && want_b > ceiling) {
        fprintf(stderr, "ntse-accel: budget %llu MiB (%s) exceeds the "
                "hardware ceiling %llu MiB; clamped\n",
                (unsigned long long)(want_b >> 20), src,
                (unsigned long long)(ceiling >> 20));
        want_b = ceiling;
        src = "clamped";
    }
    a->budget = want_b ? want_b : ceiling;
    snprintf(a->device, sizeof a->device, "%s", a->info.device);
    if (a->cfg.verbose)
        fprintf(stderr, "ntse-accel: lane %s | %s | mem %llu MiB total, "
                "%llu MiB free | budget %llu MiB (%s) | fp64 %s | "
                "fp32 denormals %s\n",
                ntse_lane_name(a->lane), a->info.device,
                (unsigned long long)(a->info.total_mem >> 20),
                (unsigned long long)(a->info.free_mem >> 20),
                (unsigned long long)(a->budget >> 20), src,
                a->info.has_fp64 ? "yes" : "no",
                a->info.denorms ? "honored" : "flushed");
    *out = a;
    return NTSE_OK;
}

void ntse_accel_close(ntse_accel *a)
{
    if (!a) return;
    if (a->be) {
        a->ops->qsync(a->be, NTSE_Q_COMPUTE);
        a->ops->qsync(a->be, NTSE_Q_COPY);
        charge_free(a, &a->ws);
        charge_free(a, &a->wexp);
        charge_free(a, &a->lin);
        charge_free(a, &a->wflag);
        a->ops->destroy(a->be);
    }
    free(a);
}

/* ------------------------------------------------------------ */
/* upload / matvec                                               */
/* ------------------------------------------------------------ */

int ntse_accel_upload(ntse_accel *a, const ntse_dtc_tensor *t,
                      ntse_dtensor **out)
{
    if (!a || !out) return NTSE_E_INVAL;
    *out = NULL;
    char why[200];
    int rc = ntse_dtc_validate(t, why, sizeof why);
    if (rc != NTSE_OK) {
        if (a->cfg.verbose) fprintf(stderr, "ntse-accel: refused: %s\n", why);
        return rc;
    }
    ntse_dtensor *d = (ntse_dtensor *)calloc(1, sizeof *d);
    if (!d) return NTSE_E_NOMEM;
    d->h = *t;
    layout_of(t, &d->L);
    if (!on_gpu(a)) { *out = d; return NTSE_OK; }

    uint8_t *host = (uint8_t *)calloc(1, d->L.total);
    if (!host) { free(d); return NTSE_E_NOMEM; }
    d->nact = layout_fill(t, &d->L, host);
    rc = charge_alloc(a, d->L.total, &d->mem);
    if (rc == NTSE_OK) {
        rc = a->ops->put(a->be, NTSE_Q_COMPUTE, &d->mem, 0, host,
                         d->L.total, NULL);
        if (rc == NTSE_OK) { a->h2d += d->L.total; d->on_device = 1; }
        else charge_free(a, &d->mem);
    }
    free(host);
    if (rc != NTSE_OK) {
        fault(a, rc, "upload");
        if (!a->cfg.allow_fallback) { free(d); return rc; }
        /* host-only handle: matvec runs the CPU reference */
    }
    *out = d;
    return NTSE_OK;
}

void ntse_accel_release(ntse_accel *a, ntse_dtensor *d)
{
    if (!d) return;
    if (a && a->be && d->on_device) {
        a->ops->qsync(a->be, NTSE_Q_COMPUTE);
        charge_free(a, &d->mem);
    }
    free(d);
}

static void kargs_tensor(ntse_kargs *k, const ntse_dtensor *d)
{
    memset(k, 0, sizeof *k);
    k->bookT.m = &d->mem; k->bookT.off = d->L.o_bookT;
    k->book.m = &d->mem;  k->book.off = d->L.o_book;
    k->idxT.m = &d->mem;  k->idxT.off = d->L.o_idxT;
    k->mnT.m = &d->mem;   k->mnT.off = d->L.o_mnT;
    k->scT.m = &d->mem;   k->scT.off = d->L.o_scT;
    k->act.m = &d->mem;   k->act.off = d->L.o_act;
    k->rows = d->h.rows; k->cols = d->h.cols; k->nseg = d->L.nseg;
    k->nact = d->nact; k->block = d->h.block; k->entries = d->h.entries;
}

/* Enqueue the kernels of one DTC op on queue q. */
static int enqueue_op(ntse_accel *a, int q, ntse_op op, const ntse_kargs *k)
{
    int rc;
    if (op == NTSE_OP_EXPANDED) {
        rc = a->ops->launch(a->be, q, NTSE_K_EXPAND, k);
        if (rc == NTSE_OK) rc = a->ops->launch(a->be, q, NTSE_K_GEMV, k);
        return rc;
    }
    rc = a->ops->launch(a->be, q, NTSE_K_SEGSUM, k);
    if (rc != NTSE_OK) return rc;
    if (op == NTSE_OP_DTC_EAGER) {
        rc = a->ops->launch(a->be, q, NTSE_K_BUILD_EAGER, k);
        if (rc == NTSE_OK) rc = a->ops->launch(a->be, q, NTSE_K_APPLY, k);
    } else if (op == NTSE_OP_DTC_ACTIVE) {
        if (k->nact) rc = a->ops->launch(a->be, q, NTSE_K_BUILD_ACTIVE, k);
        if (rc == NTSE_OK) rc = a->ops->launch(a->be, q, NTSE_K_APPLY, k);
    } else if (op == NTSE_OP_DTC_LAZY) {
        /* flags cleared, marked from this call's index, then only the
         * marked entries built; nothing is assumed from upload time */
        if (!k->F.m) return NTSE_E_STATE;
        rc = a->ops->zero(a->be, q, k->F.m);
        if (rc == NTSE_OK) rc = a->ops->launch(a->be, q, NTSE_K_MARK, k);
        if (rc == NTSE_OK) rc = a->ops->launch(a->be, q, NTSE_K_BUILD_LAZY, k);
        if (rc == NTSE_OK) rc = a->ops->launch(a->be, q, NTSE_K_APPLY, k);
    } else {
        rc = a->ops->launch(a->be, q, NTSE_K_APPLY_DIRECT, k);
    }
    return rc;
}

static int gpu_matvec(ntse_accel *a, ntse_dtensor *d, ntse_op op,
                      const float *x, uint32_t batch, float *y)
{
    ntse_kargs k;
    kargs_tensor(&k, d);
    k.batch = batch;
    if (op == NTSE_OP_DTC_ACTIVE && d->L.act_cap == 0)
        return NTSE_E_UNSUPPORTED;
    uint64_t xb = (uint64_t)batch * d->h.cols * 4;
    uint64_t Ab = (uint64_t)batch * d->L.nseg * 4;
    uint64_t Tb = (op == NTSE_OP_DTC_EAGER || op == NTSE_OP_DTC_ACTIVE
                   || op == NTSE_OP_DTC_LAZY)
                      ? (uint64_t)batch * d->L.nseg * d->h.entries * 4 : 0;
    uint64_t yb = (uint64_t)batch * d->h.rows * 4;
    uint64_t oA = al(xb), oT = al(oA + Ab), oy = al(oT + Tb);
    uint64_t total = al(oy + yb);
    int rc = ensure(a, &a->ws, total);
    if (rc != NTSE_OK) return rc;
    k.x.m = &a->ws; k.x.off = 0;
    k.A.m = &a->ws; k.A.off = oA;
    k.T.m = &a->ws; k.T.off = oT;
    k.y.m = &a->ws; k.y.off = oy;
    if (op == NTSE_OP_DTC_LAZY) {
        /* flags live alone in their buffer: zero() clears all of it */
        uint64_t fb = d->L.nseg * d->h.entries * 4;
        if (a->wflag.bytes < al(fb)) {
            charge_free(a, &a->wflag);
            rc = charge_alloc(a, al(fb), &a->wflag);
            if (rc != NTSE_OK) return rc;
        }
        k.F.m = &a->wflag; k.F.off = 0;
    }
    if (op == NTSE_OP_EXPANDED) {
        rc = ensure(a, &a->wexp, d->h.rows * d->h.cols * 4);
        if (rc != NTSE_OK) return rc;
        k.W.m = &a->wexp; k.W.off = 0;
    }
    rc = a->ops->put(a->be, NTSE_Q_COMPUTE, &a->ws, 0, x, xb, NULL);
    if (rc != NTSE_OK) return rc;
    a->h2d += xb;
    rc = enqueue_op(a, NTSE_Q_COMPUTE, op, &k);
    if (rc != NTSE_OK) return rc;
    return a->ops->get(a->be, NTSE_Q_COMPUTE, y, &a->ws, oy, yb, NULL);
}

int ntse_accel_matvec(ntse_accel *a, ntse_dtensor *d, ntse_op op,
                      const float *x, uint32_t batch, float *y,
                      ntse_lane *used)
{
    if (used) *used = NTSE_LANE_CPU;
    if (!a || !d || !x || !y || batch == 0 || batch > NTSE_MAX_BATCH
        || op < NTSE_OP_DTC_EAGER || op > NTSE_OP_DTC_LAZY)
        return NTSE_E_INVAL;
    if (on_gpu(a) && d->on_device) {
        int rc = gpu_matvec(a, d, op, x, batch, y);
        if (rc == NTSE_OK) { if (used) *used = a->lane; return NTSE_OK; }
        if (rc == NTSE_E_INVAL) return rc;
        fault(a, rc, ntse_op_name(op));
        if (!a->cfg.allow_fallback) return rc;
        if (a->cfg.verbose && rc != NTSE_E_DEVICE)
            fprintf(stderr, "ntse-accel: %s on %s: %s; this call runs "
                    "on the CPU reference\n", ntse_op_name(op),
                    ntse_lane_name(a->lane), ntse_accel_strerror(rc));
    }
    return ntse_cpu_matvec(&d->h, op, x, batch, y);
}

int ntse_accel_linear_bf16(ntse_accel *a, const uint16_t *W, const float *x,
                           float *y, int out, int in, ntse_lane *used)
{
    if (used) *used = NTSE_LANE_CPU;
    if (!a || !W || !x || !y || out <= 0 || in <= 0
        || (uint64_t)out * (uint64_t)in > (1ull << 34))
        return NTSE_E_INVAL;
    /* No CPU copy of the projection lives here on purpose: the
     * caller's own kernel (Little Nemo nn.c) is the golden
     * reference and the fallback. */
    if (!on_gpu(a)) return NTSE_E_NODEV;
    uint64_t wb = (uint64_t)out * in * 2, xb = (uint64_t)in * 4,
             yb = (uint64_t)out * 4;
    uint64_t ox = al(wb), oy = al(ox + xb), total = al(oy + yb);
    int rc = ensure(a, &a->lin, total);
    if (rc == NTSE_OK)
        rc = a->ops->put(a->be, NTSE_Q_COMPUTE, &a->lin, 0, W, wb, NULL);
    if (rc == NTSE_OK)
        rc = a->ops->put(a->be, NTSE_Q_COMPUTE, &a->lin, ox, x, xb, NULL);
    if (rc == NTSE_OK) {
        a->h2d += wb + xb;
        ntse_dref dW = { &a->lin, 0 }, dx = { &a->lin, ox }, dy = { &a->lin, oy };
        rc = a->ops->linear_bf16(a->be, NTSE_Q_COMPUTE, dW, dx, dy,
                                 (uint32_t)out, (uint32_t)in);
    }
    if (rc == NTSE_OK)
        rc = a->ops->get(a->be, NTSE_Q_COMPUTE, y, &a->lin, oy, yb, NULL);
    if (rc == NTSE_OK) { if (used) *used = a->lane; return NTSE_OK; }
    fault(a, rc, "linear-bf16");
    return rc;
}

/* ------------------------------------------------------------ */
/* Tensor Handshake, GPU side: streamed DTC                      */
/* ------------------------------------------------------------ */

#define NTSE_MAX_SLOTS 4
enum { SLOT_FREE = 0, SLOT_STAGING = 1, SLOT_STAGED = 2, SLOT_INFLIGHT = 3 };

typedef struct {
    ntse_accel *a;
    const ntse_stream_item *items;
    uint32_t n, batch, S;
    ntse_op op;
    uint8_t *pin[NTSE_MAX_SLOTS];
    void *tok[NTSE_MAX_SLOTS];
    ntse_dmem dev[NTSE_MAX_SLOTS];
    /* per-slot facts, written by the slot's owner before it
     * publishes the next state (release), read after acquire */
    ntse_layout lay[NTSE_MAX_SLOTS];
    uint64_t nact[NTSE_MAX_SLOTS], o_x[NTSE_MAX_SLOTS], o_y[NTSE_MAX_SLOTS],
             o_A[NTSE_MAX_SLOTS], o_T[NTSE_MAX_SLOTS];
    uint32_t item[NTSE_MAX_SLOTS];
    int state[NTSE_MAX_SLOTS];
    /* the ordinal whose completion the host has OBSERVED on this
     * slot, plus one (0 = nothing observed since staging) */
    uint32_t observed[NTSE_MAX_SLOTS];
    uint32_t door;
    uint32_t stop;
    uint32_t exited;
    uint64_t stager_waits, driver_waits, violations;
} stream_ctx;

static void stage_into(stream_ctx *c, uint32_t s, uint32_t n)
{
    const ntse_stream_item *it = &c->items[n];
    ntse_layout *L = &c->lay[s];
    layout_of(it->t, L);
    c->nact[s] = layout_fill(it->t, L, c->pin[s]);
    uint64_t xb = (uint64_t)c->batch * it->t->cols * 4;
    uint64_t yb = (uint64_t)c->batch * it->t->rows * 4;
    uint64_t Ab = (uint64_t)c->batch * L->nseg * 4;
    c->o_x[s] = L->total;
    memcpy(c->pin[s] + c->o_x[s], it->x, xb);
    c->o_y[s] = al(c->o_x[s] + xb);
    c->o_A[s] = al(c->o_y[s] + yb);
    c->o_T[s] = al(c->o_A[s] + Ab);
    c->item[s] = n;
    HS_STX(&c->observed[s], 0u);
}

/* Claim a slot for staging: only FREE -> STAGING is legal. */
static int claim_for_staging(stream_ctx *c, uint32_t s)
{
    return hs_cas_int(&c->state[s], SLOT_FREE, SLOT_STAGING);
}

static HS_THREAD_FN(stager_main, arg)
{
    stream_ctx *c = (stream_ctx *)arg;
    for (uint32_t n = 0; n < c->n; n++) {
        uint32_t s = n % c->S;
        for (;;) {
            uint32_t seen = hs_door_open(&c->door);
            if (HS_LD(&c->stop)) goto done;
            if (claim_for_staging(c, s)) break;
            c->stager_waits++;
            hs_door_wait(&c->door, seen);
        }
        stage_into(c, s, n);
        HS_ST(&c->state[s], SLOT_STAGED);   /* publish the staged slot */
        hs_door_bump(&c->door);
    }
done:
    hs_exit_tick(&c->exited);
    HS_THREAD_RETURN;
}

/* Return a slot to FREE. Legal only once the host has observed the
 * completion of the item the slot carries; anything else is an
 * ownership violation, refused and counted. */
static int slot_release(stream_ctx *c, uint32_t s, uint32_t n)
{
    if (HS_LD(&c->observed[s]) != n + 1 || HS_LD(&c->state[s]) != SLOT_INFLIGHT) {
        c->violations++;
        return NTSE_E_STATE;
    }
    HS_ST(&c->state[s], SLOT_FREE);
    hs_door_bump(&c->door);
    return NTSE_OK;
}

static int driver_enqueue(stream_ctx *c, uint32_t s, uint32_t n)
{
    ntse_accel *a = c->a;
    const ntse_be_ops *o = a->ops;
    const ntse_dtc_tensor *t = c->items[n].t;
    if (!hs_cas_int(&c->state[s], SLOT_STAGED, SLOT_INFLIGHT)) {
        c->violations++;
        return NTSE_E_STATE;
    }
    /* the item carried is n, staged by its single owner */
    if (c->item[s] != n) { c->violations++; return NTSE_E_STATE; }
    uint64_t xb = (uint64_t)c->batch * t->cols * 4;
    uint64_t up = c->o_x[s] + xb;
    int rc = o->put(a->be, NTSE_Q_COPY, &c->dev[s], 0, c->pin[s], up, c->tok[s]);
    if (rc == NTSE_OK) rc = o->ev_record(a->be, NTSE_Q_COPY, (int)s);
    if (rc == NTSE_OK) rc = o->ev_wait(a->be, NTSE_Q_COMPUTE, (int)s);
    if (rc != NTSE_OK) return rc;
    a->h2d += up;
    ntse_kargs k;
    memset(&k, 0, sizeof k);
    const ntse_layout *L = &c->lay[s];
    k.bookT.m = &c->dev[s]; k.bookT.off = L->o_bookT;
    k.book.m = &c->dev[s];  k.book.off = L->o_book;
    k.idxT.m = &c->dev[s];  k.idxT.off = L->o_idxT;
    k.mnT.m = &c->dev[s];   k.mnT.off = L->o_mnT;
    k.scT.m = &c->dev[s];   k.scT.off = L->o_scT;
    k.act.m = &c->dev[s];   k.act.off = L->o_act;
    k.x.m = &c->dev[s];     k.x.off = c->o_x[s];
    k.y.m = &c->dev[s];     k.y.off = c->o_y[s];
    k.A.m = &c->dev[s];     k.A.off = c->o_A[s];
    k.T.m = &c->dev[s];     k.T.off = c->o_T[s];
    k.rows = t->rows; k.cols = t->cols; k.nseg = L->nseg;
    k.nact = c->nact[s]; k.block = t->block; k.entries = t->entries;
    k.batch = c->batch;
    rc = enqueue_op(a, NTSE_Q_COMPUTE, c->op, &k);
    if (rc != NTSE_OK) return rc;
    uint64_t yb = (uint64_t)c->batch * t->rows * 4;
    rc = o->get(a->be, NTSE_Q_COMPUTE, c->pin[s] + c->o_y[s], &c->dev[s],
                c->o_y[s], yb, c->tok[s]);
    if (rc == NTSE_OK)
        rc = o->ev_record(a->be, NTSE_Q_COMPUTE, NTSE_MAX_SLOTS + (int)s);
    return rc;
}

static int driver_retire(stream_ctx *c, uint32_t m, ntse_stream_stats *st)
{
    ntse_accel *a = c->a;
    uint32_t s = m % c->S;
    const ntse_dtc_tensor *t = c->items[m].t;
    if (a->force_early_recycle && m == 0) {
        /* test hook: try to recycle before observing completion */
        int r = slot_release(c, s, m);
        if (r == NTSE_OK) return NTSE_E_STATE;   /* the guard failed */
        a->force_early_recycle = 0;
    }
    int rc = a->ops->ev_sync(a->be, NTSE_MAX_SLOTS + (int)s);
    if (rc != NTSE_OK) return rc;
    /* belt and braces: the completion and the upload must both
     * report done before the slot's buffers change hands */
    if (a->ops->ev_query(a->be, NTSE_MAX_SLOTS + (int)s) != 1
        || a->ops->ev_query(a->be, (int)s) != 1) {
        c->violations++;
        return NTSE_E_STATE;
    }
    HS_ST(&c->observed[s], m + 1);
    uint64_t yb = (uint64_t)c->batch * t->rows * 4;
    memcpy(c->items[m].y, c->pin[s] + c->o_y[s], yb);   /* commit, in order */
    st->d2h_bytes += yb;
    return slot_release(c, s, m);
}

static int stream_cpu(const ntse_stream_item *items, uint32_t n,
                      uint32_t batch, ntse_op op)
{
    for (uint32_t i = 0; i < n; i++) {
        int rc = ntse_cpu_matvec(items[i].t, op, items[i].x, batch, items[i].y);
        if (rc != NTSE_OK) return rc;
    }
    return NTSE_OK;
}

int ntse_accel_stream(ntse_accel *a, const ntse_stream_item *items,
                      uint32_t n, uint32_t batch,
                      const ntse_stream_config *cfg, ntse_stream_stats *st)
{
    ntse_stream_stats local;
    if (!st) st = &local;
    memset(st, 0, sizeof *st);
    st->lane = NTSE_LANE_CPU;
    if (!a || !items || !cfg || n == 0 || batch == 0 || batch > NTSE_MAX_BATCH
        || cfg->slots == 0 || cfg->slots > NTSE_MAX_SLOTS
        || (cfg->op != NTSE_OP_DTC_EAGER && cfg->op != NTSE_OP_DTC_ACTIVE
            && cfg->op != NTSE_OP_DTC_DIRECT))
        return NTSE_E_INVAL;
    char why[200];
    uint64_t pin_need = 0, dev_need = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (!items[i].x || !items[i].y) return NTSE_E_INVAL;
        int rc = ntse_dtc_validate(items[i].t, why, sizeof why);
        if (rc != NTSE_OK) {
            if (a->cfg.verbose)
                fprintf(stderr, "ntse-accel: stream item %u refused: %s\n", i, why);
            return rc;
        }
        if (cfg->op == NTSE_OP_DTC_ACTIVE && !act_encodable(items[i].t))
            return NTSE_E_UNSUPPORTED;
        ntse_layout L;
        layout_of(items[i].t, &L);
        uint64_t xb = (uint64_t)batch * items[i].t->cols * 4;
        uint64_t yb = (uint64_t)batch * items[i].t->rows * 4;
        uint64_t Ab = (uint64_t)batch * L.nseg * 4;
        uint64_t Tb = (uint64_t)batch * L.nseg * items[i].t->entries * 4;
        uint64_t p = al(al(L.total + xb) + yb);
        uint64_t d = al(al(p + Ab) + Tb);
        if (p > pin_need) pin_need = p;
        if (d > dev_need) dev_need = d;
    }
    st->items = n;
    st->slot_bytes = pin_need;
    double t0 = now_ms();
    if (!on_gpu(a)) {
        int rc = stream_cpu(items, n, batch, cfg->op);
        st->wall_ms = now_ms() - t0;
        return rc;
    }

    stream_ctx *c = (stream_ctx *)calloc(1, sizeof *c);
    if (!c) return NTSE_E_NOMEM;
    c->a = a; c->items = items; c->n = n; c->batch = batch;
    c->S = cfg->slots; c->op = cfg->op;
    int rc = NTSE_OK;
    /* pinned staging is bounded too: 1 GiB across all slots */
    if (pin_need > (1ull << 30) / c->S) rc = NTSE_E_BUDGET;
    for (uint32_t s = 0; s < c->S && rc == NTSE_OK; s++) {
        rc = a->ops->halloc(a->be, pin_need, (void **)&c->pin[s], &c->tok[s]);
        if (rc == NTSE_OK) rc = charge_alloc(a, dev_need, &c->dev[s]);
    }
    int threaded = 0;
#ifdef _WIN32
    HANDLE th = NULL;
#else
    pthread_t th;
#endif
    if (rc == NTSE_OK && cfg->stager_thread) {
#ifdef _WIN32
        th = CreateThread(NULL, 0, stager_main, c, 0, NULL);
        threaded = th != NULL;
#else
        threaded = pthread_create(&th, NULL, stager_main, c) == 0;
#endif
        if (!threaded && a->cfg.verbose)
            fprintf(stderr, "ntse-accel: stager thread did not start; "
                    "staging inline\n");
    }
    uint32_t retired = 0;
    for (uint32_t i = 0; i < n && rc == NTSE_OK; i++) {
        uint32_t s = i % c->S;
        if (threaded) {
            for (;;) {
                uint32_t seen = hs_door_open(&c->door);
                if (HS_LD(&c->state[s]) == SLOT_STAGED
                    && HS_LDX(&c->item[s]) == i) break;
                c->driver_waits++;
                hs_door_wait(&c->door, seen);
            }
        } else {
            if (!claim_for_staging(c, s)) { c->violations++; rc = NTSE_E_STATE; break; }
            stage_into(c, s, i);
            HS_ST(&c->state[s], SLOT_STAGED);
        }
        rc = driver_enqueue(c, s, i);
        if (rc != NTSE_OK) break;
        if (i + 1 >= c->S) {
            rc = driver_retire(c, retired, st);
            if (rc == NTSE_OK) retired++;
        }
    }
    while (rc == NTSE_OK && retired < n) {
        rc = driver_retire(c, retired, st);
        if (rc == NTSE_OK) retired++;
    }
    /* phase boundary: stop the stager and count it out */
    HS_ST(&c->stop, 1u);
    hs_door_bump(&c->door);
    if (threaded) {
        hs_wait_count(&c->exited, 1u);
#ifdef _WIN32
        WaitForSingleObject(th, INFINITE);
        CloseHandle(th);
#else
        pthread_join(th, NULL);
#endif
    }
    /* drain before any buffer is freed: nothing may be recycled
     * (here: released) while a queue may still own it */
    int r1 = a->ops->qsync(a->be, NTSE_Q_COPY);
    int r2 = a->ops->qsync(a->be, NTSE_Q_COMPUTE);
    if (rc == NTSE_OK && (r1 != NTSE_OK || r2 != NTSE_OK)) rc = NTSE_E_DEVICE;
    for (uint32_t s = 0; s < c->S; s++) {
        if (c->pin[s]) a->ops->hfree(a->be, c->pin[s], c->tok[s]);
        charge_free(a, &c->dev[s]);
    }
    st->stager_waits = c->stager_waits;
    st->driver_waits = c->driver_waits;
    st->ownership_violations = c->violations;
    st->h2d_bytes = 0;
    if (rc == NTSE_OK) {
        for (uint32_t i = 0; i < n; i++) {
            ntse_layout L;
            layout_of(items[i].t, &L);
            st->h2d_bytes += L.total + (uint64_t)batch * items[i].t->cols * 4;
        }
        st->lane = a->lane;
        st->wall_ms = now_ms() - t0;
        free(c);
        return NTSE_OK;
    }
    free(c);
    fault(a, rc, "stream");
    st->d2h_bytes = 0;
    if (!a->cfg.allow_fallback || rc == NTSE_E_INVAL) return rc;
    if (a->cfg.verbose)
        fprintf(stderr, "ntse-accel: stream on %s failed (%s); all items "
                "recomputed on the CPU reference\n", ntse_lane_name(a->lane),
                ntse_accel_strerror(rc));
    rc = stream_cpu(items, n, batch, cfg->op);
    st->wall_ms = now_ms() - t0;
    return rc;
}
#include "accel_plan.inc.h"
