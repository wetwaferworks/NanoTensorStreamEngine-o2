/* nemo_calib.c - see nemo_calib.h. Pure C, one thread, no locks. */
#define _POSIX_C_SOURCE 200809L
#include "nemo_calib.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#else
#include <dlfcn.h>
#include <sys/stat.h>
#endif

#define CALIB_VERSION 1

static double qpc_ms(void) {
#ifdef _WIN32
    LARGE_INTEGER f, t; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&t);
    return (double)t.QuadPart / ((double)f.QuadPart / 1000.0);
#else
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
#endif
}
static double now_ms_c(void) {
#ifdef _WIN32
    return (double)GetTickCount64();
#else
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
#endif
}

/* ---- OpenCL probe (no SDK: loaded at run time) ---- */
typedef int (*clGetPlatformIDs_t)(unsigned, void **, unsigned *);
typedef int (*clGetPlatformInfo_t)(void *, unsigned, size_t, void *, size_t *);
typedef int (*clGetDeviceIDs_t)(void *, unsigned long long, unsigned, void **, unsigned *);
typedef int (*clGetDeviceInfo_t)(void *, unsigned, size_t, void *, size_t *);

static void probe_opencl(nemo_calib *c) {
    snprintf(c->oclnv_status, sizeof c->oclnv_status, "rejected");
    snprintf(c->igpu_status, sizeof c->igpu_status, "rejected");
    snprintf(c->oclnv_why, sizeof c->oclnv_why, "no NVIDIA OpenCL device found");
    snprintf(c->igpu_why, sizeof c->igpu_why, "no integrated OpenCL GPU found");
#ifdef _WIN32
    HMODULE l = LoadLibraryA("OpenCL.dll");
    #define CLSYM(n) (void *)GetProcAddress(l, n)
#else
    void *l = dlopen("libOpenCL.so.1", RTLD_NOW | RTLD_LOCAL);
    #define CLSYM(n) dlsym(l, n)
#endif
    if (!l) { snprintf(c->oclnv_why, sizeof c->oclnv_why, "OpenCL runtime not installed");
              snprintf(c->igpu_why, sizeof c->igpu_why, "OpenCL runtime not installed"); return; }
    clGetPlatformIDs_t gp; clGetPlatformInfo_t gpi; clGetDeviceIDs_t gd; clGetDeviceInfo_t gdi;
    void *p0 = CLSYM("clGetPlatformIDs"), *p1 = CLSYM("clGetPlatformInfo"), *p2 = CLSYM("clGetDeviceIDs"), *p3 = CLSYM("clGetDeviceInfo");
    memcpy(&gp, &p0, sizeof gp); memcpy(&gpi, &p1, sizeof gpi); memcpy(&gd, &p2, sizeof gd); memcpy(&gdi, &p3, sizeof gdi);
    if (!p0 || !p1 || !p2 || !p3) { snprintf(c->oclnv_why, sizeof c->oclnv_why, "OpenCL runtime incomplete"); goto done; }
    void *plats[8]; unsigned np = 0;
    if (gp(8, plats, &np) != 0) np = 0;
    for (unsigned i = 0; i < np && i < 8; i++) {
        void *devs[8]; unsigned nd = 0;
        if (gd(plats[i], 4ull /* CL_DEVICE_TYPE_GPU */, 8, devs, &nd) != 0) continue;
        for (unsigned j = 0; j < nd && j < 8; j++) {
            char name[128] = "", vendor[128] = ""; unsigned um = 0; unsigned long long mem = 0;
            gdi(devs[j], 0x102B, sizeof name, name, NULL);
            gdi(devs[j], 0x102C, sizeof vendor, vendor, NULL);
            gdi(devs[j], 0x1035, sizeof um, &um, NULL);   /* CL_DEVICE_HOST_UNIFIED_MEMORY */
            gdi(devs[j], 0x101F, sizeof mem, &mem, NULL); /* CL_DEVICE_GLOBAL_MEM_SIZE */
            int nv = strstr(vendor, "NVIDIA") != NULL;
            char *dst = nv ? c->oclnv_dev : c->igpu_dev;
            char *why = nv ? c->oclnv_why : c->igpu_why;
            size_t wl = nv ? sizeof c->oclnv_why : sizeof c->igpu_why;
            snprintf(dst, 128, "%s", name);
            snprintf(why, wl, "found %s (%s, %.0f MiB%s); rejected: this build has no OpenCL Nemotron forward "
                     "(only the CUDA and CPU lanes have kernels to verify)", name, vendor, mem / 1048576.0,
                     um ? ", unified memory" : "");
        }
    }
done:
#ifdef _WIN32
    FreeLibrary(l);
#else
    dlclose(l);
#endif
}

/* ---- CPU micro-bench: single-thread streaming read ---- */
static double cpu_read_gbs(void) {
    size_t n = (size_t)128 << 20;
    unsigned long long *b = (unsigned long long *)malloc(n);
    if (!b) return 0.0;
    size_t m = n / 8;
    for (size_t i = 0; i < m; i++) b[i] = i * 0x9E3779B97F4A7C15ull;
    volatile unsigned long long sink = 0;
    double t0 = qpc_ms();
    for (int r = 0; r < 3; r++) { unsigned long long a = 0; for (size_t i = 0; i < m; i++) a += b[i]; sink += a; }
    double dt = qpc_ms() - t0;
    free(b);
    (void)sink;
    return dt > 0 ? 3.0 * n / (dt / 1000.0) / 1e9 : 0.0;
}

static void cache_path(nemo_calib *c) {
    const char *e = getenv("NEMO_CALIB_FILE");
    if (e && *e) { snprintf(c->path, sizeof c->path, "%s", e); return; }
#ifdef _WIN32
    const char *la = getenv("LOCALAPPDATA");
    char dir[400];
    snprintf(dir, sizeof dir, "%s\\NemoChat", la && *la ? la : ".");
    _mkdir(dir);
    snprintf(c->path, sizeof c->path, "%s\\nemo-calib.txt", dir);
#else
    const char *h = getenv("HOME");
    char dir[400];
    snprintf(dir, sizeof dir, "%s/.cache/nemochat", h && *h ? h : ".");
    mkdir(dir, 0755);
    snprintf(c->path, sizeof c->path, "%s/nemo-calib.txt", dir);
#endif
}

static void kv_get(const char *buf, const char *k, char *out, size_t ol) {
    char pat[64]; snprintf(pat, sizeof pat, "\n%s=", k);
    const char *p = strstr(buf, pat);
    out[0] = 0;
    if (!p) return;
    p += strlen(pat);
    size_t i = 0;
    while (p[i] && p[i] != '\n' && p[i] != '\r' && i + 1 < ol) { out[i] = p[i]; i++; }
    out[i] = 0;
}

static int cache_read(nemo_calib *c) {
    FILE *f = fopen(c->path, "rb");
    if (!f) return -1;
    char buf[8192]; size_t n = fread(buf + 1, 1, sizeof buf - 2, f); fclose(f);
    buf[0] = '\n'; buf[n + 1] = 0;
    char key[400], v[300];
    kv_get(buf, "key", key, sizeof key);
    if (strcmp(key, c->key)) return -1;
    kv_get(buf, "lane", c->lane, sizeof c->lane);
    if (strcmp(c->lane, "cuda") && strcmp(c->lane, "cpu")) return -1;
    kv_get(buf, "cuda_status", c->cuda_status, sizeof c->cuda_status);
    kv_get(buf, "cuda_why", c->cuda_why, sizeof c->cuda_why);
    kv_get(buf, "oclnv_status", c->oclnv_status, sizeof c->oclnv_status);
    kv_get(buf, "oclnv_why", c->oclnv_why, sizeof c->oclnv_why);
    kv_get(buf, "igpu_status", c->igpu_status, sizeof c->igpu_status);
    kv_get(buf, "igpu_why", c->igpu_why, sizeof c->igpu_why);
    kv_get(buf, "cpu_status", c->cpu_status, sizeof c->cpu_status);
    kv_get(buf, "cpu_why", c->cpu_why, sizeof c->cpu_why);
    kv_get(buf, "est_cuda_tps", v, sizeof v); c->est_cuda_tps = atof(v);
    kv_get(buf, "est_cpu_tps", v, sizeof v); c->est_cpu_tps = atof(v);
    kv_get(buf, "dtc_gwps", v, sizeof v); c->probe.dtc_gwps = atof(v);
    kv_get(buf, "full_gwps", v, sizeof v); c->probe.full_gwps = atof(v);
    kv_get(buf, "read_gbs", v, sizeof v); c->probe.read_gbs = atof(v);
    kv_get(buf, "device", c->probe.device, sizeof c->probe.device);
    c->cached = 1;
    return 0;
}

static void cache_write(const nemo_calib *c) {
    FILE *f = fopen(c->path, "wb");
    if (!f) return;
    time_t t = time(NULL);
    fprintf(f, "# Nemo engine lane calibration (delete or run with --recalibrate to redo)\n"
               "key=%s\nwritten=%lld\nlane=%s\ndevice=%s\ncuda_status=%s\ncuda_why=%s\n"
               "oclnv_status=%s\noclnv_why=%s\nigpu_status=%s\nigpu_why=%s\ncpu_status=%s\ncpu_why=%s\n"
               "est_cuda_tps=%.3f\nest_cpu_tps=%.4f\ndtc_gwps=%.3f\nfull_gwps=%.3f\nread_gbs=%.2f\n"
               "dtc_exact=%d\nfull_exact=%d\n",
            c->key, (long long)t, c->lane, c->probe.device, c->cuda_status, c->cuda_why,
            c->oclnv_status, c->oclnv_why, c->igpu_status, c->igpu_why, c->cpu_status, c->cpu_why,
            c->est_cuda_tps, c->est_cpu_tps, c->probe.dtc_gwps, c->probe.full_gwps, c->probe.read_gbs,
            c->probe.dtc_exact, c->probe.full_exact);
    fclose(f);
}

void nemo_calib_run(nemo_calib *c, const char *override, int recal, int have_pack) {
    memset(c, 0, sizeof *c);
    /* volatile: the start time lives in memory, not a callee-saved register.
     * A fresh run calls into the CUDA and OpenCL driver DLLs; with t0 kept in
     * a register the elapsed time came back as the whole system uptime
     * ("calibration fresh in 629628906 ms"), i.e. t0 read as 0. */
    volatile double t0 = now_ms_c();
    cache_path(c);
    char dev[128] = ""; int drv = 0, rm = 0, rn = 0;
    int have_cuda = nemo_gpu_identify(dev, sizeof dev, &drv, &rm, &rn) == 0;
    snprintf(c->key, sizeof c->key, "v%d|gpu=%s|driver=%d|nvrtc=%d.%d|pack=%d", CALIB_VERSION,
             have_cuda ? dev : "none", drv, rm, rn, have_pack);
    int forced_cuda = override && !strcmp(override, "cuda");
    int forced_cpu = override && !strcmp(override, "cpu");
    if (!recal && cache_read(c) == 0) {
        if (forced_cuda) snprintf(c->lane, sizeof c->lane, "cuda");
        if (forced_cpu) snprintf(c->lane, sizeof c->lane, "cpu");
        c->ms = now_ms_c() - t0;
        return;
    }
    /* CPU: always correct (reference implementation) */
    double gbs = cpu_read_gbs();
    c->est_cpu_tps = gbs > 0 ? gbs / 17.9 : 0.0; /* raw bf16 bytes per token */
    snprintf(c->cpu_status, sizeof c->cpu_status, "pass");
    snprintf(c->cpu_why, sizeof c->cpu_why, "reference lane; %.1f GB/s single-thread read -> ~%.3f tok/s estimate", gbs, c->est_cpu_tps);
    /* CUDA */
    if (!have_cuda) {
        snprintf(c->cuda_status, sizeof c->cuda_status, "rejected");
        snprintf(c->cuda_why, sizeof c->cuda_why, "no CUDA driver or device");
    } else if (!have_pack) {
        snprintf(c->cuda_status, sizeof c->cuda_status, "rejected");
        snprintf(c->cuda_why, sizeof c->cuda_why, "no .nq2 pack given (--cuda-pack), nothing to run on the GPU");
    } else {
        char why[256] = "";
        int ok = nemo_gpu_selftest(&c->probe, why, sizeof why) == 0;
        /* Structural check only: a 4 GiB-class card is needed for the resident
         * pack. A momentary shortage of free VRAM is not cached as a verdict;
         * the open that follows reports it and falls back for that launch. */
        if (ok && c->probe.vram_total < (uint64_t)3500 << 20) {
            ok = 0;
            snprintf(why, sizeof why, "VRAM: %.0f MiB total, the resident pack needs about 2.8 GiB plus headroom",
                     c->probe.vram_total / 1048576.0);
        }
        c->est_cuda_tps = c->probe.est_tps;
        snprintf(c->cuda_status, sizeof c->cuda_status, ok ? "pass" : "rejected");
        if (ok)
            snprintf(c->cuda_why, sizeof c->cuda_why,
                     "%s sm_%d%d driver %d nvrtc %d.%d, %.0f MiB free; DTC and FULL kernels bit-exact vs C emulation; "
                     "DTC %.1f Gw/s, FULL %.1f Gw/s, read %.1f GB/s -> ~%.1f tok/s estimate",
                     c->probe.device, c->probe.cc_major, c->probe.cc_minor, c->probe.driver,
                     c->probe.nvrtc_major, c->probe.nvrtc_minor, c->probe.vram_free / 1048576.0,
                     c->probe.dtc_gwps, c->probe.full_gwps, c->probe.read_gbs, c->est_cuda_tps);
        else
            snprintf(c->cuda_why, sizeof c->cuda_why, "%s", why[0] ? why : "selftest failed");
    }
    probe_opencl(c);
    /* fastest correct lane; CUDA preferred, CPU only when nothing else passes */
    int cuda_ok = !strcmp(c->cuda_status, "pass");
    snprintf(c->lane, sizeof c->lane, cuda_ok && c->est_cuda_tps >= c->est_cpu_tps ? "cuda" : "cpu");
    if (cuda_ok && c->est_cuda_tps < c->est_cpu_tps)
        snprintf(c->cuda_why + strlen(c->cuda_why), sizeof c->cuda_why - strlen(c->cuda_why), "; slower than CPU estimate");
    cache_write(c);
    if (forced_cuda) snprintf(c->lane, sizeof c->lane, "cuda");
    if (forced_cpu) snprintf(c->lane, sizeof c->lane, "cpu");
    c->ms = now_ms_c() - t0;
}

void nemo_calib_log(const nemo_calib *c) {
    fprintf(stderr, "nemo-engine: calibration %s in %.0f ms (%s), lane=%s\n",
            c->cached ? "cached" : "fresh", c->ms, c->path, c->lane);
    fprintf(stderr, "nemo-engine:   cuda        %-8s %s\n", c->cuda_status, c->cuda_why);
    fprintf(stderr, "nemo-engine:   opencl-nv   %-8s %s\n", c->oclnv_status, c->oclnv_why);
    fprintf(stderr, "nemo-engine:   opencl-igpu %-8s %s\n", c->igpu_status, c->igpu_why);
    fprintf(stderr, "nemo-engine:   cpu         %-8s %s\n", c->cpu_status, c->cpu_why);
}
