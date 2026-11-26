/* accel_opencl.c - the OpenCL lanes (research, scratch).
 *
 * Direction: JD (integrated-graphics lane, and a default lane for
 * NVIDIA cards without CUDA). Lane proposals: Gizmo. Co-implemented
 * by Cortana on Elfie's NTSE.
 *
 * One backend serves two lanes:
 *   NTSE_LANE_CL_IGPU  an integrated GPU (host-unified memory),
 *                      e.g. Intel UHD on the CPU package;
 *   NTSE_LANE_CL_GPU   a discrete GPU through the vendor's OpenCL
 *                      driver: the default lane wherever CUDA is not
 *                      available (no toolkit, no NVRTC, a driver
 *                      whose CUDA does not support the card).
 *
 * Why OpenCL and not Vulkan compute for these lanes: OpenCL kernels
 * are compiled from source by the installed driver at run time, so
 * the lane needs nothing beyond the driver (OpenCL.dll ships with
 * the Intel and NVIDIA display drivers). Vulkan compute needs SPIR-V
 * produced by an offline shader compiler (glslc / glslangValidator
 * from a Vulkan SDK), which this workstation does not have and which
 * this task may not download; hand-written SPIR-V would not meet the
 * robustness bar. The backend interface is lane-neutral, so a Vulkan
 * backend can be added later beside this one without touching the
 * core.
 *
 * OpenCL.dll / libOpenCL.so.1 is loaded at run time; no OpenCL
 * header or import library is needed to build.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "accel_internal.h"
#include "accel_kernels_cl.h"

typedef int cl_int;
typedef unsigned int cl_uint;
typedef unsigned long long cl_ulong;
typedef cl_ulong cl_bitfield;
typedef cl_bitfield cl_device_type;
typedef cl_bitfield cl_mem_flags;
typedef cl_bitfield cl_map_flags;
typedef cl_bitfield cl_command_queue_properties;
typedef cl_uint cl_bool;
typedef struct _cl_platform_id *cl_platform_id;
typedef struct _cl_device_id *cl_device_id;
typedef struct _cl_context *cl_context;
typedef struct _cl_command_queue *cl_command_queue;
typedef struct _cl_mem *cl_mem;
typedef struct _cl_program *cl_program;
typedef struct _cl_kernel *cl_kernel;
typedef struct _cl_event *cl_event;
typedef intptr_t cl_context_properties;

#define CL_SUCCESS 0
#define CL_COMPLETE 0
#define CL_DEVICE_TYPE_GPU (1 << 2)
#define CL_PLATFORM_NAME 0x0902
#define CL_DEVICE_VENDOR_ID 0x1001
#define CL_DEVICE_MAX_COMPUTE_UNITS 0x1002
#define CL_DEVICE_MAX_WORK_GROUP_SIZE 0x1004
#define CL_DEVICE_MAX_MEM_ALLOC_SIZE 0x1010
#define CL_DEVICE_SINGLE_FP_CONFIG 0x101B
#define CL_DEVICE_GLOBAL_MEM_SIZE 0x101F
#define CL_DEVICE_NAME 0x102B
#define CL_DRIVER_VERSION 0x102D
#define CL_DEVICE_VERSION 0x102F
#define CL_DEVICE_EXTENSIONS 0x1030
#define CL_DEVICE_HOST_UNIFIED_MEMORY 0x1035
#define CL_FP_DENORM (1 << 0)
#define CL_MEM_READ_WRITE (1 << 0)
#define CL_MEM_ALLOC_HOST_PTR (1 << 4)
#define CL_MAP_READ (1 << 0)
#define CL_MAP_WRITE (1 << 1)
#define CL_PROGRAM_BUILD_LOG 0x1183
#define CL_KERNEL_WORK_GROUP_SIZE 0x11B0
#define CL_EVENT_COMMAND_EXECUTION_STATUS 0x11D3

typedef struct {
    cl_int (*GetPlatformIDs)(cl_uint, cl_platform_id *, cl_uint *);
    cl_int (*GetPlatformInfo)(cl_platform_id, cl_uint, size_t, void *, size_t *);
    cl_int (*GetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
    cl_int (*GetDeviceInfo)(cl_device_id, cl_uint, size_t, void *, size_t *);
    cl_context (*CreateContext)(const cl_context_properties *, cl_uint, const cl_device_id *,
                                void *, void *, cl_int *);
    cl_int (*ReleaseContext)(cl_context);
    cl_command_queue (*CreateCommandQueue)(cl_context, cl_device_id,
                                           cl_command_queue_properties, cl_int *);
    cl_int (*ReleaseCommandQueue)(cl_command_queue);
    cl_mem (*CreateBuffer)(cl_context, cl_mem_flags, size_t, void *, cl_int *);
    cl_int (*ReleaseMemObject)(cl_mem);
    void *(*EnqueueMapBuffer)(cl_command_queue, cl_mem, cl_bool, cl_map_flags, size_t,
                              size_t, cl_uint, const cl_event *, cl_event *, cl_int *);
    cl_int (*EnqueueUnmapMemObject)(cl_command_queue, cl_mem, void *, cl_uint,
                                    const cl_event *, cl_event *);
    cl_int (*EnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t,
                                 const void *, cl_uint, const cl_event *, cl_event *);
    cl_int (*EnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t,
                                void *, cl_uint, const cl_event *, cl_event *);
    cl_int (*EnqueueFillBuffer)(cl_command_queue, cl_mem, const void *, size_t, size_t,
                                size_t, cl_uint, const cl_event *, cl_event *);
    cl_program (*CreateProgramWithSource)(cl_context, cl_uint, const char **,
                                          const size_t *, cl_int *);
    cl_int (*BuildProgram)(cl_program, cl_uint, const cl_device_id *, const char *,
                           void *, void *);
    cl_int (*GetProgramBuildInfo)(cl_program, cl_device_id, cl_uint, size_t, void *,
                                  size_t *);
    cl_int (*ReleaseProgram)(cl_program);
    cl_kernel (*CreateKernel)(cl_program, const char *, cl_int *);
    cl_int (*ReleaseKernel)(cl_kernel);
    cl_int (*GetKernelWorkGroupInfo)(cl_kernel, cl_device_id, cl_uint, size_t, void *,
                                     size_t *);
    cl_int (*SetKernelArg)(cl_kernel, cl_uint, size_t, const void *);
    cl_int (*EnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t *,
                                   const size_t *, const size_t *, cl_uint,
                                   const cl_event *, cl_event *);
    cl_int (*EnqueueMarkerWithWaitList)(cl_command_queue, cl_uint, const cl_event *,
                                        cl_event *);
    cl_int (*EnqueueBarrierWithWaitList)(cl_command_queue, cl_uint, const cl_event *,
                                         cl_event *);
    cl_int (*WaitForEvents)(cl_uint, const cl_event *);
    cl_int (*GetEventInfo)(cl_event, cl_uint, size_t, void *, size_t *);
    cl_int (*ReleaseEvent)(cl_event);
    cl_int (*Flush)(cl_command_queue);
    cl_int (*Finish)(cl_command_queue);
} cl_api;

static const char *const kernel_names[NTSE_K_COUNT] = {
    "k_segsum", "k_build_eager", "k_build_active", "k_apply",
    "k_apply_direct", "k_expand", "k_gemv", "k_linear_bf16",
    "k_mark", "k_build_lazy"
};

struct ntse_be {
    void *lib;
    cl_api cl;
    cl_device_id dev;
    cl_context ctx;
    cl_command_queue q[NTSE_NQUEUES];
    cl_event ev[NTSE_MAX_EVENTS];
    cl_program prog;
    cl_kernel k[NTSE_K_COUNT];
    size_t rw;          /* reduction work-group width */
    cl_uint cus;
    cl_int last;
};

static int ld(void *lib, const char *n, void *slot)
{
    void *p = ntse_dl_sym(lib, n);
    if (!p) return -1;
    memcpy(slot, &p, sizeof p);
    return 0;
}

static int load_cl(struct ntse_be *b, char *why, size_t wl)
{
#ifdef _WIN32
    b->lib = ntse_dl_open("OpenCL.dll");
#else
    b->lib = ntse_dl_open("libOpenCL.so.1");
#endif
    if (!b->lib) { snprintf(why, wl, "OpenCL ICD loader not found"); return -1; }
    struct { const char *n; void *slot; } t[] = {
        { "clGetPlatformIDs", &b->cl.GetPlatformIDs },
        { "clGetPlatformInfo", &b->cl.GetPlatformInfo },
        { "clGetDeviceIDs", &b->cl.GetDeviceIDs },
        { "clGetDeviceInfo", &b->cl.GetDeviceInfo },
        { "clCreateContext", &b->cl.CreateContext },
        { "clReleaseContext", &b->cl.ReleaseContext },
        { "clCreateCommandQueue", &b->cl.CreateCommandQueue },
        { "clReleaseCommandQueue", &b->cl.ReleaseCommandQueue },
        { "clCreateBuffer", &b->cl.CreateBuffer },
        { "clReleaseMemObject", &b->cl.ReleaseMemObject },
        { "clEnqueueMapBuffer", &b->cl.EnqueueMapBuffer },
        { "clEnqueueUnmapMemObject", &b->cl.EnqueueUnmapMemObject },
        { "clEnqueueWriteBuffer", &b->cl.EnqueueWriteBuffer },
        { "clEnqueueReadBuffer", &b->cl.EnqueueReadBuffer },
        { "clEnqueueFillBuffer", &b->cl.EnqueueFillBuffer },
        { "clCreateProgramWithSource", &b->cl.CreateProgramWithSource },
        { "clBuildProgram", &b->cl.BuildProgram },
        { "clGetProgramBuildInfo", &b->cl.GetProgramBuildInfo },
        { "clReleaseProgram", &b->cl.ReleaseProgram },
        { "clCreateKernel", &b->cl.CreateKernel },
        { "clReleaseKernel", &b->cl.ReleaseKernel },
        { "clGetKernelWorkGroupInfo", &b->cl.GetKernelWorkGroupInfo },
        { "clSetKernelArg", &b->cl.SetKernelArg },
        { "clEnqueueNDRangeKernel", &b->cl.EnqueueNDRangeKernel },
        { "clEnqueueMarkerWithWaitList", &b->cl.EnqueueMarkerWithWaitList },
        { "clEnqueueBarrierWithWaitList", &b->cl.EnqueueBarrierWithWaitList },
        { "clWaitForEvents", &b->cl.WaitForEvents },
        { "clGetEventInfo", &b->cl.GetEventInfo },
        { "clReleaseEvent", &b->cl.ReleaseEvent },
        { "clFlush", &b->cl.Flush },
        { "clFinish", &b->cl.Finish },
    };
    for (size_t i = 0; i < sizeof t / sizeof t[0]; i++)
        if (ld(b->lib, t[i].n, t[i].slot) != 0) {
            snprintf(why, wl, "OpenCL loader lacks %s (needs OpenCL 1.2)", t[i].n);
            return -1;
        }
    return 0;
}

static int ck(struct ntse_be *b, cl_int r)
{
    b->last = r;
    return r == CL_SUCCESS ? NTSE_OK : NTSE_E_DEVICE;
}

static void cl_destroy(ntse_be *b)
{
    if (!b) return;
    for (int i = 0; i < NTSE_NQUEUES; i++)
        if (b->q[i]) b->cl.Finish(b->q[i]);
    for (int i = 0; i < NTSE_MAX_EVENTS; i++)
        if (b->ev[i]) b->cl.ReleaseEvent(b->ev[i]);
    for (int i = 0; i < NTSE_K_COUNT; i++)
        if (b->k[i]) b->cl.ReleaseKernel(b->k[i]);
    if (b->prog) b->cl.ReleaseProgram(b->prog);
    for (int i = 0; i < NTSE_NQUEUES; i++)
        if (b->q[i]) b->cl.ReleaseCommandQueue(b->q[i]);
    if (b->ctx) b->cl.ReleaseContext(b->ctx);
    ntse_dl_close(b->lib);
    free(b);
}

static int qok(int q) { return q >= 0 && q < NTSE_NQUEUES; }
static int evok(int e) { return e >= 0 && e < NTSE_MAX_EVENTS; }

static int cl_dalloc(ntse_be *b, uint64_t bytes, ntse_dmem *m)
{
    cl_int e = 0;
    cl_mem o = b->cl.CreateBuffer(b->ctx, CL_MEM_READ_WRITE, (size_t)bytes, NULL, &e);
    if (!o || e != CL_SUCCESS) {
        b->last = e;
        /* CL_MEM_OBJECT_ALLOCATION_FAILURE -4, CL_OUT_OF_RESOURCES -5 */
        return (e == -4 || e == -5 || e == -61) ? NTSE_E_BUDGET : NTSE_E_DEVICE;
    }
    m->obj = o; m->ptr = 0; m->bytes = bytes;
    return NTSE_OK;
}

static void cl_dfree(ntse_be *b, ntse_dmem *m)
{
    if (m->obj) b->cl.ReleaseMemObject((cl_mem)m->obj);
    m->obj = NULL;
}

/* Pinned staging: an ALLOC_HOST_PTR buffer, mapped once for its
 * whole life; the mapped pointer is the host staging area. */
static int cl_halloc(ntse_be *b, uint64_t bytes, void **host, void **tok)
{
    cl_int e = 0;
    cl_mem o = b->cl.CreateBuffer(b->ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                  (size_t)bytes, NULL, &e);
    if (!o || e != CL_SUCCESS) { b->last = e; return NTSE_E_DEVICE; }
    void *p = b->cl.EnqueueMapBuffer(b->q[NTSE_Q_COPY], o, 1, CL_MAP_READ | CL_MAP_WRITE,
                                     0, (size_t)bytes, 0, NULL, NULL, &e);
    if (!p || e != CL_SUCCESS) {
        b->last = e;
        b->cl.ReleaseMemObject(o);
        return NTSE_E_DEVICE;
    }
    *host = p;
    *tok = o;
    return NTSE_OK;
}

static void cl_hfree(ntse_be *b, void *host, void *tok)
{
    if (!tok) return;
    if (host) {
        b->cl.EnqueueUnmapMemObject(b->q[NTSE_Q_COPY], (cl_mem)tok, host, 0, NULL, NULL);
        b->cl.Finish(b->q[NTSE_Q_COPY]);
    }
    b->cl.ReleaseMemObject((cl_mem)tok);
}

static int cl_put(ntse_be *b, int q, const ntse_dmem *dst, uint64_t off,
                  const void *src, uint64_t n, void *tok)
{
    if (!qok(q) || off > dst->bytes || n > dst->bytes - off) return NTSE_E_INVAL;
    if (n == 0) return NTSE_OK;
    cl_bool blocking = tok ? 0 : 1;
    if (ck(b, b->cl.EnqueueWriteBuffer(b->q[q], (cl_mem)dst->obj, blocking,
                                       (size_t)off, (size_t)n, src, 0, NULL, NULL)) != NTSE_OK)
        return NTSE_E_DEVICE;
    return ck(b, b->cl.Flush(b->q[q]));
}

static int cl_get(ntse_be *b, int q, void *dst, const ntse_dmem *src,
                  uint64_t off, uint64_t n, void *tok)
{
    if (!qok(q) || off > src->bytes || n > src->bytes - off) return NTSE_E_INVAL;
    if (n == 0) return NTSE_OK;
    cl_bool blocking = tok ? 0 : 1;
    if (ck(b, b->cl.EnqueueReadBuffer(b->q[q], (cl_mem)src->obj, blocking,
                                      (size_t)off, (size_t)n, dst, 0, NULL, NULL)) != NTSE_OK)
        return NTSE_E_DEVICE;
    return ck(b, b->cl.Flush(b->q[q]));
}

static int cl_zero(ntse_be *b, int q, const ntse_dmem *m)
{
    if (!qok(q)) return NTSE_E_INVAL;
    unsigned char z = 0;
    return ck(b, b->cl.EnqueueFillBuffer(b->q[q], (cl_mem)m->obj, &z, 1, 0,
                                         (size_t)m->bytes, 0, NULL, NULL));
}

static int cl_ev_record(ntse_be *b, int q, int e)
{
    if (!qok(q) || !evok(e)) return NTSE_E_INVAL;
    cl_event ne = NULL;
    if (ck(b, b->cl.EnqueueMarkerWithWaitList(b->q[q], 0, NULL, &ne)) != NTSE_OK)
        return NTSE_E_DEVICE;
    if (b->ev[e]) b->cl.ReleaseEvent(b->ev[e]);
    b->ev[e] = ne;
    return ck(b, b->cl.Flush(b->q[q]));
}

static int cl_ev_wait(ntse_be *b, int q, int e)
{
    if (!qok(q) || !evok(e) || !b->ev[e]) return NTSE_E_INVAL;
    return ck(b, b->cl.EnqueueBarrierWithWaitList(b->q[q], 1, &b->ev[e], NULL));
}

static int cl_ev_sync(ntse_be *b, int e)
{
    if (!evok(e) || !b->ev[e]) return NTSE_E_INVAL;
    return ck(b, b->cl.WaitForEvents(1, &b->ev[e]));
}

static int cl_ev_query(ntse_be *b, int e)
{
    if (!evok(e) || !b->ev[e]) return NTSE_E_INVAL;
    cl_int st = 1;
    if (ck(b, b->cl.GetEventInfo(b->ev[e], CL_EVENT_COMMAND_EXECUTION_STATUS,
                                 sizeof st, &st, NULL)) != NTSE_OK)
        return NTSE_E_DEVICE;
    if (st < 0) { b->last = st; return NTSE_E_DEVICE; }  /* command failed */
    return st == CL_COMPLETE ? 1 : 0;
}

static int cl_qsync(ntse_be *b, int q)
{
    if (!qok(q)) return NTSE_E_INVAL;
    return ck(b, b->cl.Finish(b->q[q]));
}

/* Argument helpers: a buffer argument is (cl_mem, element offset). */
typedef struct { struct ntse_be *b; cl_kernel k; cl_uint i; cl_int err; } argset;

static void a_raw(argset *s, size_t sz, const void *v)
{
    if (s->err) return;
    s->err = s->b->cl.SetKernelArg(s->k, s->i++, sz, v);
}
static void a_buf(argset *s, const ntse_dref *r, uint64_t esz)
{
    cl_mem m = r->m ? (cl_mem)r->m->obj : NULL;
    cl_ulong off = r->m ? r->off / esz : 0;
    if (r->m && r->off % esz) { s->err = -51; return; }
    a_raw(s, sizeof m, &m);
    a_raw(s, sizeof off, &off);
}
static void a_u64(argset *s, uint64_t v) { cl_ulong x = v; a_raw(s, sizeof x, &x); }
static void a_u32(argset *s, uint32_t v) { cl_uint x = v; a_raw(s, sizeof x, &x); }

static int run_flat(struct ntse_be *b, int q, cl_kernel k, uint64_t n)
{
    if (n == 0) return NTSE_OK;
    size_t local = 64;
    uint64_t g = (n + local - 1) / local * local;
    uint64_t cap = (uint64_t)local * 65536;        /* grid-stride covers the rest */
    if (g > cap) g = cap;
    size_t gs = (size_t)g;
    return ck(b, b->cl.EnqueueNDRangeKernel(b->q[q], k, 1, NULL, &gs, &local, 0, NULL, NULL));
}

static int run_groups(struct ntse_be *b, int q, cl_kernel k, uint64_t groups)
{
    if (groups == 0) return NTSE_OK;
    if (groups > 65535) groups = 65535;
    size_t local = b->rw, gs = (size_t)groups * b->rw;
    return ck(b, b->cl.EnqueueNDRangeKernel(b->q[q], k, 1, NULL, &gs, &local, 0, NULL, NULL));
}

static int cl_launch(ntse_be *b, int q, int kernel, const ntse_kargs *k)
{
    if (!qok(q) || kernel < 0 || kernel >= NTSE_K_COUNT
        || kernel == NTSE_K_LINEAR_BF16)
        return NTSE_E_INVAL;
    argset s = { b, b->k[kernel], 0, 0 };
    int rc;
    switch (kernel) {
    case NTSE_K_SEGSUM:
        a_buf(&s, &k->x, 4); a_buf(&s, &k->A, 4);
        a_u64(&s, k->nseg); a_u32(&s, k->block); a_u32(&s, k->batch); a_u64(&s, k->cols);
        if (s.err) return ck(b, s.err);
        rc = run_flat(b, q, s.k, (uint64_t)k->batch * k->nseg);
        break;
    case NTSE_K_BUILD_EAGER:
        a_buf(&s, &k->bookT, 4); a_buf(&s, &k->x, 4); a_buf(&s, &k->T, 4);
        a_u64(&s, k->nseg); a_u32(&s, k->block); a_u32(&s, k->entries);
        a_u32(&s, k->batch); a_u64(&s, k->cols);
        if (s.err) return ck(b, s.err);
        rc = run_flat(b, q, s.k, (uint64_t)k->batch * k->nseg * k->entries);
        break;
    case NTSE_K_BUILD_ACTIVE:
        a_buf(&s, &k->bookT, 4); a_buf(&s, &k->x, 4); a_buf(&s, &k->T, 4);
        a_buf(&s, &k->act, 4); a_u64(&s, k->nact);
        a_u64(&s, k->nseg); a_u32(&s, k->block); a_u32(&s, k->entries);
        a_u32(&s, k->batch); a_u64(&s, k->cols);
        if (s.err) return ck(b, s.err);
        rc = run_flat(b, q, s.k, (uint64_t)k->batch * k->nact);
        break;
    case NTSE_K_MARK:
        a_buf(&s, &k->idxT, 1); a_buf(&s, &k->F, 4);
        a_u64(&s, k->rows); a_u64(&s, k->nseg); a_u32(&s, k->entries);
        if (s.err) return ck(b, s.err);
        rc = run_flat(b, q, s.k, k->rows * k->nseg);
        break;
    case NTSE_K_BUILD_LAZY:
        a_buf(&s, &k->bookT, 4); a_buf(&s, &k->x, 4); a_buf(&s, &k->T, 4);
        a_buf(&s, &k->F, 4);
        a_u64(&s, k->nseg); a_u32(&s, k->block); a_u32(&s, k->entries);
        a_u32(&s, k->batch); a_u64(&s, k->cols);
        if (s.err) return ck(b, s.err);
        rc = run_flat(b, q, s.k, (uint64_t)k->batch * k->nseg * k->entries);
        break;
    case NTSE_K_APPLY:
        a_buf(&s, &k->idxT, 1); a_buf(&s, &k->mnT, 2); a_buf(&s, &k->scT, 2);
        a_buf(&s, &k->T, 4); a_buf(&s, &k->A, 4); a_buf(&s, &k->y, 4);
        a_u64(&s, k->rows); a_u64(&s, k->nseg); a_u32(&s, k->entries); a_u32(&s, k->batch);
        if (s.err) return ck(b, s.err);
        rc = run_flat(b, q, s.k, (uint64_t)k->batch * k->rows);
        break;
    case NTSE_K_APPLY_DIRECT:
        a_buf(&s, &k->book, 4); a_buf(&s, &k->x, 4); a_buf(&s, &k->idxT, 1);
        a_buf(&s, &k->mnT, 2); a_buf(&s, &k->scT, 2); a_buf(&s, &k->A, 4);
        a_buf(&s, &k->y, 4);
        a_u64(&s, k->rows); a_u64(&s, k->nseg); a_u32(&s, k->block);
        a_u32(&s, k->batch); a_u64(&s, k->cols);
        if (s.err) return ck(b, s.err);
        rc = run_flat(b, q, s.k, (uint64_t)k->batch * k->rows);
        break;
    case NTSE_K_EXPAND:
        a_buf(&s, &k->book, 4); a_buf(&s, &k->idxT, 1); a_buf(&s, &k->mnT, 2);
        a_buf(&s, &k->scT, 2); a_buf(&s, &k->W, 4);
        a_u64(&s, k->rows); a_u64(&s, k->cols); a_u32(&s, k->block);
        if (s.err) return ck(b, s.err);
        rc = run_flat(b, q, s.k, k->rows * k->cols);
        break;
    case NTSE_K_GEMV:
        a_buf(&s, &k->W, 4); a_buf(&s, &k->x, 4); a_buf(&s, &k->y, 4);
        a_u64(&s, k->rows); a_u64(&s, k->cols); a_u32(&s, k->batch);
        if (s.err) return ck(b, s.err);
        rc = run_groups(b, q, s.k, (uint64_t)k->batch * k->rows);
        break;
    default:
        return NTSE_E_INVAL;
    }
    if (rc != NTSE_OK) return rc;
    return ck(b, b->cl.Flush(b->q[q]));
}

static int cl_linear(ntse_be *b, int q, ntse_dref W, ntse_dref x, ntse_dref y,
                     uint32_t out, uint32_t in)
{
    if (!qok(q) || out == 0 || in == 0) return NTSE_E_INVAL;
    argset s = { b, b->k[NTSE_K_LINEAR_BF16], 0, 0 };
    a_buf(&s, &W, 2); a_buf(&s, &x, 4); a_buf(&s, &y, 4);
    a_u32(&s, out); a_u32(&s, in);
    if (s.err) return ck(b, s.err);
    int rc = run_groups(b, q, s.k, out);
    if (rc != NTSE_OK) return rc;
    return ck(b, b->cl.Flush(b->q[q]));
}

static const ntse_be_ops cl_ops = {
    "opencl", cl_destroy, cl_dalloc, cl_dfree, cl_halloc, cl_hfree, cl_put,
    cl_get, cl_zero, cl_ev_record, cl_ev_wait, cl_ev_sync, cl_ev_query,
    cl_qsync, cl_launch, cl_linear
};

static void dev_str(struct ntse_be *b, cl_device_id d, cl_uint what, char *buf, size_t n)
{
    buf[0] = '\0';
    size_t got = 0;
    if (b->cl.GetDeviceInfo(d, what, n - 1, buf, &got) != CL_SUCCESS) buf[0] = '\0';
    buf[n - 1] = '\0';
}

/* Pick the device for a lane: integrated GPUs for CL_IGPU, discrete
 * GPUs for CL_GPU (NVIDIA first, then any). NTSE_CL_DEVICE=<substr>
 * narrows the choice by device name. */
static int pick(struct ntse_be *b, ntse_lane want, cl_device_id *out, char *why, size_t wl)
{
    cl_platform_id plats[16];
    cl_uint np = 0;
    if (b->cl.GetPlatformIDs(16, plats, &np) != CL_SUCCESS || np == 0) {
        snprintf(why, wl, "no OpenCL platform");
        return -1;
    }
    if (np > 16) np = 16;
    const char *filt = getenv("NTSE_CL_DEVICE");
    cl_device_id best = NULL;
    int best_score = -1;
    for (cl_uint p = 0; p < np; p++) {
        cl_device_id devs[16];
        cl_uint nd = 0;
        if (b->cl.GetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, 16, devs, &nd) != CL_SUCCESS)
            continue;
        if (nd > 16) nd = 16;
        for (cl_uint i = 0; i < nd; i++) {
            cl_bool uni = 0;
            cl_uint vendor = 0;
            char name[256], ver[128];
            if (b->cl.GetDeviceInfo(devs[i], CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof uni, &uni, NULL) != CL_SUCCESS
                || b->cl.GetDeviceInfo(devs[i], CL_DEVICE_VENDOR_ID, sizeof vendor, &vendor, NULL) != CL_SUCCESS)
                continue;
            dev_str(b, devs[i], CL_DEVICE_NAME, name, sizeof name);
            dev_str(b, devs[i], CL_DEVICE_VERSION, ver, sizeof ver);
            /* "OpenCL X.Y ..." : need 1.2 or newer */
            int maj = 0, min = 0;
            if (sscanf(ver, "OpenCL %d.%d", &maj, &min) != 2
                || (maj == 1 && min < 2) || maj < 1)
                continue;
            if (filt && *filt && !strstr(name, filt)) continue;
            int score = -1;
            if (want == NTSE_LANE_CL_IGPU && uni)
                score = vendor == 0x8086 ? 2 : 1;
            if (want == NTSE_LANE_CL_GPU && !uni)
                score = vendor == 0x10DE ? 2 : 1;
            if (score > best_score) { best_score = score; best = devs[i]; }
        }
    }
    if (!best) {
        snprintf(why, wl, "no %s OpenCL 1.2+ GPU device",
                 want == NTSE_LANE_CL_IGPU ? "integrated" : "discrete");
        return -1;
    }
    *out = best;
    return 0;
}

int ntse_cl_create(ntse_lane want, ntse_be **out, const ntse_be_ops **ops,
                   ntse_be_info *info, char *why, size_t wl)
{
    *out = NULL;
    if (want != NTSE_LANE_CL_IGPU && want != NTSE_LANE_CL_GPU) {
        snprintf(why, wl, "not an OpenCL lane");
        return NTSE_E_INVAL;
    }
    struct ntse_be *b = (struct ntse_be *)calloc(1, sizeof *b);
    if (!b) { snprintf(why, wl, "out of host memory"); return NTSE_E_NOMEM; }
    if (load_cl(b, why, wl) != 0) goto nodev;
    if (pick(b, want, &b->dev, why, wl) != 0) goto nodev;

    cl_ulong gmem = 0, maxalloc = 0, fpc = 0;
    size_t maxwg = 0;
    cl_bool uni = 0;
    char name[256], drv[128], ver[128], ext[4096];
    if (b->cl.GetDeviceInfo(b->dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof gmem, &gmem, NULL) != CL_SUCCESS
        || b->cl.GetDeviceInfo(b->dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof maxalloc, &maxalloc, NULL) != CL_SUCCESS
        || b->cl.GetDeviceInfo(b->dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof maxwg, &maxwg, NULL) != CL_SUCCESS
        || b->cl.GetDeviceInfo(b->dev, CL_DEVICE_SINGLE_FP_CONFIG, sizeof fpc, &fpc, NULL) != CL_SUCCESS
        || b->cl.GetDeviceInfo(b->dev, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof uni, &uni, NULL) != CL_SUCCESS
        || b->cl.GetDeviceInfo(b->dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof b->cus, &b->cus, NULL) != CL_SUCCESS) {
        snprintf(why, wl, "OpenCL device query failed");
        goto nodev;
    }
    dev_str(b, b->dev, CL_DEVICE_NAME, name, sizeof name);
    dev_str(b, b->dev, CL_DRIVER_VERSION, drv, sizeof drv);
    dev_str(b, b->dev, CL_DEVICE_VERSION, ver, sizeof ver);
    dev_str(b, b->dev, CL_DEVICE_EXTENSIONS, ext, sizeof ext);
    int fp64 = strstr(ext, "cl_khr_fp64") != NULL;
    /* reduction width: the largest power of two <= min(256, maxwg) */
    b->rw = 256;
    while (b->rw > maxwg && b->rw > 1) b->rw >>= 1;
    if (b->rw < 32) { snprintf(why, wl, "work-group limit %zu too small", maxwg); goto nodev; }

    cl_int e = 0;
    b->ctx = b->cl.CreateContext(NULL, 1, &b->dev, NULL, NULL, &e);
    if (!b->ctx || e != CL_SUCCESS) { snprintf(why, wl, "clCreateContext: %d", e); goto nodev; }
    for (int i = 0; i < NTSE_NQUEUES; i++) {
        b->q[i] = b->cl.CreateCommandQueue(b->ctx, b->dev, 0, &e);   /* in-order */
        if (!b->q[i] || e != CL_SUCCESS) { snprintf(why, wl, "clCreateCommandQueue: %d", e); goto nodev; }
    }
    const char *src = ntse_cl_src;
    b->prog = b->cl.CreateProgramWithSource(b->ctx, 1, &src, NULL, &e);
    if (!b->prog || e != CL_SUCCESS) { snprintf(why, wl, "clCreateProgramWithSource: %d", e); goto nodev; }
    char opts[128];
    snprintf(opts, sizeof opts, "-cl-std=CL1.2 -DNTSE_RW=%zu -DNTSE_HAVE_FP64=%d",
             b->rw, fp64);
    e = b->cl.BuildProgram(b->prog, 1, &b->dev, opts, NULL, NULL);
    if (e != CL_SUCCESS) {
        char log[400] = "";
        b->cl.GetProgramBuildInfo(b->prog, b->dev, CL_PROGRAM_BUILD_LOG, sizeof log - 1, log, NULL);
        log[sizeof log - 1] = '\0';
        snprintf(why, wl, "OpenCL build failed (%d): %.150s", e, log);
        goto nodev;
    }
    for (int i = 0; i < NTSE_K_COUNT; i++) {
        b->k[i] = b->cl.CreateKernel(b->prog, kernel_names[i], &e);
        if (!b->k[i] || e != CL_SUCCESS) {
            snprintf(why, wl, "clCreateKernel %s: %d", kernel_names[i], e);
            goto nodev;
        }
        size_t kwg = 0;
        if (b->cl.GetKernelWorkGroupInfo(b->k[i], b->dev, CL_KERNEL_WORK_GROUP_SIZE,
                                         sizeof kwg, &kwg, NULL) != CL_SUCCESS
            || kwg < 64 || ((i == NTSE_K_GEMV || i == NTSE_K_LINEAR_BF16) && kwg < b->rw)) {
            snprintf(why, wl, "kernel %s work-group limit %zu too small", kernel_names[i], kwg);
            goto nodev;
        }
    }
    snprintf(info->device, sizeof info->device,
             "%s (%s, driver %s, %u CUs, reduce width %zu, %s)",
             name, ver, drv, b->cus, b->rw,
             fp64 ? "fp64 accumulate" : "float-float accumulate");
    info->total_mem = gmem;
    info->free_mem = 0;            /* OpenCL does not report free memory */
    info->max_alloc = maxalloc;
    info->integrated = uni ? 1 : 0;
    info->has_fp64 = fp64;
    info->denorms = (fpc & CL_FP_DENORM) ? 1 : 0;
    *out = b;
    *ops = &cl_ops;
    return NTSE_OK;
nodev:
    cl_destroy(b);
    return NTSE_E_NODEV;
}
