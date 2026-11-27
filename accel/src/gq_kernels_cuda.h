/* gq_kernels_cuda.h - CUDA C source of the nq2-quantize assignment
 * kernels, compiled at run time by NVRTC (no nvcc at build time).
 * Co-implemented by Cortana on Elfie's quantizer.
 *
 * One thread owns one (row, subvector) pair and walks the codes
 * k = 0..entries-1 in order. Two passes:
 *
 *   1. fp32 screen. f_k = sum_i (x_i - c_k,i)^2 in fp32, every
 *      operation an explicit round-to-nearest intrinsic. A proven
 *      error bound turns each f_k into an interval [lo_k, hi_k]
 *      that contains the CPU's double sum D_k:
 *         hi_k = f_k * (1 + eps) + A   (rounded up)
 *         lo_k = f_k * (1 - eps) - A   (rounded down)
 *      with eps = (d + 8) * 2^-22 and A = (d + 8) * 2^-140, about
 *      four times the worst-case fp32 plus double rounding error
 *      of a d-term sum of squares, and A covering underflow.
 *      U = min_k hi_k bounds the smallest D_k from above.
 *
 *   2. exact rescore. Any k with lo_k <= U might be the winner;
 *      every other k has D_k > U >= min D and can never win. The
 *      candidates are rescored in double, in the CPU's order
 *      (__dsub_rn, __dmul_rn, __dadd_rn: no FMA), and compared in
 *      ascending k with a strict < from best = 0, best_err = 1e30,
 *      exactly as nearest_centroid() does. So the winner is the
 *      CPU's winner, ties and all.
 *
 * If any fp32 distance is not finite (NaN or overflow), the row
 * skips the screen and runs the full double search over every k.
 *
 * The template unrolls the fixed widths 1, 2, 4, 8 and 16; other
 * widths take the runtime loop. There are no preprocessor lines
 * inside the source string.
 *
 * NVRTC runs with --fmad=false and --ftz=false as well. No atomics:
 * each thread writes only its own output bytes.
 */
#ifndef GQ_KERNELS_CUDA_H
#define GQ_KERNELS_CUDA_H

#define GQ_CU_STR(...) #__VA_ARGS__

static const char gq_cuda_src[] = GQ_CU_STR(
typedef unsigned long long u64;
typedef unsigned int u32;

template <int D>
__device__ __forceinline__ float screen(const float *x, const float *c, u32 d)
{
    float f = 0.0f;
    const u32 n = D ? (u32)D : d;
    for (u32 i = 0; i < n; i++) {
        float t = __fsub_rn(x[i], c[i]);
        f = __fadd_rn(f, __fmul_rn(t, t));
    }
    return f;
}

__device__ __forceinline__ double exact(const float *x, const float *c, u32 d)
{
    double e = 0.0;
    for (u32 i = 0; i < d; i++) {
        double t = __dsub_rn((double)x[i], (double)c[i]);
        e = __dadd_rn(e, __dmul_rn(t, t));
    }
    return e;
}

template <int D>
__device__ __forceinline__ void assign_body(
    const float *__restrict__ X, u64 n, u32 stride, u32 nsub, u32 d,
    const float *__restrict__ books, u32 entries, int use_sh,
    float onep, float onem, float A,
    unsigned char *__restrict__ out, unsigned char *__restrict__ how)
{
    extern __shared__ float sh[];
    const u32 s = blockIdx.y;
    const float *gb = books + (u64)s * entries * d;
    if (use_sh) {
        for (u32 t = threadIdx.x; t < entries * d; t += blockDim.x) sh[t] = gb[t];
        __syncthreads();
    }
    const float *C = use_sh ? (const float *)sh : gb;
    const u64 b = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n) return;
    const float *xg = X + b * stride + (u64)s * d;
    float xr[D ? D : 1];
    const float *x = xg;
    if (D) {
        for (int i = 0; i < (D ? D : 1); i++) xr[i] = xg[i];
        x = xr;
    }
    const float INF = __int_as_float(0x7f800000);
    float U = INF;
    int bad = 0;
    for (u32 k = 0; k < entries; k++) {
        float f = screen<D>(x, C + (u64)k * d, d);
        if (!(f < INF)) { bad = 1; break; }          /* NaN or +inf */
        float hi = __fadd_ru(__fmul_ru(f, onep), A);
        if (hi < U) U = hi;
    }
    int best = 0;
    double best_err = 1e30;
    unsigned char kind;
    if (bad) {
        for (u32 k = 0; k < entries; k++) {
            double e = exact(xg, C + (u64)k * d, d);
            if (e < best_err) { best_err = e; best = (int)k; }
        }
        kind = 2;
    } else {
        u32 cand = 0;
        for (u32 k = 0; k < entries; k++) {
            float f = screen<D>(x, C + (u64)k * d, d);
            float lo = __fsub_rd(__fmul_rd(f, onem), A);
            if (lo <= U) {
                cand++;
                double e = exact(xg, C + (u64)k * d, d);
                if (e < best_err) { best_err = e; best = (int)k; }
            }
        }
        kind = cand > 1 ? 1 : 0;
    }
    out[b * nsub + s] = (unsigned char)best;
    how[b * nsub + s] = kind;
}

extern "C" __global__ void k_assign_any(const float *X, u64 n, u32 stride,
    u32 nsub, u32 d, const float *books, u32 entries, int use_sh,
    float onep, float onem, float A, unsigned char *out, unsigned char *how)
{
    assign_body<0>(X, n, stride, nsub, d, books, entries, use_sh,
                   onep, onem, A, out, how);
}

extern "C" __global__ void k_assign_d1(const float *X, u64 n, u32 stride,
    u32 nsub, u32 d, const float *books, u32 entries, int use_sh,
    float onep, float onem, float A, unsigned char *out, unsigned char *how)
{
    assign_body<1>(X, n, stride, nsub, d, books, entries, use_sh,
                   onep, onem, A, out, how);
}

extern "C" __global__ void k_assign_d2(const float *X, u64 n, u32 stride,
    u32 nsub, u32 d, const float *books, u32 entries, int use_sh,
    float onep, float onem, float A, unsigned char *out, unsigned char *how)
{
    assign_body<2>(X, n, stride, nsub, d, books, entries, use_sh,
                   onep, onem, A, out, how);
}

extern "C" __global__ void k_assign_d4(const float *X, u64 n, u32 stride,
    u32 nsub, u32 d, const float *books, u32 entries, int use_sh,
    float onep, float onem, float A, unsigned char *out, unsigned char *how)
{
    assign_body<4>(X, n, stride, nsub, d, books, entries, use_sh,
                   onep, onem, A, out, how);
}

extern "C" __global__ void k_assign_d8(const float *X, u64 n, u32 stride,
    u32 nsub, u32 d, const float *books, u32 entries, int use_sh,
    float onep, float onem, float A, unsigned char *out, unsigned char *how)
{
    assign_body<8>(X, n, stride, nsub, d, books, entries, use_sh,
                   onep, onem, A, out, how);
}

extern "C" __global__ void k_assign_d16(const float *X, u64 n, u32 stride,
    u32 nsub, u32 d, const float *books, u32 entries, int use_sh,
    float onep, float onem, float A, unsigned char *out, unsigned char *how)
{
    assign_body<16>(X, n, stride, nsub, d, books, entries, use_sh,
                   onep, onem, A, out, how);
}
);

#endif
