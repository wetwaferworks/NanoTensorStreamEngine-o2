/* accel_kernels_cuda.h - CUDA C source of the NTSE accel kernels,
 * compiled at run time by NVRTC (no nvcc / MSVC at build time).
 * Co-implemented by Cortana; the algebra and the pinned
 * accumulation orders are Elfie's (tools/nq2-dc-kernel.h).
 *
 * Exactness: every fp32 add and multiply is an explicit
 * round-to-nearest intrinsic (__fadd_rn / __fmul_rn), which the
 * compiler may never contract into an FMA, and NVRTC also runs
 * with --fmad=false. Together with the same loop orders as the CPU
 * kernel, the DTC kernels produce the CPU kernel's bits. fp16
 * frames widen through cvt.f32.f16, an exact conversion.
 * No atomics anywhere; every reduction is a fixed tree.
 */
#ifndef NTSE_ACCEL_KERNELS_CUDA_H
#define NTSE_ACCEL_KERNELS_CUDA_H

#define NTSE_CU_STR(...) #__VA_ARGS__

static const char ntse_cuda_src[] = NTSE_CU_STR(
typedef unsigned long long u64;
typedef unsigned int u32;

__device__ __forceinline__ float h2f(unsigned short h)
{
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

extern "C" __global__ void k_segsum(const float *__restrict__ x,
    float *__restrict__ A, u64 nseg, u32 B, u32 batch, u64 cols)
{
    u64 n = (u64)batch * nseg;
    for (u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x; t < n;
         t += (u64)gridDim.x * blockDim.x) {
        u64 b = t / nseg, j = t - b * nseg;
        const float *xs = x + b * cols + j * B;
        float a = 0.0f;
        for (u32 i = 0; i < B; i++) a = __fadd_rn(a, xs[i]);
        A[t] = a;
    }
}

extern "C" __global__ void k_build_eager(const float *__restrict__ bookT,
    const float *__restrict__ x, float *__restrict__ T, u64 nseg, u32 B,
    u32 K, u32 batch, u64 cols)
{
    u64 n = (u64)batch * nseg * K;
    for (u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x; t < n;
         t += (u64)gridDim.x * blockDim.x) {
        u64 k = t % K, bj = t / K;
        u64 b = bj / nseg, j = bj - b * nseg;
        const float *xs = x + b * cols + j * B;
        float acc = 0.0f;
        for (u32 i = 0; i < B; i++)
            acc = __fadd_rn(acc, __fmul_rn(bookT[(u64)i * K + k], xs[i]));
        T[t] = acc;
    }
}

extern "C" __global__ void k_build_active(const float *__restrict__ bookT,
    const float *__restrict__ x, float *__restrict__ T,
    const u32 *__restrict__ act, u64 nact, u64 nseg, u32 B, u32 K,
    u32 batch, u64 cols)
{
    u64 n = (u64)batch * nact;
    for (u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x; t < n;
         t += (u64)gridDim.x * blockDim.x) {
        u64 b = t / nact, a = t - b * nact;
        u64 code = act[a];
        u64 j = code / K, k = code - j * K;
        const float *xs = x + b * cols + j * B;
        float acc = 0.0f;
        for (u32 i = 0; i < B; i++)
            acc = __fadd_rn(acc, __fmul_rn(bookT[(u64)i * K + k], xs[i]));
        T[(b * nseg + j) * K + k] = acc;
    }
}

/* lazy: idempotent flag writes (every writer stores 1), no
 * read-modify-write, so concurrent marks are race-free */
extern "C" __global__ void k_mark(const unsigned char *__restrict__ idxT,
    unsigned int *__restrict__ F, u64 rows, u64 nseg, u32 K)
{
    u64 n = rows * nseg;
    for (u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x; t < n;
         t += (u64)gridDim.x * blockDim.x) {
        u64 j = t / rows;
        F[j * K + idxT[t]] = 1u;
    }
}

extern "C" __global__ void k_build_lazy(const float *__restrict__ bookT,
    const float *__restrict__ x, float *__restrict__ T,
    const unsigned int *__restrict__ F, u64 nseg, u32 B, u32 K, u32 batch,
    u64 cols)
{
    u64 n = (u64)batch * nseg * K;
    for (u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x; t < n;
         t += (u64)gridDim.x * blockDim.x) {
        u64 k = t % K, bj = t / K;
        u64 b = bj / nseg, j = bj - b * nseg;
        if (!F[j * K + k]) continue;
        const float *xs = x + b * cols + j * B;
        float acc = 0.0f;
        for (u32 i = 0; i < B; i++)
            acc = __fadd_rn(acc, __fmul_rn(bookT[(u64)i * K + k], xs[i]));
        T[t] = acc;
    }
}

extern "C" __global__ void k_apply(const unsigned char *__restrict__ idxT,
    const unsigned short *__restrict__ mnT,
    const unsigned short *__restrict__ scT, const float *__restrict__ T,
    const float *__restrict__ A, float *__restrict__ y, u64 rows, u64 nseg,
    u32 K, u32 batch)
{
    u64 n = (u64)batch * rows;
    for (u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x; t < n;
         t += (u64)gridDim.x * blockDim.x) {
        u64 b = t / rows, r = t - b * rows;
        const float *Tb = T + b * nseg * K;
        const float *Ab = A + b * nseg;
        float acc = 0.0f;
        for (u64 j = 0; j < nseg; j++) {
            u64 e = j * rows + r;
            float s = h2f(scT[e]), m = h2f(mnT[e]);
            u32 k = idxT[e];
            acc = __fadd_rn(acc, __fadd_rn(__fmul_rn(s, Tb[j * K + k]),
                                           __fmul_rn(m, Ab[j])));
        }
        y[b * rows + r] = acc;
    }
}

extern "C" __global__ void k_apply_direct(const float *__restrict__ book,
    const float *__restrict__ x, const unsigned char *__restrict__ idxT,
    const unsigned short *__restrict__ mnT,
    const unsigned short *__restrict__ scT, const float *__restrict__ A,
    float *__restrict__ y, u64 rows, u64 nseg, u32 B, u32 batch, u64 cols)
{
    u64 n = (u64)batch * rows;
    for (u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x; t < n;
         t += (u64)gridDim.x * blockDim.x) {
        u64 b = t / rows, r = t - b * rows;
        const float *Ab = A + b * nseg;
        float acc = 0.0f;
        for (u64 j = 0; j < nseg; j++) {
            u64 e = j * rows + r;
            float s = h2f(scT[e]), m = h2f(mnT[e]);
            const float *proto = book + (u64)idxT[e] * B;
            const float *xs = x + b * cols + j * B;
            float dot = 0.0f;
            for (u32 i = 0; i < B; i++)
                dot = __fadd_rn(dot, __fmul_rn(proto[i], xs[i]));
            acc = __fadd_rn(acc, __fadd_rn(__fmul_rn(s, dot),
                                           __fmul_rn(m, Ab[j])));
        }
        y[b * rows + r] = acc;
    }
}

extern "C" __global__ void k_expand(const float *__restrict__ book,
    const unsigned char *__restrict__ idxT,
    const unsigned short *__restrict__ mnT,
    const unsigned short *__restrict__ scT, float *__restrict__ W,
    u64 rows, u64 cols, u32 B)
{
    u64 n = rows * cols;
    for (u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x; t < n;
         t += (u64)gridDim.x * blockDim.x) {
        u64 r = t / cols, c = t - r * cols;
        u64 j = c / B, i = c - j * B;
        u64 e = j * rows + r;
        W[t] = __fadd_rn(h2f(mnT[e]),
                         __fmul_rn(h2f(scT[e]), book[(u64)idxT[e] * B + i]));
    }
}

extern "C" __global__ void __launch_bounds__(256) k_gemv(
    const float *__restrict__ W, const float *__restrict__ x,
    float *__restrict__ y, u64 rows, u64 cols, u32 batch)
{
    __shared__ float sh[256];
    u64 npair = (u64)batch * rows;
    for (u64 p = blockIdx.x; p < npair; p += gridDim.x) {
        u64 b = p / rows, r = p - b * rows;
        const float *w = W + r * cols;
        const float *xb = x + b * cols;
        float acc = 0.0f;
        for (u64 c = threadIdx.x; c < cols; c += 256)
            acc = __fadd_rn(acc, __fmul_rn(w[c], xb[c]));
        sh[threadIdx.x] = acc;
        __syncthreads();
        for (u32 s = 128; s > 0; s >>= 1) {
            if (threadIdx.x < s)
                sh[threadIdx.x] = __fadd_rn(sh[threadIdx.x], sh[threadIdx.x + s]);
            __syncthreads();
        }
        if (threadIdx.x == 0) y[b * rows + r] = sh[0];
        __syncthreads();
    }
}

extern "C" __global__ void __launch_bounds__(256) k_linear_bf16(
    const unsigned short *__restrict__ W, const float *__restrict__ x,
    float *__restrict__ y, u32 out, u32 in)
{
    __shared__ double sh[256];
    for (u64 o = blockIdx.x; o < out; o += gridDim.x) {
        const unsigned short *row = W + o * in;
        double acc = 0.0;
        for (u32 i = threadIdx.x; i < in; i += 256) {
            float w = __uint_as_float(((u32)row[i]) << 16);
            acc = __dadd_rn(acc, __dmul_rn((double)w, (double)x[i]));
        }
        sh[threadIdx.x] = acc;
        __syncthreads();
        for (u32 s = 128; s > 0; s >>= 1) {
            if (threadIdx.x < s)
                sh[threadIdx.x] = __dadd_rn(sh[threadIdx.x], sh[threadIdx.x + s]);
            __syncthreads();
        }
        if (threadIdx.x == 0) y[o] = __double2float_rn(sh[0]);
        __syncthreads();
    }
}
);

#endif
