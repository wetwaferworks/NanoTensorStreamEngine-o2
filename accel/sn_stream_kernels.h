/* sn_stream_kernels.h - CUDA kernels for the SuperNemo expert stream (NVRTC source, compiled at run time by sn_cuda.c).
 *
 * JD's standing CUDA design (2026-11-27): compressed layers travel to the GPU, are expanded there with CUDA, the math runs
 * on the GPU, and the expanded copy falls away before the next token. These kernels are the "expand there" and "math there"
 * halves. Every one of them is written operation for operation like the CPU code it replaces and is compiled with
 * --fmad=false, so the GPU produces the SAME BITS as the CPU:
 *
 *   xs_expand_pq     packed NanoTensor blocks -> float weights, w = mn + sc * book[sub][idx]   (sn_expand_book)
 *   xs_expand_raw    raw float16 -> float                                                      (sn_expand_book, raw)
 *   xs_gemm          Y = act(W X) on expanded weights, 8-lane fused multiply-add chains summed like the AVX2 hsum8
 *                    (sn_gemm_rows rows_avx2), or one plain running sum (rows_scalar) when the CPU has no AVX2
 *   xs_relu2_prefix  ReLU squared on an expert's middle activations and their double prefix sum (fx_job in sn_model.c)
 *
 * The decode path does not materialise weights at all: the DTC kernels in sn_cuda.c decode the packed indexes in registers
 * through per-token tables (bit-identical to the CPU DTC kernel). That is still "expanded on the GPU", just never written
 * out, which keeps the card's memory traffic at the compressed size.
 *
 * Appended to the base kernel string in sn_cuda.c (it defines u8, u16, u64 and h2f). No locks, no atomics.
 *
 * Direction by JD. Co-implementation by Cortana, built with Grok Bot by SpaceX. */
#ifndef SN_STREAM_KERNELS_H
#define SN_STREAM_KERNELS_H

static const char *SN_STREAM_KERNELS =
/* one thread per 4-weight group: group G of the tensor lives in block G >> 6 at sub-book G & 63 (block 256, PQ d4) */
"extern \"C\" __global__ void xs_expand_pq(const u8 *idx, const u16 *mn, const u16 *sc, const float4 *book,\n"
"                                         u64 g0, u64 ng, float4 *out) {\n"
"  u64 g = (u64)blockIdx.x * (u64)blockDim.x + (u64)threadIdx.x;\n"
"  if (g >= ng) return;\n"
"  u64 G = g0 + g, b = G >> 6; int sub = (int)(G & 63ull);\n"
"  float m = h2f(mn[b]), s = h2f(sc[b]);\n"
"  float4 c = book[sub * 256 + idx[G]];\n"
"  float4 o; o.x = m + s * c.x; o.y = m + s * c.y; o.z = m + s * c.z; o.w = m + s * c.w;\n"
"  out[g] = o;\n"
"}\n"
"extern \"C\" __global__ void xs_expand_raw(const u16 *w, u64 n, float *out) {\n"
"  u64 i = (u64)blockIdx.x * (u64)blockDim.x + (u64)threadIdx.x;\n"
"  if (i < n) out[i] = h2f(w[i]);\n"
"}\n"
/* thread (row r, token n). xi / yi map token n to its input / output row (NULL = n). Writes Y[yrow * ystride + r0 + r]. */
"extern \"C\" __global__ void xs_gemm(const float *W, int rows, int K, const float *X, const int *xi, int N,\n"
"                                    float *Y, const int *yi, int ystride, int r0, int act, int simd) {\n"
"  int n = blockIdx.x * blockDim.x + threadIdx.x;\n"
"  int r = blockIdx.y * blockDim.y + threadIdx.y;\n"
"  if (n >= N || r >= rows) return;\n"
"  const float *x = X + (u64)(xi ? xi[n] : n) * (u64)K;\n"
"  const float *w = W + (u64)r * (u64)K;\n"
"  float s;\n"
"  if (simd) {\n"
"    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, a4 = 0.f, a5 = 0.f, a6 = 0.f, a7 = 0.f;\n"
"    for (int k = 0; k < K; k += 8) {\n"
"      a0 = __fmaf_rn(w[k], x[k], a0);         a1 = __fmaf_rn(w[k + 1], x[k + 1], a1);\n"
"      a2 = __fmaf_rn(w[k + 2], x[k + 2], a2); a3 = __fmaf_rn(w[k + 3], x[k + 3], a3);\n"
"      a4 = __fmaf_rn(w[k + 4], x[k + 4], a4); a5 = __fmaf_rn(w[k + 5], x[k + 5], a5);\n"
"      a6 = __fmaf_rn(w[k + 6], x[k + 6], a6); a7 = __fmaf_rn(w[k + 7], x[k + 7], a7);\n"
"    }\n"
"    s = ((a0 + a4) + (a1 + a5)) + ((a2 + a6) + (a3 + a7));\n"
"  } else {\n"
"    s = 0.f;\n"
"    for (int k = 0; k < K; k++) s += w[k] * x[k];\n"
"  }\n"
"  s = s + 0.0f;\n"
"  if (act == 2) { float q = s > 0.f ? s : 0.f; s = q * q; }\n"
"  Y[(u64)(yi ? yi[n] : n) * (u64)ystride + (u64)r0 + (u64)r] = s;\n"
"}\n"
/* one block: ReLU squared in parallel, then thread 0 runs the double prefix sum in order (as the CPU does) */
"extern \"C\" __global__ void xs_relu2_prefix(float *mid, double *P, int I) {\n"
"  extern __shared__ float sm[];\n"
"  for (int j = threadIdx.x; j < I; j += blockDim.x) { float v = mid[j]; v = v > 0.0f ? v * v : 0.0f; mid[j] = v; sm[j] = v; }\n"
"  __syncthreads();\n"
"  if (threadIdx.x == 0) {\n"
"    double p = 0.0; P[0] = 0.0;\n"
"    for (int j = 0; j < I; j++) { p = p + (double)sm[j]; P[j + 1] = p; }\n"
"  }\n"
"}\n";

#endif
