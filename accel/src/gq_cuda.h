/* gq_cuda.h - CUDA lane for nq2-quantize (research, scratch).
 *
 * Direction: JD. Co-implemented by Cortana on Elfie's quantizer.
 *
 * Nearest-code assignment on the GPU with the CPU path's exact
 * answer. For every row b and subvector s the result is
 *
 *     argmin over k of  sum_i ((double)x[i] - (double)c_k[i])^2
 *
 * summed in i = 0..d-1, compared in ascending k with a strict <,
 * starting from best = 0 and best_err = 1e30: the same rule, and
 * the same double roundings, as nearest_centroid() in
 * tools/nq2-quantize.c. See gq_cuda.c for how the fast fp32 screen
 * stays exact.
 *
 * Threads: any number of host threads may call in at once. Each
 * calling thread gets its own stream and its own device buffers
 * (thread-local), so nothing is shared and nothing is locked.
 * gq_init() must have returned 0 before the first call, and
 * gq_thread_release() frees the calling thread's buffers.
 */
#ifndef GQ_CUDA_H
#define GQ_CUDA_H

#include <stddef.h>
#include <stdint.h>

/* Load the CUDA driver and NVRTC, pick the device, compile the
 * kernels. 0 on success; otherwise why says what is missing. */
int gq_init(int device, char *why, size_t wl);
const char *gq_device_name(void);
void gq_shutdown(void);

/* Upload n rows of `stride` floats for this thread to work on. */
int gq_set_rows(const float *x, uint64_t n, uint32_t stride);

/* Assign every resident row's nsub subvectors (subvector s starts
 * at float s*d of the row) to their nearest code. books holds nsub
 * books of `entries` codes of d floats. out[b * nsub + s]. */
int gq_assign_rows(uint32_t nsub, uint32_t d, const float *books,
                   uint32_t entries, uint8_t *out);

/* Exact-rescore telemetry for this thread: rows whose fp32 screen
 * left more than one candidate, and rows that fell back to the
 * full double search (a non-finite fp32 distance). */
void gq_thread_stats(uint64_t *rows, uint64_t *multi, uint64_t *full);

void gq_thread_release(void);

#endif
