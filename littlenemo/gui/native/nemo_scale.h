/* nemo_scale.h - a small, safe-critical RGBA8 scaler for the NTSE GUI.
 *
 * Pure C, no globals, no locks, no threads. Works on premultiplied
 * RGBA8 buffers. Filters: nearest (integer-snap, for Pixel Nemo),
 * Lanczos3 and Mitchell-Netravali (smooth art).
 *
 * The caller owns every buffer. Dimensions are bounded and every
 * size is overflow-checked before use.
 */
#ifndef NEMO_SCALE_H
#define NEMO_SCALE_H

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#  ifdef NEMO_SCALE_BUILD
#    define NEMO_SCALE_API __declspec(dllexport)
#  else
#    define NEMO_SCALE_API __declspec(dllimport)
#  endif
#else
#  define NEMO_SCALE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Inclusive maximum for a single side. A buffer may not hold more
 * than NEMO_SCALE_MAX_PIXELS pixels (checked on both src and dst). */
#define NEMO_SCALE_MAX_DIM     8192
#define NEMO_SCALE_MAX_PIXELS  (4096u * 4096u)

enum nemo_scale_filter {
    NEMO_SCALE_NEAREST  = 0,
    NEMO_SCALE_LANCZOS3 = 1,
    NEMO_SCALE_MITCHELL = 2
};

enum nemo_scale_status {
    NEMO_SCALE_OK           = 0,
    NEMO_SCALE_ERR_NULL     = 1,
    NEMO_SCALE_ERR_DIM      = 2,
    NEMO_SCALE_ERR_OVERFLOW = 3,
    NEMO_SCALE_ERR_BUF      = 4,
    NEMO_SCALE_ERR_FILTER   = 5,
    NEMO_SCALE_ERR_NOMEM    = 6
};

/* Scale a premultiplied RGBA8 image.
 *
 * stride is bytes per row; 0 means tightly packed (width * 4).
 * Returns NEMO_SCALE_OK or a nemo_scale_status error. For the
 * separable filters this allocates a temporary workspace and frees
 * it before returning; nearest uses no heap. */
NEMO_SCALE_API int nemo_scale_rgba8(
    const uint8_t *src, int src_w, int src_h, int src_stride,
    uint8_t *dst, int dst_w, int dst_h, int dst_stride,
    int filter);

/* Workspace bytes needed by nemo_scale_rgba8_ex. 0 for nearest, for
 * an identity copy, or if the dimensions are already rejected. */
NEMO_SCALE_API size_t nemo_scale_workspace_size(
    int src_w, int src_h, int dst_w, int dst_h, int filter);

/* Like nemo_scale_rgba8 but uses a caller-provided workspace (no
 * heap). work may be NULL when workspace_size() is 0. */
NEMO_SCALE_API int nemo_scale_rgba8_ex(
    const uint8_t *src, int src_w, int src_h, int src_stride,
    uint8_t *dst, int dst_w, int dst_h, int dst_stride,
    int filter, void *work, size_t work_bytes);

NEMO_SCALE_API const char *nemo_scale_version(void);

#ifdef __cplusplus
}
#endif

#endif
