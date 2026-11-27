/* sn_pqk.h - SuperNemo packed-matrix kernels: multi-vector, resident-tensor and Dense Tensor Compute (DTC) forms.
 * Pure C99, no locks, no allocation inside the kernels.
 *
 * A PQ4 tensor is a [rows][cols] row-major matrix stored as 256-weight blocks over the flattened matrix; every block
 * carries an fp16 minimum/scale and 64 one-byte indexes (one per 4 weights), the index at position sub choosing an
 * entry of sub-book sub of the tensor's own book. Two ways to multiply it by a vector without ever rebuilding it:
 *
 *   direct: per 4 weights, one book entry dotted with 4 activations.
 *   DTC:    per (activation group, phase) build a 256-entry table T[k] = <book[sub][k], x[4G..4G+3]> ONCE, then every
 *           output row is one table lookup per index byte. The table is shared by all rows, so the build is paid per
 *           activation, not per row; DTC wins on tall matrices (see sn_dtc_pays). A row that does not start on a
 *           block boundary (an expert down_proj has 2688 columns, 10.5 blocks) sees a different sub-book for the same
 *           activation group, so tables are built per PHASE (row*cols mod 256); there are 256/gcd(cols,256) phases.
 *
 * Multi-vector: up to SN_NV activation vectors share one pass over the packed data (speculative-decode verify).
 * Every vector is computed with exactly the operations a single-vector call would use, so a vector's result does not
 * depend on how many other vectors ride along (bit-identical), and rows can be split across lanes freely.
 *
 * DTC reassociates the sum relative to direct (4 interleaved float accumulators per segment), so DTC equals direct to
 * float noise (relative ~1e-6), not bit for bit; the DTC kernels are deterministic and mirrored operation for
 * operation by the GPU lane (sn_cuda_*dtc*), which is what lets caches be bit-identical on and off.
 *
 * Direction by JD. Co-implementation by Cortana, built with Grok Bot by SpaceX. */
#ifndef SN_PQK_H
#define SN_PQK_H

#include "sn_pack.h"

#define SN_NV 8

typedef struct {
    int nv;
    const float *x[SN_NV];
    const double *P[SN_NV];      /* prefix sums of x, cols + 1 (packed tensors) */
    const float *T[SN_NV];       /* DTC tables from sn_dtc_build; NULL = direct */
    float *y[SN_NV];             /* rows r0..r1 land in y[v][0 .. r1-r0) */
} sn_mv;
#define SN_MV_INIT(n) { (n), { 0 }, { 0 }, { 0 }, { 0 } }

/* A whole packed tensor resident in RAM (what the hot-expert cache keeps). Identical bytes to the pack, so every
 * kernel returns identical bits from it. */
typedef struct {
    uint8_t *idx; uint16_t *mn, *sc; float *book;
    uint64_t nblocks, bytes;
} sn_pq_res;

uint64_t sn_res_bytes(const sn_pack *p, const sn_tensor *t);          /* RAM a resident copy takes (0 for raw) */
int sn_res_load(const sn_pack *p, sn_rd *r, const sn_tensor *t, sn_pq_res *out);   /* 0 / -1 */
void sn_res_free(sn_pq_res *res);

/* ---- DTC geometry and tables ---- */
/* 0 if DTC applies to a [rows][cols] tensor of this pack (PQ4, cols % 4 == 0); fills phases and groups. */
int sn_dtc_geom(const sn_pack *p, int cols, int *nph, int *ng);
size_t sn_dtc_floats(const sn_pack *p, int cols);                      /* floats in one vector's table set */
/* Build groups [g0, g1) of every phase (disjoint ranges can be built by different lanes into the same T). */
void sn_dtc_build(const sn_pack *p, const float *book, int cols, const float *x, int g0, int g1, float *T);
/* Would DTC beat direct for this shape? build_lanes: how many lanes share the table build. */
int sn_dtc_pays(const sn_pack *p, int rows, int cols, int nv, int build_lanes);

/* ---- kernels (row range [r0, r1) of a [*][cols] tensor) ---- */
int sn_pqk_rows(const sn_pack *p, sn_rd *r, const sn_tensor *t, const float *book, int cols, const sn_mv *mv, int r0, int r1);
int sn_pqk_rows_res(const sn_pack *p, const sn_pq_res *res, const sn_tensor *t, int cols, const sn_mv *mv, int r0, int r1);
int sn_rawk_rows(const sn_pack *p, sn_rd *r, const sn_tensor *t, int cols, const sn_mv *mv, int r0, int r1);

#endif
