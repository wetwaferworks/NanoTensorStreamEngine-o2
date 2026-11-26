/* nq2-dc-kernel.h ... THE dictionary-compute kernel, shared.
 *
 * Implemented by Elfie (E.L.F.I.E.). Direction and executive
 * planning by JD.
 *
 * Extracted 2026-11-25 (DTC code review P1-1 / readiness R-1):
 * before this header existed, tools/nq2-dc.c (the shipped
 * probe) and tools/nq2-dc-adv.c (the adversarial campaign)
 * each carried a private copy of the build/apply logic, so the
 * campaign's 23 HOLDs certified the copy, not the shipped
 * code. Now there is exactly one implementation, and both
 * tools call it. The consume-loop integration (DTC on the
 * Triforce consume loop) plugs in this same kernel.
 *
 * The algebra: with weights stored as
 *     w[r][jB+i] = s[r][j] * C[idx[r][j]][i] + m[r][j]
 * the product y = W x is computed without reconstructing W:
 *     T[j][k] = sum_i C[k][i] * x[jB+i],   A[j] = sum_i x[jB+i]
 *     y[r]    = sum_j ( s[r][j] * T[j][idx[r][j]] + m[r][j] * A[j] )
 *
 * Interface: the kernel takes the book pointer, the entry
 * count and the block size as parameters and knows nothing
 * about any pack file layout. The v1 probe passes its global
 * book; a v2 caller passes the tensor's own book pointer per
 * call. Nothing here allocates; T (nseg * entries floats) and
 * A (nseg floats) are caller-owned workspace.
 *
 * DETERMINISM CONTRACT: the accumulation orders below are the
 * pinned behavior, not an implementation detail. A[j] sums a
 * segment in ascending i; each table entry sums ascending i;
 * apply accumulates per row in ascending j. Reordering any of
 * these changes the output bits and moves the campaign's FNV
 * anchors (839d3f72ad1a3d60, bdae48bdf7a78a40,
 * 35e1adcc254bf68e, 1cd08a5dff229720). Do not "optimize" the
 * loop order here.
 */

#ifndef NQ2_DC_KERNEL_H
#define NQ2_DC_KERNEL_H

#include <stdint.h>
#include <stddef.h>

/* A[j]: the sum of one activation segment. */
static float nqdc_seg_sum(const float *xseg, uint32_t block)
{
    float a = 0;
    for (uint32_t i = 0; i < block; i++) a += xseg[i];
    return a;
}

/* T[j][k]: one table entry, the dot of prototype k with the
 * segment starting at xseg. Exposed on its own because the
 * lazy and active-mask construction policies build single
 * entries on demand; they must produce the same bits as the
 * eager build, so they call this, not a local loop. */
static float nqdc_entry(const float *book, uint32_t block,
                        const float *xseg, uint32_t k)
{
    const float *proto = book + (size_t)k * block;
    float acc = 0;
    for (uint32_t i = 0; i < block; i++) acc += proto[i] * xseg[i];
    return acc;
}

/* Eager build: fill T (nseg * entries) and A (nseg) for the
 * activation x (nseg * block floats) against the given book. */
static void nqdc_build(const float *book, uint32_t entries,
                       uint32_t block, const float *x,
                       uint64_t nseg, float *T, float *A)
{
    for (uint64_t j = 0; j < nseg; j++) {
        const float *xseg = x + (size_t)j * block;
        A[j] = nqdc_seg_sum(xseg, block);
        for (uint32_t k = 0; k < entries; k++)
            T[(size_t)j * entries + k] = nqdc_entry(book, block, xseg, k);
    }
}

/* Apply: y[r] = sum_j ( sc[r][j] * T[j][idx[r][j]]
 *                     + mn[r][j] * A[j] ), rows ascending,
 * segments ascending per row. idx/sc/mn are row-major
 * rows * nseg. */
static void nqdc_apply(const uint8_t *idx, const float *sc,
                       const float *mn, const float *T,
                       const float *A, uint32_t entries,
                       uint64_t nseg, uint64_t rows, float *y)
{
    for (uint64_t r = 0; r < rows; r++) {
        float acc = 0;
        for (uint64_t j = 0; j < nseg; j++) {
            uint64_t b = r * nseg + j;
            acc += sc[b] * T[(size_t)j * entries + idx[b]]
                 + mn[b] * A[j];
        }
        y[r] = acc;
    }
}

#endif /* NQ2_DC_KERNEL_H */
