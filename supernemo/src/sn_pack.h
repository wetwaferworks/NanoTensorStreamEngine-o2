/* sn_pack.h - SuperNemo pack reader (pure C99, no locks).
 *
 * Reads the NQ2 v3 per-tensor-book pack written by nq2-quantize with
 * --books per-tensor --pq D (and the raw float16 section), plus its
 * .nq2manifest (tensor names, block ranges) and the .shapes.tsv sidecar
 * (tensor element counts: a raw tensor's size is not in the manifest).
 *
 * Expansion is per tensor and per weight range, so a caller can stream a
 * 76 M-weight projection in slabs, or expand one 2.75 M-weight expert at
 * a time. Every reader thread owns its own sn_rd (two FILE handles); the
 * sn_pack itself is immutable after open, so no locks exist anywhere.
 *
 * Direction by JD. Converter by Elfie. Co-implementation by Cortana, built
 * with Grok Bot by SpaceX.
 */
#ifndef SN_PACK_H
#define SN_PACK_H

#include <stdint.h>
#include <stdio.h>

typedef struct {
    char *name;
    int raw;                 /* 1: raw float16 section, 0: packed */
    uint64_t count;          /* weights (from the shapes sidecar) */
    uint64_t block0;         /* first block, global (packed only) */
    uint64_t nblocks;        /* blocks (packed only) */
    uint64_t book_off;       /* byte offset of this tensor's book in .nq2book */
    uint64_t raw_off;        /* byte offset of the raw float16 data in .nq2 */
} sn_tensor;

typedef struct {
    char *prefix;
    uint32_t block, entries, pq_d, nsub;   /* nsub = block / pq_d, or 1 for flat */
    uint64_t gen, tot_blocks, body_off;    /* body_off = H, start of the index body */
    uint64_t book_floats;                  /* entries * block */
    sn_tensor *t;
    int n;
    int *ht;                               /* open-addressing name table */
    int ht_cap;
} sn_pack;

typedef struct { FILE *fs, *fb; } sn_rd;

/* Returns 0 on success; on failure prints a plain reason to stderr. */
int sn_pack_open(sn_pack *p, const char *prefix, const char *shapes_tsv);
void sn_pack_close(sn_pack *p);
const sn_tensor *sn_pack_find(const sn_pack *p, const char *name);

int sn_rd_open(const sn_pack *p, sn_rd *r);
void sn_rd_close(sn_rd *r);

/* Expand weights [first, first+n) of tensor t to float32 in out.
 * Packed tensors reconstruct w = mn + sc * book[sub][idx][i]; raw tensors
 * widen float16. Returns 0 or -1 (range error, short read). */
int sn_expand(const sn_pack *p, sn_rd *r, const sn_tensor *t,
              uint64_t first, uint64_t n, float *out);

float sn_f16_to_f32(uint16_t h);

#endif
