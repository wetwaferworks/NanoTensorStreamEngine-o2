/* nemo_tokpipe.h - prompt tokenization overlapped with the GPU prefill.
 *
 * The prompt is cut into segments at points where encoding the pieces
 * separately gives exactly the ids of encoding the whole text (between an
 * ASCII letter and a space that is followed by an ASCII letter, and only
 * when nemo_tokenizer_cut_safe says the tokenizer allows it). Segment 0 is
 * encoded on the caller's thread so the prefill can start at once; the
 * rest are encoded in order by one background job (the nemo_pool prep
 * thread) while the GPU runs the first chunks. Each finished segment is
 * published by bumping `ready` (release); the consumer reads it with
 * acquire. Zero mutexes: one writer per field, epoch-style handoff.
 * Idea and direction: JD. Co-implementation: Cortana. Pure C. */
#ifndef NEMO_TOKPIPE_H
#define NEMO_TOKPIPE_H
#include <stddef.h>
#include <stdint.h>
#include "tokenizer.h"

#define NEMO_TOKPIPE_MAXSEG 512

typedef struct {
    const nemo_tokenizer *tk;
    const char *text;
    size_t len;
    int nseg;                              /* segments (1: not split) */
    size_t cut[NEMO_TOKPIPE_MAXSEG + 1];   /* byte offsets, cut[0] = 0, cut[nseg] = len */
    int *buf;                              /* per-segment id slices */
    size_t off[NEMO_TOKPIPE_MAXSEG + 1];   /* slice k starts at buf + off[k] */
    int cnt[NEMO_TOKPIPE_MAXSEG];          /* ids in segment k (valid once published) */
    char *seg;                             /* scratch copy for NUL-terminated segments (job side) */
    uint32_t ready;                        /* segments published; the job is the one writer */
    uint32_t failed;                       /* set (before publishing) when a segment fails */
} nemo_tokpipe;

/* Plan the cuts: segment 0 about first_bytes, the rest about seg_bytes.
 * Returns the number of segments (1 when the text cannot or need not be
 * split), or -1 on allocation failure. */
int nemo_tokpipe_plan(nemo_tokpipe *t, const nemo_tokenizer *tk, const char *text,
                      size_t first_bytes, size_t seg_bytes);
/* Encode segment 0 on the calling thread and publish it. Returns its id
 * count (or -1). */
int nemo_tokpipe_head(nemo_tokpipe *t);
/* Background body (nemo_prep_fn): encodes segments 1..nseg-1 in order. */
void nemo_tokpipe_run(void *ctx);
/* Wait until segment k is published; returns its id count, -1 on failure.
 * *ids receives a pointer to its ids. */
int nemo_tokpipe_wait(nemo_tokpipe *t, int k, const int **ids);
void nemo_tokpipe_free(nemo_tokpipe *t);
#endif
