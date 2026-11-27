/* nemo_tokpipe.c - see nemo_tokpipe.h. Pure C, zero mutexes. */
#include "nemo_tokpipe.h"
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
static void tp_yield(void) { SwitchToThread(); }
#else
#include <sched.h>
static void tp_yield(void) { sched_yield(); }
#endif

static int is_al(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }

/* p is a safe cut when text[p-1] is an ASCII letter, text[p] a space and
 * text[p+1] an ASCII letter. Under the checkpoint's Split pattern a piece
 * that contains a letter never takes a following space (no alternative
 * matches letter-then-space), and a space before a letter starts a new
 * piece ([^\r\n\p{L}\p{N}]?[letters]); the scanner's state is only its
 * position, so encoding [0,p) and [p,len) separately gives the same
 * pieces, and BPE never merges across a piece. */
static int safe_at(const char *s, size_t len, size_t p) {
    return p > 0 && p + 1 < len && s[p] == ' ' && is_al(s[p - 1]) && is_al(s[p + 1]);
}

int nemo_tokpipe_plan(nemo_tokpipe *t, const nemo_tokenizer *tk, const char *text,
                      size_t first_bytes, size_t seg_bytes) {
    memset(t, 0, sizeof *t);
    t->tk = tk; t->text = text; t->len = strlen(text);
    t->cut[0] = 0; t->nseg = 1;
    if (nemo_tokenizer_cut_safe(tk) && seg_bytes > 0) {
        size_t want = first_bytes;
        size_t p = want;
        while (p < t->len && t->nseg < NEMO_TOKPIPE_MAXSEG) {
            while (p < t->len && !safe_at(text, t->len, p)) p++;
            if (p >= t->len) break;
            t->cut[t->nseg++] = p;
            p += seg_bytes;
        }
    }
    t->cut[t->nseg] = t->len;
    /* every emitted id consumes at least one byte: bytes + 1 per slice */
    size_t tot = 0;
    for (int k = 0; k < t->nseg; k++) { t->off[k] = tot; tot += t->cut[k + 1] - t->cut[k] + 1; }
    t->off[t->nseg] = tot;
    t->buf = (int *)malloc(tot * sizeof(int));
    size_t mx = 0;
    for (int k = 0; k < t->nseg; k++) if (t->cut[k + 1] - t->cut[k] > mx) mx = t->cut[k + 1] - t->cut[k];
    t->seg = (char *)malloc(mx + 1);
    if (!t->buf || !t->seg) { free(t->buf); free(t->seg); t->buf = NULL; t->seg = NULL; return -1; }
    return t->nseg;
}

static int enc_seg(nemo_tokpipe *t, int k, char *scratch) {
    size_t b0 = t->cut[k], b1 = t->cut[k + 1];
    memcpy(scratch, t->text + b0, b1 - b0); scratch[b1 - b0] = 0;
    return nemo_tokenizer_encode(t->tk, scratch, t->buf + t->off[k], (int)(b1 - b0 + 1));
}

int nemo_tokpipe_head(nemo_tokpipe *t) {
    char *s = (char *)malloc(t->cut[1] - t->cut[0] + 1);
    int c = s ? enc_seg(t, 0, s) : -1;
    free(s);
    t->cnt[0] = c;
    if (c < 0) t->failed = 1;
    __atomic_store_n(&t->ready, 1u, __ATOMIC_RELEASE);
    return c;
}

void nemo_tokpipe_run(void *ctx) {
    nemo_tokpipe *t = (nemo_tokpipe *)ctx;
    for (int k = 1; k < t->nseg; k++) {
        int c = t->failed ? -1 : enc_seg(t, k, t->seg);
        t->cnt[k] = c;
        if (c < 0) t->failed = 1;
        __atomic_store_n(&t->ready, (uint32_t)(k + 1), __ATOMIC_RELEASE);
    }
}

int nemo_tokpipe_wait(nemo_tokpipe *t, int k, const int **ids) {
    int spins = 0;
    while (__atomic_load_n(&t->ready, __ATOMIC_ACQUIRE) <= (uint32_t)k) {
        if (++spins > 256) tp_yield();
    }
    *ids = t->buf + t->off[k];
    return t->cnt[k];
}

void nemo_tokpipe_free(nemo_tokpipe *t) { free(t->buf); free(t->seg); t->buf = NULL; t->seg = NULL; }
