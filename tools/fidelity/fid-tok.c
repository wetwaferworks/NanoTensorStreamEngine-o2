/* fid-tok.c ... fidelity lab: real text to token-id sequences.
 *
 * Co-implemented by Cortana at JD's direction, 2026-11-27, for the
 * fidelity roadmap (docs/ROADMAP.md). Pure C99.
 *
 * Reads text files already on disk (plain text, or HTML with the
 * tags stripped here), tokenizes each with a Hugging Face
 * tokenizer.json through Little Nemo's byte-level BPE tokenizer,
 * and cuts non-overlapping windows of LEN tokens: BOS first, then
 * LEN-1 text tokens. One window per output line, ids separated by
 * spaces, ready for nq2-infer-lab --teacher.
 *
 * Usage:
 *   fid-tok --tokenizer tokenizer.json --len 256 --per-file N
 *           --bos 1 --out ids.txt file1 [file2 ...]
 *
 * Windows are taken evenly spaced through each file, at most N per
 * file, so a long manual does not crowd out the others. A summary
 * (files, windows, bytes per token) goes to stderr.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../littlenemo/src/tokenizer.h"

static char *slurp(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    size_t cap = 1 << 16, n = 0;
    char *b = (char *)malloc(cap + 1);
    for (;;) {
        if (n == cap) { cap *= 2; b = (char *)realloc(b, cap + 1); if (!b) { fclose(f); return NULL; } }
        size_t r = fread(b + n, 1, cap - n, f);
        if (r == 0) break;
        n += r;
    }
    fclose(f);
    b[n] = 0;
    *len = n;
    return b;
}

static int ends_with(const char *s, const char *suf)
{
    size_t a = strlen(s), b = strlen(suf);
    if (a < b) return 0;
    for (size_t i = 0; i < b; i++) {
        char c = s[a - b + i];
        if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
        if (c != suf[i]) return 0;
    }
    return 1;
}

/* Strip tags, scripts and styles; decode the common entities;
 * collapse runs of spaces and of blank lines. In place. */
static void clean_text(char *s, int html)
{
    char *o = s, *p = s;
    while (*p) {
        if (html && *p == '<') {
            if (!strncmp(p, "<script", 7) || !strncmp(p, "<style", 6)) {
                const char *end = strstr(p, !strncmp(p, "<script", 7) ? "</script" : "</style");
                if (!end) break;
                p = (char *)end;
            }
            int block = !strncmp(p, "<p", 2) || !strncmp(p, "<br", 3) || !strncmp(p, "<h", 2)
                     || !strncmp(p, "<li", 3) || !strncmp(p, "<div", 4) || !strncmp(p, "<tr", 3)
                     || !strncmp(p, "</p", 3) || !strncmp(p, "<pre", 4) || !strncmp(p, "</pre", 5)
                     || !strncmp(p, "<dt", 3) || !strncmp(p, "<dd", 3);
            while (*p && *p != '>') p++;
            if (*p) p++;
            if (block) *o++ = '\n';
            continue;
        }
        if (html && *p == '&') {
            static const struct { const char *e; char c; } ent[] = {
                {"&amp;", '&'}, {"&lt;", '<'}, {"&gt;", '>'}, {"&quot;", '"'},
                {"&#39;", '\''}, {"&nbsp;", ' '}, {"&apos;", '\''}, {"&#x27;", '\''} };
            int hit = 0;
            for (size_t k = 0; k < sizeof(ent) / sizeof(ent[0]); k++) {
                size_t L = strlen(ent[k].e);
                if (!strncmp(p, ent[k].e, L)) { *o++ = ent[k].c; p += L; hit = 1; break; }
            }
            if (hit) continue;
        }
        if (*p == '\r') { p++; continue; }
        if (*p == '\f' || *p == '\v') { *o++ = '\n'; p++; continue; }
        *o++ = *p++;
    }
    *o = 0;
    /* collapse */
    char *q = s; o = s;
    int nl = 0, sp = 0;
    while (*q == ' ' || *q == '\n' || *q == '\t') q++;
    for (; *q; q++) {
        if (*q == '\n') { nl++; sp = 0; continue; }
        if (*q == ' ' || *q == '\t') { sp = 1; continue; }
        if (nl) { *o++ = '\n'; if (nl > 1) *o++ = '\n'; nl = 0; sp = 0; }
        else if (sp) { *o++ = ' '; sp = 0; }
        *o++ = *q;
    }
    *o = 0;
}

int main(int argc, char **argv)
{
    const char *tkp = NULL, *outp = NULL;
    int len = 256, per = 4, bos = 1;
    int first = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--tokenizer") && i + 1 < argc) tkp = argv[++i];
        else if (!strcmp(argv[i], "--len") && i + 1 < argc) len = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--per-file") && i + 1 < argc) per = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bos") && i + 1 < argc) bos = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) outp = argv[++i];
        else { first = i; break; }
    }
    if (!tkp || !outp || !first || len < 2 || per < 1) {
        fprintf(stderr, "usage: fid-tok --tokenizer T --len 256 --per-file N --bos 1 --out ids.txt files...\n");
        return 1;
    }
    nemo_tokenizer *tk = nemo_tokenizer_load(tkp);
    if (!tk) { fprintf(stderr, "cannot load tokenizer %s\n", tkp); return 1; }
    FILE *out = fopen(outp, "w");
    if (!out) { fprintf(stderr, "cannot open %s\n", outp); return 1; }
    long total_w = 0;
    double bytes = 0, toks = 0;
    for (int fi = first; fi < argc; fi++) {
        size_t n = 0;
        char *txt = slurp(argv[fi], &n);
        if (!txt) { fprintf(stderr, "skip %s: cannot read\n", argv[fi]); continue; }
        clean_text(txt, ends_with(argv[fi], ".html") || ends_with(argv[fi], ".htm"));
        size_t clen = strlen(txt);
        int *ids = (int *)malloc((clen + 1) * sizeof(int));
        int m = ids ? nemo_tokenizer_encode(tk, txt, ids, (int)clen + 1) : -1;
        if (m < 0) { fprintf(stderr, "skip %s: tokenize failed\n", argv[fi]); free(txt); free(ids); continue; }
        int body = len - 1;
        int avail = m / body;
        int take = avail < per ? avail : per;
        for (int w = 0; w < take; w++) {
            long start = (long)((double)w * (avail - 1 > 0 ? avail - 1 : 0) / (take > 1 ? take - 1 : 1)) * body;
            fprintf(out, "%d", bos);
            for (int k = 0; k < body; k++) fprintf(out, " %d", ids[start + k]);
            fprintf(out, "\n");
        }
        fprintf(stderr, "%s: %zu chars, %d tokens, %d windows\n", argv[fi], clen, m, take);
        total_w += take;
        bytes += (double)clen; toks += m;
        free(txt); free(ids);
    }
    fclose(out);
    fprintf(stderr, "total: %ld windows of %d tokens, %.2f chars per token\n",
            total_w, len, toks > 0 ? bytes / toks : 0.0);
    nemo_tokenizer_free(tk);
    return 0;
}
