/* test_sn_pack_manifest.c - hostile-manifest checks for sn_pack_open().
 *
 * Starts from the tiny SuperNemo pack and mutates only the .nq2manifest:
 * unknown kind, wrong block size, missing trailing "end", and valid packs
 * that carry optional gate lines before or after "end".
 *
 * Found in Archie's round 7 review.
 * Pure C99, no locks.
 *
 *   test_sn_pack_manifest [TINY_PREFIX]
 * Default prefix: testdata/tiny/tiny (run from the tree root) or
 * ../testdata/tiny/tiny (run from supernemo/).
 */
#include "../src/sn_pack.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  include <direct.h>
#  include <io.h>
#  define MKDIR(p) _mkdir(p)
#else
#  include <sys/stat.h>
#  include <unistd.h>
#  define MKDIR(p) mkdir((p), 0700)
#endif

static int passes, fails;
#define CHECK(c, ...) do { if (c) { passes++; printf("PASS "); } else { fails++; printf("FAIL "); } printf(__VA_ARGS__); printf("\n"); } while (0)

static int copy_file(const char *src, const char *dst)
{
    FILE *in = fopen(src, "rb"), *out = fopen(dst, "wb");
    if (!in || !out) { if (in) fclose(in); if (out) fclose(out); return -1; }
    char buf[1 << 16]; size_t n;
    int rc = 0;
    while ((n = fread(buf, 1, sizeof buf, in)) > 0)
        if (fwrite(buf, 1, n, out) != n) { rc = -1; break; }
    if (fclose(in) || fclose(out)) rc = -1;
    return rc;
}

static char *slurp(const char *path, size_t *n)
{
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f); if (sz < 0) { fclose(f); return NULL; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    char *b = (char *)malloc((size_t)sz + 1);
    if (!b || fread(b, 1, (size_t)sz, f) != (size_t)sz) { free(b); fclose(f); return NULL; }
    b[sz] = 0; fclose(f); if (n) *n = (size_t)sz; return b;
}

static int write_all(const char *path, const char *s, size_t n)
{
    FILE *f = fopen(path, "wb"); if (!f) return -1;
    int rc = fwrite(s, 1, n, f) == n ? 0 : -1;
    if (fclose(f)) rc = -1;
    return rc;
}

/* Mutate a copy of the manifest. kind_from/kind_to rewrite a kind token;
 * blk_from/blk_to rewrite a block-size field (0 = leave);
 * end_mode: 0 = keep end, 1 = drop end, 2 = gate lines before end,
 * 3 = gate lines after end. */
static int write_manifest(const char *src, const char *dst,
                          const char *kind_from, const char *kind_to,
                          const char *blk_from, const char *blk_to, int end_mode)
{
    size_t n = 0; char *s = slurp(src, &n); if (!s) return -1;
    FILE *f = fopen(dst, "wb");
    if (!f) { free(s); return -1; }
    int rc = 0, seen_end = 0, rewrote_kind = 0, rewrote_blk = 0;
    char *cur = s;
    while (*cur) {
        char *nl = strchr(cur, '\n');
        size_t ln = nl ? (size_t)(nl - cur + 1) : strlen(cur);
        char line[4096];
        if (ln >= sizeof line) { rc = -1; break; }
        memcpy(line, cur, ln); line[ln] = 0;
        cur += ln;

        char trim[4096]; memcpy(trim, line, ln + 1);
        size_t t = strlen(trim);
        while (t && (trim[t - 1] == '\n' || trim[t - 1] == '\r')) trim[--t] = 0;

        if (strcmp(trim, "end") == 0) {
            if (end_mode == 1) { seen_end = 1; continue; } /* drop */
            if (end_mode == 2) fputs("gate 1\ngate-gen 00\n", f);
            fputs(line, f);
            if (end_mode == 3) fputs("gate 1\ngate-gen 00\n", f);
            seen_end = 1;
            continue;
        }
        if (kind_from && !rewrote_kind && strncmp(trim, "tensor ", 7) == 0
            && strstr(trim, kind_from)) {
            /* replace the last token if it is kind_from */
            char *sp = strrchr(trim, ' ');
            if (sp && strcmp(sp + 1, kind_from) == 0) {
                fprintf(f, "%.*s%s\n", (int)(sp - trim + 1), trim, kind_to);
                rewrote_kind = 1;
                continue;
            }
        }
        if (blk_from && !rewrote_blk && strncmp(trim, "tensor ", 7) == 0) {
            /* fields: tensor name b0 nbl bk digest digest kind */
            char name[1024], kind[16];
            unsigned long long b0, nbl, d1, d2; unsigned bk;
            if (sscanf(trim, "tensor %1023s %llu %llu %u %llx %llx %15s",
                       name, &b0, &nbl, &bk, &d1, &d2, kind) == 7
                && bk == (unsigned)strtoul(blk_from, NULL, 10)) {
                fprintf(f, "tensor %s %llu %llu %s %016llx %016llx %s\n",
                        name, b0, nbl, blk_to, d1, d2, kind);
                rewrote_blk = 1;
                continue;
            }
        }
        fputs(line, f);
    }
    if (fclose(f)) rc = -1;
    free(s);
    (void)seen_end; (void)rewrote_kind; (void)rewrote_blk;
    return rc;
}

static int try_open(const char *prefix, const char *shapes, int want_ok, const char *label)
{
    sn_pack p;
    int rc = sn_pack_open(&p, prefix, shapes);
    if (rc == 0) sn_pack_close(&p);
    else sn_pack_close(&p); /* prefix may have been allocated */
    int ok = want_ok ? rc == 0 : rc != 0;
    CHECK(ok, "%s (rc=%d)", label, rc);
    return ok;
}

static int try_src(const char *prefix, const char *shapes)
{
    FILE *s = fopen(shapes, "rb");
    char stream[1100];
    snprintf(stream, sizeof stream, "%s.nq2", prefix);
    FILE *q = fopen(stream, "rb");
    int ok = s && q;
    if (s) fclose(s);
    if (q) fclose(q);
    return ok ? 0 : -1;
}

int main(int argc, char **argv)
{
    const char *src_prefix = argc > 1 ? argv[1] : NULL;
    const char *cands[] = {
        src_prefix,
        "testdata/tiny/tiny",
        "../testdata/tiny/tiny",
        "D:/scratch/ntse-o2-2026-11-27/testdata/tiny/tiny",
        NULL
    };
    char src_shapes[1100] = "", src_man[1100] = "", src_nq2[1100] = "", src_book[1100] = "";
    int found = 0;
    for (int i = 0; cands[i]; i++) {
        if (!cands[i]) continue;
        snprintf(src_nq2, sizeof src_nq2, "%s.nq2", cands[i]);
        snprintf(src_book, sizeof src_book, "%s.nq2book", cands[i]);
        snprintf(src_man, sizeof src_man, "%s.nq2manifest", cands[i]);
        /* shapes.tsv sits beside the prefix */
        snprintf(src_shapes, sizeof src_shapes, "%s", cands[i]);
        char *slash = strrchr(src_shapes, '/');
        char *bslash = strrchr(src_shapes, '\\');
        if (bslash && (!slash || bslash > slash)) slash = bslash;
        if (slash) { slash[1] = 0; strcat(src_shapes, "shapes.tsv"); }
        else snprintf(src_shapes, sizeof src_shapes, "shapes.tsv");
        if (try_src(cands[i], src_shapes) == 0) { src_prefix = cands[i]; found = 1; break; }
    }
    if (!found) {
        printf("FAIL no tiny pack found (pass a prefix)\n");
        return 1;
    }

    char dir[512], dst_prefix[600], dst_man[650], dst_nq2[650], dst_book[650];
#if defined(_WIN32)
    snprintf(dir, sizeof dir, "%s\\ntse-r7-pack-manifest", getenv("TEMP") ? getenv("TEMP") : ".");
#else
    snprintf(dir, sizeof dir, "/tmp/ntse-r7-pack-manifest");
#endif
    MKDIR(dir);
    snprintf(dst_prefix, sizeof dst_prefix, "%s/host", dir);
    snprintf(dst_nq2, sizeof dst_nq2, "%s.nq2", dst_prefix);
    snprintf(dst_book, sizeof dst_book, "%s.nq2book", dst_prefix);
    snprintf(dst_man, sizeof dst_man, "%s.nq2manifest", dst_prefix);

    CHECK(copy_file(src_nq2, dst_nq2) == 0, "copy stream");
    CHECK(copy_file(src_book, dst_book) == 0, "copy book");
    if (fails) { printf("%d PASS / %d FAIL\n", passes, fails); return 1; }

    /* valid, unmodified end */
    CHECK(copy_file(src_man, dst_man) == 0, "copy valid manifest");
    try_open(dst_prefix, src_shapes, 1, "valid packed+raw manifest with end");

    if (write_manifest(src_man, dst_man, NULL, NULL, NULL, NULL, 2) == 0)
        try_open(dst_prefix, src_shapes, 1, "valid manifest with gate lines before end");
    else CHECK(0, "write gate-before-end manifest");

    if (write_manifest(src_man, dst_man, NULL, NULL, NULL, NULL, 3) == 0)
        try_open(dst_prefix, src_shapes, 1, "valid manifest with gate lines after end");
    else CHECK(0, "write gate-after-end manifest");

    if (write_manifest(src_man, dst_man, "packed", "foo", NULL, NULL, 0) == 0)
        try_open(dst_prefix, src_shapes, 0, "unknown kind is refused");
    else CHECK(0, "write unknown-kind manifest");

    if (write_manifest(src_man, dst_man, NULL, NULL, "256", "128", 0) == 0)
        try_open(dst_prefix, src_shapes, 0, "wrong block size is refused");
    else CHECK(0, "write wrong-block manifest");

    if (write_manifest(src_man, dst_man, NULL, NULL, NULL, NULL, 1) == 0)
        try_open(dst_prefix, src_shapes, 0, "missing trailing end line is refused");
    else CHECK(0, "write missing-end manifest");

    printf("%d PASS / %d FAIL\n", passes, fails);
    return fails ? 1 : 0;
}
