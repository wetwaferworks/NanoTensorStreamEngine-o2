/* mk_fixtures.c - build the checkpoint fixtures that test_hostile and
 * test_coherence need, without shipping any weights.
 *
 *     mk_fixtures <checkpoint_dir> <out_dir>
 *
 * Writes under <out_dir>:
 *   hostile/refuse-*   tiny crafted safetensors files; opening any of them
 *                      must fail cleanly (test_hostile refuse <dir>)
 *   hostile/i32        a valid file holding one I32 tensor; the checkpoint
 *                      opens but load_f32 must refuse it (test_hostile i32)
 *   dtype              the embedding table is declared with the right shape
 *                      and dtype, and layer 0's norm is I32 (test_coherence's
 *                      second argument). The embedding payload is a sparse
 *                      hole, so the file takes almost no disc space and holds
 *                      no model data.
 *   partial            hard links (or symlinks) to the first two shards of
 *                      <checkpoint_dir>, taken in name order. They stand for a
 *                      checkpoint with layers 0..32 only (test_coherence's
 *                      first argument). Nothing is copied.
 *
 * The output directory holds only synthetic files and links, so it can be
 * deleted at any time. Keep it outside the repository.
 *
 * Idea, shape and direction: JD. Co-implementation: Cortana. */
#include <dirent.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#include <windows.h>
#define MKDIR(p) _mkdir(p)
#else
#include <unistd.h>
#define MKDIR(p) mkdir((p), 0777)
#endif

#include "../src/forward.h" /* NEMO_VOCAB, NEMO_HIDDEN */

static void make_dir(const char *path) {
    if (MKDIR(path) != 0 && errno != EEXIST) {
        fprintf(stderr, "cannot create %s\n", path);
        exit(2);
    }
}

static void put_u64le(unsigned char *b, uint64_t v) {
    for (int i = 0; i < 8; i++) b[i] = (unsigned char)(v >> (8 * i));
}

/* Write a safetensors-style file: an 8-byte little-endian header length,
 * the header text, then payload_len zero bytes. `declared_hlen` is what
 * the length field claims, so a case can lie about it. */
static void write_shard(const char *path, const char *header,
                        uint64_t declared_hlen, uint64_t payload_len) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "cannot write %s\n", path);
        exit(2);
    }
    unsigned char len[8];
    put_u64le(len, declared_hlen);
    fwrite(len, 1, 8, f);
    fwrite(header, 1, strlen(header), f);
    for (uint64_t i = 0; i < payload_len; i++) fputc(0, f);
    fclose(f);
}

/* One hostile case: <out>/hostile/<name>/model.safetensors. */
static void refuse_case(const char *out, const char *name, const char *header,
                        uint64_t declared_hlen, uint64_t payload_len) {
    char dir[1024], path[1100];
    snprintf(dir, sizeof dir, "%s/hostile/%s", out, name);
    make_dir(dir);
    snprintf(path, sizeof path, "%s/model.safetensors", dir);
    write_shard(path, header, declared_hlen, payload_len);
}

static void make_hostile(const char *out) {
    char dir[1024];
    snprintf(dir, sizeof dir, "%s/hostile", out);
    make_dir(dir);

    /* The header claims more bytes than the file holds. */
    {
        const char *h = "{\"t\":{\"dtype\":\"F32\",\"shape\":[4],"
                        "\"data_offsets\":[0,16]}}";
        refuse_case(out, "refuse-short-header", h, strlen(h) + 4096, 16);
    }
    /* The header length is zero. */
    refuse_case(out, "refuse-zero-header", "{}", 0, 0);
    /* The header length is far past the 256 MiB cap. */
    refuse_case(out, "refuse-huge-header", "{}", (uint64_t)1 << 40, 0);
    /* The header is cut off in the middle of a tensor name. */
    {
        const char *h = "{\"tensor_name_never_closed";
        refuse_case(out, "refuse-truncated-name", h, strlen(h), 16);
    }
    /* A negative dimension must not wrap to a huge unsigned size. */
    {
        const char *h = "{\"t\":{\"dtype\":\"F32\",\"shape\":[-1],"
                        "\"data_offsets\":[0,16]}}";
        refuse_case(out, "refuse-negative-shape", h, strlen(h), 16);
    }
    /* The element count times the element size wraps past 2^64. */
    {
        const char *h = "{\"t\":{\"dtype\":\"F32\",\"shape\":"
                        "[4611686018427387904],\"data_offsets\":[0,0]}}";
        refuse_case(out, "refuse-size-wrap", h, strlen(h), 16);
    }
    /* The end offset is before the start offset. */
    {
        const char *h = "{\"t\":{\"dtype\":\"F32\",\"shape\":[4],"
                        "\"data_offsets\":[16,0]}}";
        refuse_case(out, "refuse-reversed-offsets", h, strlen(h), 16);
    }
    /* The offsets disagree with shape times element size. */
    {
        const char *h = "{\"t\":{\"dtype\":\"F32\",\"shape\":[4],"
                        "\"data_offsets\":[0,8]}}";
        refuse_case(out, "refuse-span-mismatch", h, strlen(h), 16);
    }
    /* A truncated download: the header is intact, the payload is short. */
    {
        const char *h = "{\"t\":{\"dtype\":\"F32\",\"shape\":[1024],"
                        "\"data_offsets\":[0,4096]}}";
        refuse_case(out, "refuse-short-payload", h, strlen(h), 16);
    }
    /* A dtype the loader does not know. */
    {
        const char *h = "{\"t\":{\"dtype\":\"Q99\",\"shape\":[4],"
                        "\"data_offsets\":[0,16]}}";
        refuse_case(out, "refuse-unknown-dtype", h, strlen(h), 16);
    }
    /* A valid file with an I32 tensor: opens, but load_f32 must refuse. */
    {
        const char *h = "{\"t\":{\"dtype\":\"I32\",\"shape\":[4],"
                        "\"data_offsets\":[0,16]}}";
        char d[1024], p[1100];
        snprintf(d, sizeof d, "%s/hostile/i32", out);
        make_dir(d);
        snprintf(p, sizeof p, "%s/model.safetensors", d);
        write_shard(p, h, strlen(h), 16);
    }
}

/* Create `path` as a file of `size` bytes whose body is a hole where the
 * file system supports it (NTFS sparse flag on Windows; any POSIX file
 * system). Returns 0 on success. */
static int make_sparse(const char *path, uint64_t size) {
#ifdef _WIN32
    HANDLE h = CreateFileA(path, GENERIC_READ | GENERIC_WRITE, 0, NULL,
                           CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (h == INVALID_HANDLE_VALUE) return -1;
    DWORD got = 0;
    DeviceIoControl(h, FSCTL_SET_SPARSE, NULL, 0, NULL, 0, &got, NULL);
    LARGE_INTEGER to;
    to.QuadPart = (LONGLONG)size;
    int ok = SetFilePointerEx(h, to, NULL, FILE_BEGIN) && SetEndOfFile(h);
    CloseHandle(h);
    return ok ? 0 : -1;
#else
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    int ok = fseeko(f, (off_t)size - 1, SEEK_SET) == 0 && fputc(0, f) != EOF;
    fclose(f);
    return ok ? 0 : -1;
#endif
}

static void make_dtype(const char *out) {
    char dir[1024], path[1100], header[512];
    snprintf(dir, sizeof dir, "%s/dtype", out);
    make_dir(dir);
    snprintf(path, sizeof path, "%s/model.safetensors", dir);

    uint64_t emb = (uint64_t)NEMO_VOCAB * NEMO_HIDDEN * 2; /* bf16 */
    uint64_t norm = (uint64_t)NEMO_HIDDEN * 4;             /* I32 */
    snprintf(header, sizeof header,
             "{\"backbone.embeddings.weight\":{\"dtype\":\"BF16\","
             "\"shape\":[%d,%d],\"data_offsets\":[0,%llu]},"
             "\"backbone.layers.0.norm.weight\":{\"dtype\":\"I32\","
             "\"shape\":[%d],\"data_offsets\":[%llu,%llu]}}",
             NEMO_VOCAB, NEMO_HIDDEN, (unsigned long long)emb, NEMO_HIDDEN,
             (unsigned long long)emb, (unsigned long long)(emb + norm));
    uint64_t hlen = strlen(header);
    if (make_sparse(path, 8 + hlen + emb + norm) != 0) {
        fprintf(stderr, "cannot create %s\n", path);
        exit(2);
    }
    FILE *f = fopen(path, "r+b");
    if (!f) {
        fprintf(stderr, "cannot reopen %s\n", path);
        exit(2);
    }
    unsigned char len[8];
    put_u64le(len, hlen);
    fwrite(len, 1, 8, f);
    fwrite(header, 1, hlen, f);
    fclose(f);
}

static int cmp_name(const void *a, const void *b) {
    return strcmp(*(char *const *)a, *(char *const *)b);
}

static int link_file(const char *src, const char *dst) {
#ifdef _WIN32
    /* Hard links need the same volume; there is no unprivileged
     * symlink fallback on Windows. */
    return CreateHardLinkA(dst, src, NULL) ? 0 : -1;
#else
    if (link(src, dst) == 0) return 0;
    return symlink(src, dst);
#endif
}

/* Link the first two shards (by name) into <out>/partial. */
static void make_partial(const char *ckpt, const char *out) {
    char dir[1024];
    snprintf(dir, sizeof dir, "%s/partial", out);
    make_dir(dir);

    DIR *d = opendir(ckpt);
    if (!d) {
        fprintf(stderr, "cannot open checkpoint dir %s\n", ckpt);
        exit(2);
    }
    char *names[64];
    int n = 0;
    struct dirent *e;
    while ((e = readdir(d)) != NULL && n < 64) {
        size_t l = strlen(e->d_name);
        if (l > 12 && !strcmp(e->d_name + l - 12, ".safetensors"))
            names[n++] = strdup(e->d_name);
    }
    closedir(d);
    if (n < 3) {
        fprintf(stderr, "need at least 3 shards in %s to build a partial "
                        "checkpoint, found %d\n", ckpt, n);
        exit(2);
    }
    qsort(names, (size_t)n, sizeof names[0], cmp_name);
    for (int i = 0; i < 2; i++) {
        char src[1100], dst[1100];
        snprintf(src, sizeof src, "%s/%s", ckpt, names[i]);
        snprintf(dst, sizeof dst, "%s/%s", dir, names[i]);
        remove(dst);
        if (link_file(src, dst) != 0) {
            fprintf(stderr, "cannot link %s -> %s (hard links need the same "
                            "volume as the checkpoint)\n", dst, src);
            exit(2);
        }
    }
    for (int i = 0; i < n; i++) free(names[i]);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: mk_fixtures <checkpoint_dir> <out_dir>\n");
        return 2;
    }
    make_dir(argv[2]);
    make_hostile(argv[2]);
    make_dtype(argv[2]);
    make_partial(argv[1], argv[2]);
    printf("fixtures written under %s\n", argv[2]);
    return 0;
}
