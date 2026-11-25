/* nq2-run.c ... NanoQuatTwo streaming runner.
 *
 * Implemented by Elfie (E.L.F.I.E.): the streaming expander and
 * this code. Direction and executive planning by JD.
 *
 * Loads a nanoquantized model (.nq2 index stream + .nq2book
 * memory file) and streams it back open, megatexture style:
 * only a small window of blocks is ever expanded at once, so
 * resident memory is the window plus the book, never the model.
 * Expansion is exact: the same index always calls back the same
 * prototype, so the checksum of the streamed expansion equals
 * the checksum of a full expansion, at a fraction of the RAM.
 *
 * Usage:
 *   nq2-run model.nq2 model.nq2book           stream + report
 *   nq2-run model.nq2 model.nq2book peek N    expand only block N
 *
 * Plain hosted C99, no dependencies. Build with build.sh (Linux)
 * or build.bat (Windows, MinGW gcc).
 */

#define _POSIX_C_SOURCE 200809L   /* fseeko/ftello declarations */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif

/* 64-bit seek/tell. Plain fseek/ftell take a long, which is
 * 32 bits wide on Windows; stream files are small today, but
 * the format does not promise to stay that way. */
static int seek_to(FILE *f, uint64_t off)
{
#ifdef _WIN32
    return _fseeki64(f, (long long)off, SEEK_SET);
#else
    return fseeko(f, (off_t)off, SEEK_SET);
#endif
}

static uint64_t tell_of(FILE *f)
{
#ifdef _WIN32
    return (uint64_t)_ftelli64(f);
#else
    return (uint64_t)ftello(f);
#endif
}

static float half_to_float(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            /* subnormal: normalize by hand */
            exp = 1;
            while (!(mant & 0x400u)) { mant <<= 1; exp--; }
            mant &= 0x3ffu;
            bits = sign | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static uint32_t rd_u32(FILE *f)
{
    uint8_t b[4];
    if (fread(b, 1, 4, f) != 4) { fprintf(stderr, "short read\n"); exit(1); }
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) |
           ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

static uint64_t rd_u64(FILE *f)
{
    uint64_t lo = rd_u32(f), hi = rd_u32(f);
    return lo | (hi << 32);
}

static double now_sec(void)
{
    /* Wall time, not CPU time: the rates this prints are meant
     * to describe the disc-to-memory trip a user would see. */
#ifdef _WIN32
    LARGE_INTEGER fq, cn;
    QueryPerformanceFrequency(&fq);
    QueryPerformanceCounter(&cn);
    return (double)cn.QuadPart / (double)fq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: nq2-run model.nq2 model.nq2book [peek N]\n");
        return 1;
    }
    const char *stream_path = argv[1];
    const char *book_path = argv[2];
    long peek = -1;
    if (argc >= 5 && strcmp(argv[3], "peek") == 0)
        peek = strtol(argv[4], NULL, 10);

    /* ---- the index stream header first: its format version
     * decides how the memory file is laid out ---- */
    FILE *sf = fopen(stream_path, "rb");
    if (!sf) { fprintf(stderr, "cannot open %s\n", stream_path); return 1; }
    char magic[4];
    if (fread(magic, 1, 4, sf) != 4 || memcmp(magic, "NQ2\0", 4) != 0) {
        fprintf(stderr, "%s: not a NanoQuatTwo index stream\n", stream_path);
        return 1;
    }
    uint32_t version = rd_u32(sf);
    uint32_t block = rd_u32(sf);
    uint32_t entries = rd_u32(sf);
    uint64_t total_weights = rd_u64(sf);
    uint64_t n_blocks = rd_u64(sf);
    /* T2 hardening (CODE-REVIEW-2026-11-24): validate the pack
     * geometry before any division or allocation trusts it. */
    if (block == 0 || block > 4096 || entries == 0 || entries > 256) {
        fprintf(stderr, "%s: unsupported pack geometry "
                "(block %u, entries %u)\n", stream_path, block, entries);
        return 1;
    }
    uint32_t n_books = 1;
    uint64_t *seg_blocks = NULL;   /* v2 only: blocks per tensor */
    if (version == 2) {
        n_books = rd_u32(sf);
        seg_blocks = (uint64_t *)malloc(sizeof(uint64_t) * n_books);
        if (!seg_blocks) { fprintf(stderr, "out of memory\n"); return 1; }
        for (uint32_t i = 0; i < n_books; i++) seg_blocks[i] = rd_u64(sf);
    } else if (version != 1) {
        fprintf(stderr, "unsupported format version %u\n", version);
        return 1;
    }
    uint64_t header_bytes = tell_of(sf);

    /* ---- the memory file: small enough to keep resident ---- */
    FILE *bf = fopen(book_path, "rb");
    if (!bf) { fprintf(stderr, "cannot open %s\n", book_path); return 1; }
    if (fread(magic, 1, 4, bf) != 4 || memcmp(magic, "NQ2B", 4) != 0) {
        fprintf(stderr, "%s: not a NanoQuatTwo memory file\n", book_path);
        return 1;
    }
    if (version == 2) {
        uint32_t bver = rd_u32(bf);
        uint32_t b_block = rd_u32(bf);
        uint32_t b_entries = rd_u32(bf);
        uint32_t b_books = rd_u32(bf);
        if (bver != 2 || b_block != block || b_entries != entries
                || b_books != n_books) {
            fprintf(stderr, "stream and book disagree\n");
            return 1;
        }
    } else {
        uint32_t b_block = rd_u32(bf);
        uint32_t b_entries = rd_u32(bf);
        if (b_block != block || b_entries != entries) {
            fprintf(stderr, "stream and book disagree (block %u vs %u)\n",
                    b_block, block);
            return 1;
        }
    }
    float *book = (float *)malloc((size_t)n_books * entries * block
                                  * sizeof(float));
    if (!book) { fprintf(stderr, "out of memory\n"); return 1; }
    if (fread(book, sizeof(float), (size_t)n_books * entries * block, bf)
            != (size_t)n_books * entries * block) {
        fprintf(stderr, "%s: short book\n", book_path);
        return 1;
    }
    fclose(bf);
    size_t book_bytes = (version == 2 ? 20 : 12)
                        + (size_t)n_books * entries * block * sizeof(float);
    fseek(sf, 0, SEEK_END);
    uint64_t stream_bytes = tell_of(sf);
    seek_to(sf, header_bytes);

    printf("NanoQuatTwo streaming runner\n");
    printf("model format version: %u\n", version);
    printf("weights: %llu in %llu blocks of %u\n",
           (unsigned long long)total_weights,
           (unsigned long long)n_blocks, block);
    printf("index stream: %llu bytes; memory file: %zu bytes\n",
           (unsigned long long)stream_bytes, book_bytes);
    if (version == 2)
        printf("books: %u (one per tensor)\n", n_books);
    printf("expanded size: %llu bytes (never held in memory)\n",
           (unsigned long long)total_weights * 4ull);

    if (peek >= 0) {
        /* Random access: expand exactly one block, like paging in
         * exactly one texture tile. Seek math: indexes start at
         * header_bytes; mins follow the indexes; scales follow mins. */
        if ((uint64_t)peek >= n_blocks) {
            fprintf(stderr, "block %ld out of range\n", peek);
            return 1;
        }
        uint8_t idx;
        uint16_t mn_h, sc_h;
        seek_to(sf, header_bytes + (uint64_t)peek);
        if (fread(&idx, 1, 1, sf) != 1) return 1;
        seek_to(sf, header_bytes + n_blocks + 2 * (uint64_t)peek);
        if (fread(&mn_h, 2, 1, sf) != 1) return 1;
        seek_to(sf, header_bytes + 3 * n_blocks + 2 * (uint64_t)peek);
        if (fread(&sc_h, 2, 1, sf) != 1) return 1;
        if (idx >= entries) {
            fprintf(stderr, "index %u out of range (entries %u)\n",
                    idx, entries);
            return 1;
        }
        float mn = half_to_float(mn_h), sc = half_to_float(sc_h);
        const float *book_base = book;
        if (version == 2) {
            /* find which tensor's segment this block lives in */
            uint64_t acc = 0;
            uint32_t seg = 0;
            while (seg + 1 < n_books
                    && (uint64_t)peek >= acc + seg_blocks[seg]) {
                acc += seg_blocks[seg];
                seg++;
            }
            book_base = book + (size_t)seg * entries * block;
            printf("block %ld is in tensor segment %u of %u\n",
                   peek, seg, n_books);
        }
        printf("block %ld: index %u, min %.6f, scale %.6f\n",
               peek, idx, mn, sc);
        printf("first weights:");
        for (uint32_t i = 0; i < 8 && i < block; i++)
            printf(" %.6f", mn + sc * book_base[(size_t)idx * block + i]);
        printf("\n");
        fclose(sf);
        free(book);
        return 0;
    }

    /* ---- stream the whole model through a small window ---- */
    /* Window: about 4 MB of expanded weights, however many blocks
     * that is at this block size. */
    uint64_t win_blocks = (1ull << 20) / block;
    if (win_blocks == 0) win_blocks = 1;
    if (win_blocks > n_blocks) win_blocks = n_blocks;
    uint8_t *idx_buf = (uint8_t *)malloc((size_t)win_blocks);
    uint16_t *mn_buf = (uint16_t *)malloc((size_t)win_blocks * 2);
    uint16_t *sc_buf = (uint16_t *)malloc((size_t)win_blocks * 2);
    float *win = (float *)malloc((size_t)win_blocks * block * sizeof(float));
    if (!idx_buf || !mn_buf || !sc_buf || !win) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }

    /* The three stream sections are stored back to back; read them
     * in window-sized passes using three file positions. */
    uint64_t idx_pos = header_bytes;
    uint64_t mn_pos = header_bytes + n_blocks;
    uint64_t sc_pos = header_bytes + 3 * n_blocks;

    uint64_t hash = 1469598103934665603ull;   /* FNV-1a 64 */
    uint64_t done = 0;
    uint32_t seg = 0;
    uint64_t seg_left = version == 2 ? seg_blocks[0] : n_blocks;
    /* Expansion and hashing are timed separately: the hash is a
     * verification tax paid once, while the expansion loop is the
     * streaming path (disc to memory to CPU) that runs every load.
     * Reporting one blended rate hides which is which. */
    double t0 = now_sec();
    double t_expand = 0.0, t_hash = 0.0;
    while (done < n_blocks) {
        uint64_t want = n_blocks - done;
        if (want > win_blocks) want = win_blocks;
        seek_to(sf, idx_pos + done);
        if (fread(idx_buf, 1, (size_t)want, sf) != (size_t)want) break;
        seek_to(sf, mn_pos + 2 * done);
        if (fread(mn_buf, 2, (size_t)want, sf) != (size_t)want) break;
        seek_to(sf, sc_pos + 2 * done);
        if (fread(sc_buf, 2, (size_t)want, sf) != (size_t)want) break;
        for (uint64_t b = 0; b < want; b++) {
            if (idx_buf[b] >= entries) {
                fprintf(stderr, "index %u out of range (entries %u) "
                        "at block %llu\n", idx_buf[b], entries,
                        (unsigned long long)(done + b));
                return 1;
            }
            float mn = half_to_float(mn_buf[b]);
            float sc = half_to_float(sc_buf[b]);
            if (version == 2) {
                /* advance across tensor segments as blocks pass;
                 * each segment expands from its own book. The
                 * advance is bounded at the last segment so a
                 * corrupt tensor table cannot walk off the array. */
                while (seg_left == 0 && seg + 1 < n_books) {
                    seg++;
                    seg_left = seg_blocks[seg];
                }
                if (seg_left > 0) seg_left--;
            }
            const float *book_base = book
                + (size_t)(version == 2 ? seg : 0) * entries * block;
            const float *proto = book_base + (size_t)idx_buf[b] * block;
            float *dst = win + (size_t)b * block;
            const float *restrict src = proto;
            float *restrict out = dst;
            for (uint32_t i = 0; i < block; i++)
                out[i] = mn + sc * src[i];
        }
        t_expand += now_sec() - t0; t0 = now_sec();
        const uint8_t *bytes = (const uint8_t *)win;
        size_t nbytes = (size_t)want * block * sizeof(float);
        for (size_t i = 0; i < nbytes; i++) {
            hash ^= bytes[i];
            hash *= 1099511628211ull;
        }
        t_hash += now_sec() - t0; t0 = now_sec();
        done += want;
    }
    double dt = t_expand + t_hash;
    fclose(sf);
    if (done != n_blocks)
        fprintf(stderr, "warning: stream truncated, expanded %llu "
                "of %llu blocks; checksum covers the partial "
                "expansion only\n",
                (unsigned long long)done,
                (unsigned long long)n_blocks);

    size_t window_bytes = (size_t)win_blocks * block * sizeof(float);
    size_t peak = book_bytes + window_bytes +
                  (size_t)win_blocks * 5;
    printf("streamed %llu blocks through a %llu-block window\n",
           (unsigned long long)done, (unsigned long long)win_blocks);
    printf("peak resident: %zu bytes (book + window + stream buffers)\n",
           peak);
    printf("expanded checksum (FNV-1a 64): %016llx\n",
           (unsigned long long)hash);
    if (dt > 0)
        printf("expansion rate: %.1f M weights/sec (%.1f MB/sec expanded)\n",
               total_weights / dt / 1e6,
               total_weights * 4.0 / dt / 1e6);
    if (t_expand > 0)
        printf("streaming path (read + reconstruct): %.1f M weights/sec "
               "(%.1f MB/sec)\n",
               total_weights / t_expand / 1e6,
               total_weights * 4.0 / t_expand / 1e6);
    if (t_hash > 0)
        printf("checksum pass alone: %.1f M weights/sec\n",
               total_weights / t_hash / 1e6);
    printf("compression: %.4f bits per weight (stream), "
           "book amortized separately\n",
           stream_bytes * 8.0 / (double)total_weights);
    free(idx_buf); free(mn_buf); free(sc_buf); free(win); free(book);
    return 0;
}
