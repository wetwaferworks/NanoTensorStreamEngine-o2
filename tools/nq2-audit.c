/*
 * nq2-audit.c - NanoQuatTwo pack auditor.
 *
 * Applies the two-part-code accounting of Greene, "Compression and
 * Intelligence" (greene.sh, 2026) to a packed model: an honest size
 * claim charges for the data bitstream AND the program that decodes
 * it. For NanoQuatTwo the bitstream is the .nq2 index stream, the
 * program is the .nq2book memory file plus the decoder binary, and
 * the headline figure is the complete description length per weight.
 *
 * Also measures, from the index stream itself:
 *   - the Shannon entropy of the index distribution (bits per index),
 *     which bounds what an entropy-coded stream could save over the
 *     flat 8 bits per index stored today;
 *   - the first-order conditional entropy H(next | prev) within each
 *     tensor segment, which bounds a context-coded stream;
 *   - codebook utilization: an entry only earns its book bytes if
 *     blocks actually select it (Greene's reuse principle). Dead
 *     entries are program bits paying no rent.
 *
 * Pure C99, no dependencies beyond libc and libm.
 * Usage: nq2-audit STREAM.nq2 BOOK.nq2book [DECODER_BINARY]
 *
 * Direction and executive planning: JD (David Rittenhouse).
 * Implementation: E.L.F.I.E.
 */

#define _FILE_OFFSET_BITS 64 /* 64-bit off_t on Windows too (Archie P0-2) */
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

static uint32_t rd_u32(FILE *f)
{
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) { fprintf(stderr, "short read\n"); exit(1); }
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8)
         | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

static uint64_t rd_u64(FILE *f)
{
    unsigned char b[8];
    if (fread(b, 1, 8, f) != 8) { fprintf(stderr, "short read\n"); exit(1); }
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | b[i];
    return v;
}

static uint64_t file_size(FILE *f)
{
    if (fseeko(f, 0, SEEK_END) != 0) { fprintf(stderr, "seek fail\n"); exit(1); }
    off_t s = ftello(f);
    if (s < 0) { fprintf(stderr, "tell fail\n"); exit(1); }
    return (uint64_t)s;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: nq2-audit STREAM.nq2 BOOK.nq2book "
                        "[DECODER_BINARY]\n");
        return 1;
    }
    FILE *sf = fopen(argv[1], "rb");
    if (!sf) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    char magic[4];
    if (fread(magic, 1, 4, sf) != 4 || memcmp(magic, "NQ2\0", 4) != 0) {
        fprintf(stderr, "%s: not a NanoQuatTwo index stream\n", argv[1]);
        return 1;
    }
    uint32_t version = rd_u32(sf);
    uint32_t block = rd_u32(sf);
    uint32_t entries = rd_u32(sf);
    uint64_t total_weights = rd_u64(sf);
    uint64_t n_blocks = rd_u64(sf);
    uint32_t n_books = 1;
    uint64_t *seg_blocks = NULL;
    if (version == 2) {
        n_books = rd_u32(sf);
        seg_blocks = (uint64_t *)malloc(sizeof(uint64_t) * n_books);
        if (!seg_blocks) { fprintf(stderr, "out of memory\n"); return 1; }
        for (uint32_t i = 0; i < n_books; i++) seg_blocks[i] = rd_u64(sf);
    } else if (version != 1) {
        fprintf(stderr, "unsupported format version %u\n", version);
        return 1;
    }
    if (entries == 0 || entries > 256) {
        fprintf(stderr, "entries %u outside the 1-byte index range\n",
                entries);
        return 1;
    }
    uint64_t header_bytes = (uint64_t)ftello(sf);
    uint64_t stream_bytes = file_size(sf);
    /* The stream must contain the complete frame payload
     * (Sol M10): one index byte plus the 4-byte fp16 frame
     * per block, after the header. The audit reads only the
     * index bytes, so a stream truncated inside the frames
     * used to pass while no engine could expand it. */
    if (stream_bytes != header_bytes + n_blocks * 5) {
        fprintf(stderr, "%s: stream is %llu bytes; header plus "
                "%llu blocks of index+frame require %llu\n",
                argv[1], (unsigned long long)stream_bytes,
                (unsigned long long)n_blocks,
                (unsigned long long)(header_bytes + n_blocks * 5));
        return 1;
    }

    FILE *bf = fopen(argv[2], "rb");
    if (!bf) { fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
    uint64_t book_bytes = file_size(bf);
    /* Structural validation of the book (Sol M10): existence
     * used to be the only check, so an unrelated or incomplete
     * file passed the audit and its size was charged against
     * this stream. The book must carry the NQ2B magic, agree
     * with the stream on the whole geometry (version, block,
     * entries, book count... the same fields the engines
     * cross-check in packed_open), and be exactly the size
     * that geometry implies. */
    {
        char bmagic[4];
        uint64_t want_bytes;
        if (book_bytes < 12 || fseeko(bf, 0, SEEK_SET) != 0
                || fread(bmagic, 1, 4, bf) != 4
                || memcmp(bmagic, "NQ2B", 4) != 0) {
            fprintf(stderr, "%s: not a NanoQuatTwo book "
                    "(bad magic)\n", argv[2]);
            return 1;
        }
        if (version == 2) {
            uint32_t bver = rd_u32(bf);
            uint32_t b_block = rd_u32(bf);
            uint32_t b_entries = rd_u32(bf);
            uint32_t b_books = rd_u32(bf);
            if (bver != 2 || b_block != block
                    || b_entries != entries || b_books != n_books) {
                fprintf(stderr, "%s: book header (v%u, block "
                        "%u, entries %u, books %u) disagrees "
                        "with the stream (v%u, block %u, "
                        "entries %u, books %u)\n",
                        argv[2], bver, b_block, b_entries,
                        b_books, version, block, entries, n_books);
                return 1;
            }
            want_bytes = 20 + (uint64_t)n_books * entries
                       * block * 4;
        } else {
            uint32_t b_block = rd_u32(bf);
            uint32_t b_entries = rd_u32(bf);
            if (b_block != block || b_entries != entries) {
                fprintf(stderr, "%s: book header (block %u, "
                        "entries %u) disagrees with the stream "
                        "(block %u, entries %u)\n",
                        argv[2], b_block, b_entries, block,
                        entries);
                return 1;
            }
            want_bytes = 12 + (uint64_t)entries * block * 4;
        }
        if (book_bytes != want_bytes) {
            fprintf(stderr, "%s: book is %llu bytes; its "
                    "geometry requires %llu\n",
                    argv[2], (unsigned long long)book_bytes,
                    (unsigned long long)want_bytes);
            return 1;
        }
    }
    fclose(bf);

    uint64_t prog_bytes = 0;
    if (argc >= 4) {
        FILE *pf = fopen(argv[3], "rb");
        if (!pf) { fprintf(stderr, "cannot open %s\n", argv[3]); return 1; }
        prog_bytes = file_size(pf);
        fclose(pf);
    }

    /* Per-book index histograms and the global first-order pair
     * counts. Pairs never cross a segment boundary: the previous
     * block of another tensor is not context. */
    uint64_t *hist = (uint64_t *)calloc((size_t)n_books * entries,
                                        sizeof(uint64_t));
    uint64_t *pairs = (uint64_t *)calloc((size_t)entries * entries,
                                         sizeof(uint64_t));
    if (!hist || !pairs) { fprintf(stderr, "out of memory\n"); return 1; }

    if (fseeko(sf, (off_t)header_bytes, SEEK_SET) != 0) {
        fprintf(stderr, "seek fail\n"); return 1;
    }
    unsigned char buf[1 << 20];
    for (uint32_t seg = 0; seg < n_books; seg++) {
        uint64_t left = (version == 2) ? seg_blocks[seg] : n_blocks;
        int prev = -1;
        while (left > 0) {
            size_t want = left > sizeof buf ? sizeof buf : (size_t)left;
            size_t got = fread(buf, 1, want, sf);
            if (got == 0) { fprintf(stderr, "short stream\n"); return 1; }
            for (size_t i = 0; i < got; i++) {
                unsigned idx = buf[i];
                /* Stream bytes run 0..255 but the book may hold
                 * fewer entries; an unchecked idx writes outside
                 * hist/pairs (Archie P0-4). */
                if (idx >= entries) {
                    fprintf(stderr, "stream index %u outside book "
                            "of %u entries\n", idx, entries);
                    return 1;
                }
                hist[(size_t)seg * entries + idx]++;
                if (prev >= 0) pairs[(size_t)prev * entries + idx]++;
                prev = (int)idx;
            }
            left -= got;
        }
    }
    fclose(sf);

    /* ---- entropies ---- */
    double h0 = 0.0;
    uint64_t npairs = 0;
    for (uint32_t e = 0; e < entries; e++) {
        uint64_t c = 0;
        for (uint32_t s = 0; s < n_books; s++)
            c += hist[(size_t)s * entries + e];
        if (c > 0) {
            double p = (double)c / (double)n_blocks;
            h0 -= p * log2(p);
        }
    }
    double hpair = 0.0, hprev = 0.0;
    for (uint32_t a = 0; a < entries; a++)
        for (uint32_t b = 0; b < entries; b++)
            npairs += pairs[(size_t)a * entries + b];
    for (uint32_t a = 0; a < entries; a++) {
        uint64_t rowa = 0;
        for (uint32_t b = 0; b < entries; b++) rowa += pairs[(size_t)a * entries + b];
        if (rowa == 0) continue;
        double pa = (double)rowa / (double)npairs;
        hprev -= pa * log2(pa);
        for (uint32_t b = 0; b < entries; b++) {
            uint64_t c = pairs[(size_t)a * entries + b];
            if (c == 0) continue;
            double pab = (double)c / (double)npairs;
            hpair -= pab * log2(pab);
        }
    }
    double h1 = npairs > 0 ? hpair - hprev : 0.0;

    /* ---- utilization ---- */
    uint64_t used_total = 0, dead_total = 0;
    uint32_t books_with_dead = 0, min_used = entries, max_used = 0;
    for (uint32_t s = 0; s < n_books; s++) {
        uint32_t used = 0;
        for (uint32_t e = 0; e < entries; e++)
            if (hist[(size_t)s * entries + e] > 0) used++;
        used_total += used;
        dead_total += entries - used;
        if (used < entries) books_with_dead++;
        if (used < min_used) min_used = used;
        if (used > max_used) max_used = used;
    }

    /* ---- the two-part accounting ---- */
    double w = total_weights > 0 ? (double)total_weights : 1.0;
    double stream_bpw = 8.0 * (double)stream_bytes / w;
    double book_bpw = 8.0 * (double)book_bytes / w;
    double complete_bpw = 8.0 * (double)(stream_bytes + book_bytes
                                         + prog_bytes) / w;
    /* Entropy-coded stream estimates: header and the fp16 min/scale
     * frame (4 bytes per block) stay; only the index bytes shrink. */
    double est_h0 = (double)header_bytes
                  + (double)n_blocks * (h0 / 8.0 + 4.0);
    double est_h1 = (double)header_bytes
                  + (double)n_blocks * (h1 / 8.0 + 4.0);

    printf("NanoQuatTwo pack audit (two-part accounting)\n");
    printf("format v%u, block %u, entries %u, books %u\n",
           version, block, entries, n_books);
    printf("weights %llu in %llu blocks\n",
           (unsigned long long)total_weights,
           (unsigned long long)n_blocks);
    printf("\n-- sizes (the program is charged, per Greene) --\n");
    printf("stream (data bitstream):  %12llu bytes  %.5f bits/weight\n",
           (unsigned long long)stream_bytes, stream_bpw);
    printf("book   (program):         %12llu bytes  %.5f bits/weight\n",
           (unsigned long long)book_bytes, book_bpw);
    if (argc >= 4)
        printf("decoder (program):        %12llu bytes  %.5f bits/weight\n",
               (unsigned long long)prog_bytes,
               8.0 * (double)prog_bytes / w);
    printf("COMPLETE description:     %12llu bytes  %.5f bits/weight\n",
           (unsigned long long)(stream_bytes + book_bytes + prog_bytes),
           complete_bpw);
    printf("frame cost inside stream: %.5f bits/weight "
           "(fp16 min+scale, 4 of 5 bytes per block)\n",
           32.0 / (double)block);
    printf("\n-- index entropy (flat storage today: 8 bits/index) --\n");
    printf("H0  marginal:    %.4f bits/index\n", h0);
    printf("H1  cond(prev):  %.4f bits/index over %llu pairs\n",
           h1, (unsigned long long)npairs);
    printf("stream at H0:    %12.0f bytes  %.5f bits/weight complete*\n",
           est_h0, 8.0 * (est_h0 + (double)book_bytes
                        + (double)prog_bytes) / w);
    printf("stream at H1:    %12.0f bytes  %.5f bits/weight complete*\n",
           est_h1, 8.0 * (est_h1 + (double)book_bytes
                        + (double)prog_bytes) / w);
    printf("(* estimates: index bytes only, arithmetic coding "
           "overhead excluded)\n");
    printf("\n-- codebook utilization (reuse pays the rent) --\n");
    printf("entries used: %llu of %llu across %u book(s); dead %llu\n",
           (unsigned long long)used_total,
           (unsigned long long)((uint64_t)entries * n_books),
           n_books, (unsigned long long)dead_total);
    if (n_books > 1)
        printf("per book: min used %u, max used %u, "
               "books with dead entries %u\n",
               min_used, max_used, books_with_dead);

    free(hist);
    free(pairs);
    free(seg_blocks);
    return 0;
}
