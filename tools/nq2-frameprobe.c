/* nq2-frameprobe.c ... NanoQuatTwo frame profiler.
 *
 * Implemented by Elfie (E.L.F.I.E.). Direction and executive
 * planning by JD. Built for the frame-coding study proposed in
 * Gizmo's engineering handoff of 2026-11-23 (section 5): before
 * anyone designs a frame coder, measure the frame streams of a
 * real pack... their byte share, their value distributions, and
 * the entropy of the residuals that candidate codings would
 * actually store (adjacent deltas for minima, log-domain deltas
 * for scales), with periodic-checkpoint costs priced in.
 *
 * These are entropy ESTIMATES of candidate codings, not
 * implemented coders. No pack is modified.
 *
 * Usage: nq2-frameprobe model.nq2
 *
 * Plain hosted C99. Build: cc -O2 -std=c99 -o nq2-frameprobe
 * tools/nq2-frameprobe.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

static float half_to_float(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {
            exp = 1;
            while (!(mant & 0x400u)) { mant <<= 1; exp--; }
            mant &= 0x3ffu;
            bits = sign | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) bits = sign | 0x7f800000u | (mant << 13);
    else bits = sign | ((exp + 112) << 23) | (mant << 13);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static uint32_t rd32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t rd64(const uint8_t *p)
{
    return (uint64_t)rd32(p) | ((uint64_t)rd32(p + 4) << 32);
}

/* Byte-wise 16-bit load: the pack's half sections are not
 * guaranteed 2-byte aligned in the stream buffer, so they must
 * never be read through a uint16_t pointer (Archie P0-4). */
static uint16_t rd16(const uint8_t *p)
{
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static double entropy_of(const uint64_t *counts, int n, uint64_t total)
{
    double h = 0.0;
    for (int i = 0; i < n; i++) {
        if (!counts[i]) continue;
        double p = (double)counts[i] / (double)total;
        h -= p * log2(p);
    }
    return h;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: nq2-frameprobe model.nq2\n");
        return 1;
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    fseek(f, 0, SEEK_END);
    long fsz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = (uint8_t *)malloc((size_t)fsz);
    if (!buf) { fprintf(stderr, "out of memory\n"); return 1; }
    if (fread(buf, 1, (size_t)fsz, f) != (size_t)fsz) {
        fprintf(stderr, "short read\n");
        return 1;
    }
    fclose(f);

    if (fsz < 36 || memcmp(buf, "NQ2\0", 4) != 0) {
        fprintf(stderr, "not an .nq2 stream\n");
        return 1;
    }
    uint32_t version = rd32(buf + 4);
    uint32_t block   = rd32(buf + 8);
    uint64_t total_w = rd64(buf + 16);
    uint64_t n_blocks = rd64(buf + 24);
    uint64_t header = 32;
    if (version == 2) {
        uint32_t n_books = rd32(buf + 32);
        header = 36 + (uint64_t)n_books * 8;
    }
    /* T5 hardening (CODE-REVIEW-2026-11-24): the size check must
     * not be defeatable by uint64 wrap in 5 * n_blocks. */
    if (n_blocks > (UINT64_MAX - header) / 5
        || header + 5 * n_blocks > (uint64_t)fsz) {
        fprintf(stderr, "stream shorter than its header claims\n");
        return 1;
    }
    const uint8_t *mins = buf + header + n_blocks;
    const uint8_t *scales = buf + header + 3 * n_blocks;

    printf("frame probe: %s\n", argv[1]);
    printf("blocks: %llu of %u weights; stream %ld bytes\n",
           (unsigned long long)n_blocks, block, fsz);
    printf("frame bytes: %llu of %llu stream bytes (%.1f%%); "
           "frame rate %.5f bits/weight\n",
           (unsigned long long)(4 * n_blocks),
           (unsigned long long)fsz,
           100.0 * (double)(4 * n_blocks) / (double)fsz,
           (double)(32 * n_blocks) / (double)total_w);

    /* ---- per-section statistics ---- */
    const char *names[2] = { "min", "scale" };
    double delta_h0[2] = { 0.0, 0.0 };
    for (int s = 0; s < 2; s++) {
        const uint8_t *sec = s ? scales : mins;
        uint64_t *code_hist = (uint64_t *)calloc(65536, sizeof(uint64_t));
        uint64_t *byte_hist = (uint64_t *)calloc(256, sizeof(uint64_t));
        /* code deltas range over [-65535, 65535] */
        uint64_t *delt_hist = (uint64_t *)calloc(131071, sizeof(uint64_t));
        if (!code_hist || !byte_hist || !delt_hist) {
            fprintf(stderr, "out of memory\n");
            return 1;
        }
        uint64_t distinct = 0, zero_d = 0;
        double sum_abs_d = 0.0;
        for (uint64_t i = 0; i < n_blocks; i++) {
            uint16_t c = rd16(sec + 2 * i);
            if (code_hist[c] == 0) distinct++;
            code_hist[c]++;
            byte_hist[c & 0xff]++;
            byte_hist[(c >> 8) & 0xff]++;
            if (i > 0) {
                int d = (int)c - (int)rd16(sec + 2 * (i - 1));
                delt_hist[d + 65535]++;
                if (d == 0) zero_d++;
                sum_abs_d += d < 0 ? -d : d;
            }
        }
        double h_code = entropy_of(code_hist, 65536, n_blocks);
        double h_byte = entropy_of(byte_hist, 256, 2 * n_blocks);
        double h_delt = entropy_of(delt_hist, 131071, n_blocks - 1);
        delta_h0[s] = h_delt;
        printf("\n%s section:\n", names[s]);
        printf("  distinct fp16 codes: %llu of 65536\n",
               (unsigned long long)distinct);
        printf("  H0 of codes: %.4f bits (raw is 16)\n", h_code);
        printf("  H0 of raw bytes: %.4f bits/byte\n", h_byte);
        printf("  adjacent code delta: H0 %.4f bits, mean |d| %.1f "
               "codes, %.2f%% zero\n",
               h_delt, sum_abs_d / (double)(n_blocks - 1),
               100.0 * (double)zero_d / (double)(n_blocks - 1));
        free(code_hist); free(byte_hist); free(delt_hist);
    }

    /* ---- scales in the log domain (Gizmo proposal B) ---- */
    /* Quantize successive log2(scale) deltas at 1/64 steps and
     * measure the symbol entropy and the rounding this implies. */
    {
        uint64_t *hist = (uint64_t *)calloc(4096, sizeof(uint64_t));
        if (!hist) { fprintf(stderr, "out of memory\n"); return 1; }
        double prev = log2((double)half_to_float(rd16(scales)));
        double sq_err = 0.0;
        uint64_t clipped = 0;
        for (uint64_t i = 1; i < n_blocks; i++) {
            double cur = log2((double)half_to_float(rd16(scales + 2 * i)));
            double dl = cur - prev;
            prev = cur;
            long sym = lrint(dl * 64.0);
            double qerr = dl - (double)sym / 64.0;
            sq_err += qerr * qerr;
            if (sym < -2048 || sym > 2047) { clipped++; continue; }
            hist[sym + 2048]++;
        }
        double h = entropy_of(hist, 4096, n_blocks - 1 - clipped);
        double rms = sqrt(sq_err / (double)(n_blocks - 1));
        printf("\nscale log2 deltas, quantized at 1/64:\n");
        printf("  symbol H0: %.4f bits; RMS quantization error %.5f "
               "in log2 (about %.2f%% relative on scale)\n",
               h, rms, 100.0 * (exp2(rms) - 1.0));
        printf("  deltas outside the +/-%d symbol range: %llu\n",
               2048, (unsigned long long)clipped);
        /* carry this coding's bits into the checkpoint model */
        delta_h0[1] = h;
        free(hist);
    }

    /* ---- checkpoint costing (Gizmo proposals A+B+D+E) ---- */
    /* Model: absolute frames (32 bits) every C blocks; between
     * checkpoints, min code deltas and scale log deltas entropy
     * coded at their measured H0. Index stream is untouched. */
    printf("\ncheckpoint model, estimated frame bits per weight "
           "(block %u):\n", block);
    printf("  current (all frames raw): %.5f\n",
           32.0 / (double)block);
    const uint64_t intervals[5] = { 32, 64, 256, 1024, 0 };
    for (int k = 0; k < 5; k++) {
        uint64_t C = intervals[k];
        double frac_ckpt = C ? 1.0 / (double)C : 0.0;
        double bits_per_block = frac_ckpt * 32.0
            + (1.0 - frac_ckpt) * (delta_h0[0] + delta_h0[1]);
        if (C)
            printf("  checkpoint every %llu blocks: %.5f\n",
                   (unsigned long long)C, bits_per_block / block);
        else
            printf("  no checkpoints (one chain): %.5f\n",
                   bits_per_block / block);
    }
    printf("\nNote: entropy estimates of candidate codings, not "
           "implemented coders.\nReal coders land above their "
           "entropy; the gaps between rows are the signal.\n");
    free(buf);
    return 0;
}
