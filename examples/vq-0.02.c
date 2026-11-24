/* vq-0.02.c ... NanoQuatTwo flagship example: 0.02 bits per
 * weight, 64 quad words, and a first reading lesson.
 *
 * FREESTANDING: no standard library, no libc, no libm. Every
 * helper below (output, sine, memory, random) is implemented in
 * this file. Build:
 *
 *     gcc -O2 -nostdlib -static -ffreestanding -fno-stack-protector \
 *         -fno-builtin -o vq-0.02 vq-0.02.c
 *
 * The packing: weights travel in blocks of 400. A codebook of 256
 * prototypical blocks (the memory file) lives on the side. Each
 * block is stored as one 8-bit index.
 *
 *     8 bits / 400 weights = 0.02 bits per weight
 *     512 indexes x 8 bits = 4096 bits = 64 quad words
 *     512 blocks  x 400    = 204,800 weights
 *
 * The lesson: the first four blocks of weights are the letters
 * N, E, M, O drawn on 20x20 grids. Their prototypes are FROZEN
 * into the codebook (taught, not learned). The other 252 entries
 * are learned from the remaining wave blocks by k-means. After
 * packing and unpacking, we read the letters back.
 */

/* ---------- own runtime: syscalls, output, memory, math ---------- */

typedef unsigned char u8;
typedef unsigned long u64;

static long sys3(long n, long a, long b, long c)
{
    long r;
    __asm__ volatile("syscall" : "=a"(r)
                     : "a"(n), "D"(a), "S"(b), "d"(c)
                     : "rcx", "r11", "memory");
    return r;
}

static void out_char(char ch) { sys3(1, 1, (long)&ch, 1); }

static void out_str(const char *s)
{
    long n = 0;
    while (s[n]) n++;
    sys3(1, 1, (long)s, n);
}

static void out_u64(u64 v)
{
    char buf[24];
    int i = 0;
    if (!v) { out_char('0'); return; }
    while (v) { buf[i++] = (char)('0' + (v % 10)); v /= 10; }
    while (i) out_char(buf[--i]);
}

static void out_i64(long v)
{
    if (v < 0) { out_char('-'); out_u64((u64)(-v)); }
    else out_u64((u64)v);
}

static void out_f64(double v, int digits)
{
    if (v < 0) { out_char('-'); v = -v; }
    u64 ip = (u64)v;
    double frac = v - (double)ip;
    out_u64(ip);
    out_char('.');
    for (int i = 0; i < digits; i++) {
        frac *= 10.0;
        u64 d = (u64)frac;
        if (d > 9) d = 9;
        out_char((char)('0' + d));
        frac -= (double)d;
    }
}

void *memcpy(void *dst, const void *src, u64 n)
{
    volatile u8 *d = (volatile u8 *)dst;
    const u8 *s = (const u8 *)src;
    for (u64 i = 0; i < n; i++) d[i] = s[i];
    return dst;
}

void *memset(void *dst, int v, u64 n)
{
    volatile u8 *d = (volatile u8 *)dst;
    for (u64 i = 0; i < n; i++) d[i] = (u8)v;
    return dst;
}

void *memmove(void *dst, const void *src, u64 n)
{
    volatile u8 *d = (volatile u8 *)dst;
    const u8 *s = (const u8 *)src;
    if ((const u8 *)dst < s) {
        for (u64 i = 0; i < n; i++) d[i] = s[i];
    } else {
        for (u64 i = n; i > 0; i--) d[i - 1] = s[i - 1];
    }
    return dst;
}

static double dabs(double x) { return x < 0.0 ? -x : x; }

/* Own sine: range reduction to [-pi/2, pi/2], Taylor through x^11. */
static double dsin(double x)
{
    const double PI = 3.14159265358979323846;
    const double TAU = 6.28318530717958647692;
    double k = (double)(long)(x / TAU);
    x -= k * TAU;
    if (x > PI) x -= TAU;
    else if (x < -PI) x += TAU;
    if (x > PI / 2) x = PI - x;
    else if (x < -PI / 2) x = -PI - x;
    double x2 = x * x;
    return x * (1.0 + x2 * (-1.0 / 6.0 + x2 * (1.0 / 120.0 +
           x2 * (-1.0 / 5040.0 + x2 * (1.0 / 362880.0 +
           x2 * (-1.0 / 39916800.0))))));
}

/* Own random: xorshift64, deterministic everywhere. */
static u64 rng_state = 0x9e3779b97f4a7c15ull;
static double frand(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (double)(rng_state >> 11) * (1.0 / 9007199254740992.0);
}
static double noise(double amp) { return (frand() * 2.0 - 1.0) * amp; }

/* ------------------------------ the toy ------------------------------ */

#define BLOCK    400                 /* weights per block          */
#define ENTRIES  256                 /* codebook size, 8-bit index */
#define NBLOCKS  512                 /* blocks in the toy tensor   */
#define NWEIGHTS (NBLOCKS * BLOCK)   /* 204,800 weights            */
#define NLETTERS 4                   /* taught blocks: N E M O     */

static float codebook[ENTRIES][BLOCK];
static float data[NWEIGHTS];
static float out[NWEIGHTS];
static u8 packed[NBLOCKS];           /* 512 bytes = 64 quad words  */

/* 5x7 letters for N, E, M, O, one string per row. */
static const char *glyph[NLETTERS][7] = {
    { "10001", "11001", "10101", "10011", "10001", "10001", "10001" }, /* N */
    { "11111", "10000", "10000", "11110", "10000", "10000", "11111" }, /* E */
    { "10001", "11011", "10101", "10101", "10001", "10001", "10001" }, /* M */
    { "01110", "10001", "10001", "10001", "10001", "10001", "01110" }  /* O */
};

/* Draw letter L into a 20x20 block: each font pixel becomes 4 wide
 * and 2 tall, centered with 3 blank rows above and below. */
static void draw_letter(float *blk, int L)
{
    for (int r = 0; r < 20; r++) {
        int sr = (r - 3) / 2;
        for (int c = 0; c < 20; c++) {
            int sc = c / 4;
            int on = (r >= 3 && r < 17) && glyph[L][sr][sc] == '1';
            blk[r * 20 + c] = on ? 0.9f : -0.9f;
        }
    }
}

/* A smooth wave block, standing in for an ordinary layer's weights. */
static void wave_block(float *blk, int seed)
{
    double p1 = seed * 0.37, p2 = seed * 1.13;
    double f1 = 0.05 + (seed % 7) * 0.03, f2 = 0.21 + (seed % 5) * 0.05;
    for (int i = 0; i < BLOCK; i++)
        blk[i] = (float)(0.6 * dsin(p1 + i * f1) + 0.4 * dsin(p2 + i * f2));
}

static int nearest_block(const float *w)
{
    int best = 0;
    double best_err = 1e30;
    for (int k = 0; k < ENTRIES; k++) {
        double err = 0.0;
        for (int i = 0; i < BLOCK; i++) {
            double d = (double)w[i] - codebook[k][i];
            err += d * d;
        }
        if (err < best_err) { best_err = err; best = k; }
    }
    return best;
}

void run_main(void)
{
    /* The tensor: 4 letter blocks, then 508 wave blocks, all with
     * a breath of noise on them, like real weights have. */
    for (int L = 0; L < NLETTERS; L++) {
        draw_letter(&data[L * BLOCK], L);
        for (int i = 0; i < BLOCK; i++) data[L * BLOCK + i] += (float)noise(0.05);
    }
    for (int b = NLETTERS; b < NBLOCKS; b++) {
        wave_block(&data[b * BLOCK], b);
        for (int i = 0; i < BLOCK; i++) data[b * BLOCK + i] += (float)noise(0.05);
    }

    /* TEACH the book the alphabet: letter prototypes frozen in. */
    for (int L = 0; L < NLETTERS; L++)
        draw_letter(codebook[L], L);

    /* The rest of the book starts as copies of real wave blocks... */
    for (int k = NLETTERS; k < ENTRIES; k++) {
        int src = NLETTERS + (int)(frand() * (NBLOCKS - NLETTERS));
        memcpy(codebook[k], &data[src * BLOCK], sizeof(float) * BLOCK);
    }

    /* ...and then LEARNS by k-means over the wave blocks only.
     * The letter entries stay frozen: curriculum is curriculum. */
    static double sum[ENTRIES][BLOCK];
    static int count[ENTRIES];
    for (int iter = 0; iter < 8; iter++) {
        memset(sum, 0, sizeof(sum));
        memset(count, 0, sizeof(count));
        for (int b = NLETTERS; b < NBLOCKS; b++) {
            int k = nearest_block(&data[b * BLOCK]);
            if (k < NLETTERS) k = NLETTERS;   /* letters take no students */
            count[k]++;
            for (int i = 0; i < BLOCK; i++) sum[k][i] += data[b * BLOCK + i];
        }
        for (int k = NLETTERS; k < ENTRIES; k++)
            if (count[k] > 0)
                for (int i = 0; i < BLOCK; i++)
                    codebook[k][i] = (float)(sum[k][i] / count[k]);
    }

    /* PACK: one index per block. 512 indexes = 64 quad words. */
    for (int b = 0; b < NBLOCKS; b++)
        packed[b] = (u8)nearest_block(&data[b * BLOCK]);

    /* UNPACK: call every prototype back off the side. */
    for (int b = 0; b < NBLOCKS; b++)
        memcpy(&out[b * BLOCK], codebook[packed[b]], sizeof(float) * BLOCK);

    double err_sum = 0.0, err_max = 0.0;
    for (int i = 0; i < NWEIGHTS; i++) {
        double d = dabs((double)data[i] - out[i]);
        err_sum += d;
        if (d > err_max) err_max = d;
    }
    out_str("packed   = ");
    out_i64(NWEIGHTS);
    out_str(" weights into ");
    out_i64(NBLOCKS / 8);
    out_str(" quad words (");
    out_i64(NBLOCKS * 8);
    out_str(" index bits)\n");
    out_str("rate     = ");
    out_f64((double)(NBLOCKS * 8) / NWEIGHTS, 3);
    out_str(" bits per weight\n");
    out_str("error    = mean ");
    out_f64(err_sum / NWEIGHTS, 4);
    out_str(", max ");
    out_f64(err_max, 4);
    out_str(" (waves pay the toll)\n");

    /* READ: render the first four unpacked blocks as letters. */
    int letter_px = 0, letter_ok = 0;
    for (int L = 0; L < NLETTERS; L++) {
        out_str("\nblock ");
        out_i64(L);
        out_str(", read back:\n");
        for (int r = 0; r < 20; r++) {
            for (int c = 0; c < 20; c++) {
                float v = out[L * BLOCK + r * 20 + c];
                float d = data[L * BLOCK + r * 20 + c];
                out_char(v > 0.0f ? '#' : '.');
                letter_px++;
                if ((v > 0.0f) == (d > 0.0f)) letter_ok++;
            }
            out_char('\n');
        }
    }
    out_str("\nletter pixels right = ");
    out_i64(letter_ok);
    out_str(" of ");
    out_i64(letter_px);
    out_str(" (");
    out_f64(100.0 * letter_ok / letter_px, 1);
    out_str("%)\n");
}

/* Entry stub in raw assembly: at process entry the stack is
 * 16-byte aligned, but compiled C expects the post-call state
 * (rsp % 16 == 8 at function entry). Realign by hand, call the
 * program, then exit(0) by syscall. Without this, -O2 code that
 * uses aligned SSE moves on stack arrays faults. */
__asm__(
    ".globl _start\n"
    "_start:\n"
    "    and $-16, %rsp\n"
    "    call run_main\n"
    "    mov $60, %rax\n"
    "    xor %edi, %edi\n"
    "    syscall\n"
);
