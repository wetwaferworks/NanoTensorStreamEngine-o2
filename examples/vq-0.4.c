/* vq-0.4.c ... NanoQuatTwo example 1: 0.4 bits per weight.
 *
 * FREESTANDING: no standard library, no libc, no libm. Every helper
 * below (output, sine, memory) is implemented in this file. Build:
 *
 *     gcc -O2 -nostdlib -static -ffreestanding -fno-stack-protector \
 *         -fno-builtin -o vq-0.4 vq-0.4.c
 *
 * The packing: weights travel in blocks of 20. A codebook of 256
 * prototypical blocks lives on the side. Each block is stored as
 * one 8-bit index. 8 bits / 20 weights = 0.4 bits per weight.
 * One quad word (64 bits) holds 8 indexes = 160 weights.
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

static void sys_exit(int code)
{
    sys3(60, code, 0, 0);
    __builtin_unreachable();
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

/* integer right-aligned in a field of width 3, like printf %3d */
static void out_i64_w3(long v)
{
    char buf[24];
    int i = 0, neg = 0;
    u64 x;
    if (v < 0) { neg = 1; x = (u64)(-v); } else x = (u64)v;
    if (!x) buf[i++] = '0';
    while (x) { buf[i++] = (char)('0' + (x % 10)); x /= 10; }
    int len = i + neg;
    while (len < 3) { out_char(' '); len++; }
    if (neg) out_char('-');
    while (i) out_char(buf[--i]);
}

static void out_hex64(u64 v)
{
    out_str("0x");
    for (int s = 60; s >= 0; s -= 4) {
        int d = (int)((v >> s) & 15);
        out_char((char)(d < 10 ? '0' + d : 'a' + d - 10));
    }
}

/* double with a fixed number of decimals, like printf %.*f */
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

/* Own sine: range reduction to [-pi/2, pi/2], then a Taylor series
 * through x^11. Error under 2e-6 on the folded range: plenty for a
 * toy whose job is choosing the nearest prototype. */
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

/* ------------------------------ the toy ------------------------------ */

#define BLOCK          20   /* weights per prototypical block      */
#define ENTRIES       256   /* codebook size -> 8-bit index        */
#define BLOCKS_PER_QW   8   /* 8 indexes x 8 bits = 64 bits        */
#define NWEIGHTS (BLOCKS_PER_QW * BLOCK)   /* 160 weights per quad word */

static float codebook[ENTRIES][BLOCK];

static void build_codebook(void)
{
    for (int k = 0; k < ENTRIES; k++) {
        double phase = (k % 16) * 0.39;
        double freq  = 0.15 + (k / 16) * 0.11;
        double amp   = 0.25 + (k % 5) * 0.18;
        for (int i = 0; i < BLOCK; i++)
            codebook[k][i] = (float)(amp * dsin(phase + i * freq));
    }
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
    build_codebook();

    float w[NWEIGHTS];
    for (int i = 0; i < NWEIGHTS; i++)
        w[i] = (float)(0.7 * dsin(i * 0.35) + 0.3 * dsin(i * 0.9 + 1.0));

    u64 qw = 0;
    for (int b = 0; b < BLOCKS_PER_QW; b++) {
        int idx = nearest_block(&w[b * BLOCK]);
        qw |= (u64)(u8)idx << (8 * b);
    }
    out_str("quad word = ");
    out_hex64(qw);
    out_str("\nindexes   =");
    for (int b = 0; b < BLOCKS_PER_QW; b++)
        out_i64_w3((long)((qw >> (8 * b)) & 0xff));
    out_str("\n");

    float out[NWEIGHTS];
    for (int b = 0; b < BLOCKS_PER_QW; b++) {
        int idx = (int)((qw >> (8 * b)) & 0xff);
        for (int i = 0; i < BLOCK; i++)
            out[b * BLOCK + i] = codebook[idx][i];
    }

    double max_err = 0.0;
    for (int i = 0; i < NWEIGHTS; i++) {
        double d = dabs((double)w[i] - out[i]);
        if (d > max_err) max_err = d;
    }

    out_str("packed    = ");
    out_i64(NWEIGHTS);
    out_str(" weights into 64 bits = ");
    out_f64(64.0 / NWEIGHTS, 3);
    out_str(" bits per weight\n");
    out_str("max error = ");
    out_f64(max_err, 4);
    out_str("  (the toll the weights paid to travel)\n");
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
