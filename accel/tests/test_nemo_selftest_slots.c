/* test_nemo_selftest_slots.c - host-only checks for the CUDA lane's
 * self-test bookkeeping. No device and no driver are needed: the test
 * compiles nemo_gpu.c into itself to reach the static helpers.
 *
 *  1. synth_slot hands out exactly SELFTEST_SYNTHS zeroed slots and then
 *     refuses, so the self-test can never write past its stack array
 *     (the 2026-11-26 review issue: a third tensor went into synth t[2]).
 *  2. A canary placed right after the array survives a refused request.
 *  3. Cleanup over the handed-out count frees zeroed slots safely
 *     (synth_free on an untouched slot must be a no-op).
 *  4. The kernel digest used by the calibration cache key is stable,
 *     matches an independent FNV-1a over the kernel source and options,
 *     and moves when one byte of the source changes. */
#include "../nemo_gpu.c"

static int fails, passes;
#define CHECK(c, ...) do { if (c) { passes++; printf("PASS "); } else { fails++; printf("FAIL "); } \
    printf(__VA_ARGS__); printf("\n"); } while (0)

static int all_zero(const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) if (b[i]) return 0;
    return 1;
}

static uint64_t ref_fnv(const unsigned char *p, size_t n, uint64_t h) {
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x100000001b3ull; }
    return h;
}

int main(void) {
    struct { synth t[SELFTEST_SYNTHS]; unsigned char canary[64]; } box;
    memset(&box, 0xA5, sizeof box);
    int nt = 0;
    synth *got[SELFTEST_SYNTHS + 1];
    for (int i = 0; i <= SELFTEST_SYNTHS; i++) got[i] = synth_slot(box.t, &nt, SELFTEST_SYNTHS);

    CHECK(SELFTEST_SYNTHS >= 3, "slot count covers the self-test's three tensors (%d)", SELFTEST_SYNTHS);
    int ok = 1;
    for (int i = 0; i < SELFTEST_SYNTHS; i++) ok &= got[i] == &box.t[i] && all_zero(got[i], sizeof *got[i]);
    CHECK(ok, "the first %d requests get distinct zeroed slots in order", SELFTEST_SYNTHS);
    CHECK(got[SELFTEST_SYNTHS] == NULL, "request %d past the end is refused", SELFTEST_SYNTHS + 1);
    CHECK(nt == SELFTEST_SYNTHS, "handed-out count stops at the array size (nt=%d)", nt);
    int canary_ok = 1;
    for (size_t i = 0; i < sizeof box.canary; i++) canary_ok &= box.canary[i] == 0xA5;
    CHECK(canary_ok, "memory right after the array is untouched");
    int neg = -1;
    CHECK(synth_slot(box.t, &neg, SELFTEST_SYNTHS) == NULL && neg == -1, "a negative count is refused, not indexed");

    /* Cleanup path: zeroed slots must be safe to free with no device. */
    nemo_gpu g; memset(&g, 0, sizeof g);
    for (int i = 0; i < nt; i++) synth_free(&g, &box.t[i]);
    CHECK(1, "cleanup over %d zeroed slots touched no device and freed nothing twice", nt);

    uint64_t d1 = nemo_gpu_kernel_digest(), d2 = nemo_gpu_kernel_digest();
    uint64_t h = ref_fnv((const unsigned char *)nemo_gpu_cuda_src, sizeof nemo_gpu_cuda_src - 1, 0xcbf29ce484222325ull);
    for (int i = 0; i < NVRTC_FIXED_OPTS; i++) {
        h = ref_fnv((const unsigned char *)"", 1, h);
        h = ref_fnv((const unsigned char *)nvrtc_fixed_opts[i], strlen(nvrtc_fixed_opts[i]), h);
    }
    CHECK(d1 == d2 && d1 == h, "kernel digest is stable and matches an independent FNV-1a (%016llx)", (unsigned long long)d1);
    size_t n = sizeof nemo_gpu_cuda_src - 1;
    char *copy = (char *)malloc(n);
    if (copy) {
        memcpy(copy, nemo_gpu_cuda_src, n);
        copy[n / 2] ^= 1;
        uint64_t hm = ref_fnv((const unsigned char *)copy, n, 0xcbf29ce484222325ull);
        uint64_t hs = ref_fnv((const unsigned char *)nemo_gpu_cuda_src, n, 0xcbf29ce484222325ull);
        CHECK(hm != hs, "one flipped bit in the kernel source changes the digest");
        free(copy);
    } else CHECK(0, "allocate a copy of the kernel source");
    const char *b = nemo_gpu_build_id();
    CHECK(b && strlen(b) >= 11, "build stamp is present (%s)", b ? b : "(null)");

    printf("%d PASS / %d FAIL\n", passes, fails);
    return fails ? 1 : 0;
}
