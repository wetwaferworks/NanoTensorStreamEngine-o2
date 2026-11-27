/* test-handshake.c - the zero-mutex compute pool and claim words.
 *
 * Builds with: gcc -O1 -g -std=c99 -pthread -I tools
 *              tools/test-handshake.c -o test-handshake
 * Optional:    -fsanitize=thread
 *
 * Usage: test-handshake [nworkers] [rounds]
 * Default nworkers is 12, rounds is 256. A run prints a 64-bit
 * FNV-1a of the reduced output, which must match the serial
 * reference and stay the same at every worker count.
 *
 * Direction by JD. Co-implementation by Cortana, built with
 * Grok Bot by SpaceX.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>

#define NQ2HS_ENGINE_POOL
#include "nq2-handshake.h"

static int checks, failures;

static void expect(const char *name, int cond)
{
    checks++;
    if (cond) {
        printf("ok   %s\n", name);
    } else {
        failures++;
        printf("FAIL %s\n", name);
    }
}

static uint64_t fnv1a64(const void *p, size_t n)
{
    const uint8_t *b = (const uint8_t *)p;
    uint64_t h = 14695981039346656037ull;
    for (size_t i = 0; i < n; i++) {
        h ^= b[i];
        h *= 1099511628211ull;
    }
    return h;
}

typedef struct {
    double *in;
    double *out;
    int n;
} job;

/* Disjoint partition: each output element is written by exactly
 * one worker. The join-side hash is a fixed-order reduce. */
static void squares(void *ctx, int worker, int nworkers)
{
    job *j = (job *)ctx;
    int n = j->n;
    int a = (int)((long)worker * n / nworkers);
    int b = (int)((long)(worker + 1) * n / nworkers);
    for (int i = a; i < b; i++)
        j->out[i] = j->in[i] * j->in[i] + (double)i;
}

static void fill_serial(double *in, double *out, int n)
{
    job j;
    j.in = in;
    j.out = out;
    j.n = n;
    squares(&j, 0, 1);
}

typedef struct {
    int *slots;
    int *wins;
    int nslots;
} claim_job;

static void take_slots(void *ctx, int worker, int nworkers)
{
    claim_job *c = (claim_job *)ctx;
    (void)nworkers;
    for (int i = 0; i < c->nslots; i++) {
        if (hs_cas_int(&c->slots[i], 0, worker + 1))
            c->wins[worker]++;
    }
}

int main(int argc, char **argv)
{
    int nworkers = 12;
    int rounds = 256;
    const int n = 4096;
    if (argc > 1) nworkers = atoi(argv[1]);
    if (argc > 2) rounds = atoi(argv[2]);
    if (nworkers < 1) nworkers = 1;
    if (rounds < 1) rounds = 1;

    double *in = (double *)malloc(sizeof(double) * (size_t)n);
    double *out = (double *)malloc(sizeof(double) * (size_t)n);
    double *ref = (double *)malloc(sizeof(double) * (size_t)n);
    if (!in || !out || !ref) {
        fprintf(stderr, "oom\n");
        return 1;
    }
    for (int i = 0; i < n; i++)
        in[i] = (double)(i * 17 + 3) / 256.0;
    fill_serial(in, ref, n);
    uint64_t href = fnv1a64(ref, sizeof(double) * (size_t)n);

    pool_init(nworkers);
    expect("pool started", g_pool.nworkers >= 1
           && g_pool.nworkers <= nworkers);

    job j;
    j.in = in;
    j.out = out;
    j.n = n;
    uint64_t last = 0;
    int all_match = 1;
    for (int r = 0; r < rounds; r++) {
        memset(out, 0, sizeof(double) * (size_t)n);
        /* A changing input so a stale buffer cannot hide a missed
         * dispatch. The serial reference is rebuilt to match. */
        in[r % n] += 1.0;
        fill_serial(in, ref, n);
        href = fnv1a64(ref, sizeof(double) * (size_t)n);
        pool_run(squares, &j);
        uint64_t h = fnv1a64(out, sizeof(double) * (size_t)n);
        if (h != href) all_match = 0;
        last = h;
    }
    expect("pool output matches serial at every round", all_match);
    printf("hash %016llx  workers %d  rounds %d\n",
           (unsigned long long)last, g_pool.nworkers, rounds);

    /* Claim words: exactly one winner per slot, any worker count. */
    {
        const int nslots = 1024;
        int *slots = (int *)calloc((size_t)nslots, sizeof(int));
        int *wins = (int *)calloc((size_t)g_pool.nworkers, sizeof(int));
        if (!slots || !wins) {
            fprintf(stderr, "oom\n");
            pool_shutdown();
            return 1;
        }
        claim_job c;
        c.slots = slots;
        c.wins = wins;
        c.nslots = nslots;
        pool_run(take_slots, &c);
        int taken = 0, over = 0, sum = 0;
        for (int i = 0; i < nslots; i++) {
            if (slots[i] == 0) over++;
            else taken++;
        }
        for (int w = 0; w < g_pool.nworkers; w++)
            sum += wins[w];
        expect("every slot claimed exactly once",
               taken == nslots && over == 0 && sum == nslots);
        free(slots);
        free(wins);
    }

    pool_shutdown();
    expect("shutdown returned", 1);

    free(in);
    free(out);
    free(ref);
    printf("RESULT: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
