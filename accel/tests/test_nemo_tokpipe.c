/* test_nemo_tokpipe.c - the prompt tokenizer pipeline (accel/nemo_tokpipe):
 *  1. every safe cut of a mixed text: encode(a) ++ encode(b) == encode(a+b);
 *  2. pipelined (head + background segments) ids == one-shot ids, several
 *     plans and prompt sizes;
 *  3. timing: one-shot encode vs the part left on the critical path
 *     (segment 0) when the rest overlaps the GPU prefill.
 * Usage: test_nemo_tokpipe <tokenizer.json> */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
static double now_ms(void) { LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c); return (double)c.QuadPart * 1000.0 / (double)f.QuadPart; }
#else
#include <time.h>
static double now_ms(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6; }
#endif
#include "tokenizer.h"
#include "pool.h"
#include "nemo_tokpipe.h"

static const char *STORY =
    "The river ran past the old mill where the farmers brought their grain each autumn, and the miller kept a "
    "ledger of every sack, every price and every promise made over the long winter months while the children "
    "played along the frozen banks and the wheel stood still under a coat of ice.\n";
static const char *MIXED =
    "<SPECIAL_10>System\nYou are Nemo. Answer in one short sentence.\n<SPECIAL_11>User\n"
    "What's 7 times 8? It's 56, isn't it?! Call +1 (555) 010-2030 at 09:45, ok?\n"
    "Caf\xc3\xa9 na\xc3\xafve r\xc3\xa9sum\xc3\xa9 \xe2\x80\x94 \xe6\x9d\xb1\xe4\xba\xac and \xd0\x9c\xd0\xbe\xd1\x81\xd0\xba\xd0\xb2\xd0\xb0, "
    "CamelCase HTTPServer iPhone McDonald's  double  spaces\t\ttabs   \n\n\n"
    "code: for (int i = 0; i < n; i++) { x[i] = y[i] * 2.5e-3; } // done\n"
    "URL https://example.com/a/b?c=d&e=f and path C:\\Users\\name\\file.txt end\n"
    "<think>\nshould stay as one added token</think> Final answer: The capital of Ohio is Columbus.\n";

static int same(const int *a, int na, const int *b, int nb) { return na == nb && !memcmp(a, b, (size_t)na * sizeof(int)); }

static int pipe_ids(nemo_pool *pl, const nemo_tokenizer *tk, const char *text, size_t fb, size_t sb,
                    int *out, int cap, int *nseg, double *head_ms) {
    nemo_tokpipe tp;
    if (nemo_tokpipe_plan(&tp, tk, text, fb, sb) < 0) return -1;
    *nseg = tp.nseg;
    double a = now_ms();
    int n = nemo_tokpipe_head(&tp);
    *head_ms = now_ms() - a;
    if (n < 0 || n > cap) { nemo_tokpipe_free(&tp); return -1; }
    const int *p; memcpy(out, tp.buf, (size_t)n * sizeof(int));
    if (tp.nseg > 1) nemo_pool_prep(pl, nemo_tokpipe_run, &tp);
    for (int k = 1; k < tp.nseg; k++) {
        int c = nemo_tokpipe_wait(&tp, k, &p);
        if (c < 0 || n + c > cap) { n = -1; break; }
        memcpy(out + n, p, (size_t)c * sizeof(int)); n += c;
    }
    nemo_pool_prep_wait(pl);
    nemo_tokpipe_free(&tp);
    return n;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: test_nemo_tokpipe <tokenizer.json>\n"); return 2; }
    nemo_tokenizer *tk = nemo_tokenizer_load(argv[1]);
    if (!tk) { printf("FAIL tokenizer load\n"); return 1; }
    printf("cut_safe=%d\n", nemo_tokenizer_cut_safe(tk));
    if (!nemo_tokenizer_cut_safe(tk)) { printf("FAIL tokenizer does not allow cuts\n"); return 1; }
    int fails = 0;
    int cap = 1 << 20;
    int *A = malloc((size_t)cap * 4), *B = malloc((size_t)cap * 4), *C = malloc((size_t)cap * 4);
    char *tmp = malloc(1 << 20);
    /* 1: every safe cut of the mixed text and of a story paragraph */
    const char *texts[2] = {MIXED, STORY};
    for (int ti = 0; ti < 2; ti++) {
        const char *s = texts[ti]; size_t L = strlen(s);
        int na = nemo_tokenizer_encode(tk, s, A, cap); int cuts = 0, bad = 0;
        for (size_t p = 1; p + 1 < L; p++) {
            char a = s[p - 1], b = s[p + 1];
            int al = ((a|32) >= 'a' && (a|32) <= 'z'), bl = ((b|32) >= 'a' && (b|32) <= 'z');
            if (s[p] != ' ' || !al || !bl) continue;
            memcpy(tmp, s, p); tmp[p] = 0; int n1 = nemo_tokenizer_encode(tk, tmp, B, cap);
            int n2 = nemo_tokenizer_encode(tk, s + p, B + n1, cap - n1);
            cuts++; if (n1 < 0 || n2 < 0 || !same(A, na, B, n1 + n2)) bad++;
        }
        printf("%-34s %s  (%d safe cuts, %d differ, %d ids)\n", ti ? "every safe cut, story" : "every safe cut, mixed text",
               bad ? "FAIL" : "PASS", cuts, bad, na);
        fails += bad != 0;
    }
    /* 2 and 3: pipelined vs one-shot, several sizes and plans */
    nemo_pool pl; memset(&pl, 0, sizeof pl);
    if (nemo_pool_init(&pl, 2)) { printf("FAIL pool\n"); return 1; }
    int words[] = {8, 52, 211, 419, 1000, 4000};
    size_t plans[3][2] = {{256, 1024}, {64, 64}, {1, 1}};
    for (int wi = 0; wi < 6; wi++) {
        /* prompt of about words[wi] words: STORY repeated, then MIXED once */
        size_t L = 0; int w = 0;
        while (w < words[wi]) {
            const char *s = STORY; while (*s && w < words[wi]) { tmp[L++] = *s; if (*s == ' ') w++; s++; }
        }
        tmp[L] = 0;
        if (words[wi] >= 1000) { strcat(tmp, MIXED); L = strlen(tmp); }
        double t0 = now_ms(); int na = 0; int reps = words[wi] <= 52 ? 50 : 5;
        for (int r = 0; r < reps; r++) na = nemo_tokenizer_encode(tk, tmp, A, cap);
        double one = (now_ms() - t0) / reps;
        int ok = 1; int nseg0 = 0; double head0 = 0;
        for (int pi = 0; pi < 3; pi++) {
            int nseg; double hm;
            int nb = pipe_ids(&pl, tk, tmp, plans[pi][0], plans[pi][1], B, cap, &nseg, &hm);
            if (!same(A, na, B, nb)) ok = 0;
            if (pi == 0) { nseg0 = nseg; head0 = hm; }
        }
        printf("%5d words (%5d ids, %6zu B): one-shot encode %8.3f ms; pipelined head %7.3f ms, %3d segments; ids %s\n",
               words[wi], na, L, one, head0, nseg0, ok ? "identical (3 plans) PASS" : "DIFFER FAIL");
        fails += !ok;
    }
    nemo_pool_shutdown(&pl);
    (void)C;
    printf(fails ? "FAIL\n" : "PASS\n");
    nemo_tokenizer_free(tk);
    return fails ? 1 : 0;
}
