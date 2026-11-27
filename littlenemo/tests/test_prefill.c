/* test_prefill.c - layer-major prefill must be bit-identical to the
 * token-by-token path. Runs a truncated stack (default 16 layers) on a real
 * checkpoint, feeds 7 tokens in two chunks (4 + 3) to cross a chunk boundary,
 * then checks the carried state with one more token and the refusal cases.
 *
 *     test_prefill <checkpoint_dir> [layers] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/forward.h"
#include "../src/loader.h"

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: test_prefill <ckpt> [layers]\n"); return 2; }
    int layers = argc > 2 ? atoi(argv[2]) : 16;
    nemo_checkpoint cp;
    if (nemo_checkpoint_open(&cp, argv[1]) != 0) { printf("FAIL open\n"); return 1; }
    int toks[7] = {1032, 1261, 5000, 12, 777, 42, 31337};
    int n = 7, fails = 0;
    static float ha[NEMO_HIDDEN], hb[NEMO_HIDDEN], hc[NEMO_HIDDEN], hd[NEMO_HIDDEN];
    nemo_model a, b;
    if (nemo_model_init_ex(&a, &cp, layers, 64, 3) != 0 ||
        nemo_model_init_ex(&b, &cp, layers, 64, 3) != 0) { printf("FAIL init\n"); return 1; }
    for (int i = 0; i < n; i++)
        if (nemo_model_step_hidden(&a, toks[i], ha) != 0) { printf("FAIL seq step %d\n", i); return 1; }
    /* batch in two chunks (4 + 3) to exercise the chunk boundary */
    if (nemo_model_prefill(&b, toks, 4, hb) != 0) { printf("FAIL prefill 1\n"); return 1; }
    if (nemo_model_prefill(&b, toks + 4, 3, hb) != 0) { printf("FAIL prefill 2\n"); return 1; }
    int same = memcmp(ha, hb, sizeof ha) == 0;
    printf("%s last hidden bit-identical (layers=%d, 7 tokens, chunks 4+3)\n", same ? "ok  " : "FAIL", layers);
    if (!same) fails++;
    printf("%s n_pos agrees (%d / %d)\n", a.n_pos == b.n_pos ? "ok  " : "FAIL", a.n_pos, b.n_pos);
    if (a.n_pos != b.n_pos) fails++;
    /* continue both with one more token: the carried state must agree too */
    if (nemo_model_step_hidden(&a, 99, hc) != 0 || nemo_model_step_hidden(&b, 99, hd) != 0) { printf("FAIL follow-on step\n"); return 1; }
    same = memcmp(hc, hd, sizeof hc) == 0;
    printf("%s follow-on token after prefill bit-identical (state carried)\n", same ? "ok  " : "FAIL");
    if (!same) fails++;
    /* refusals */
    printf("%s n=0 refused\n", nemo_model_prefill(&b, toks, 0, hb) == -1 ? "ok  " : "FAIL");
    printf("%s n>chunk refused\n", nemo_model_prefill(&b, toks, NEMO_PREFILL_CHUNK + 1, hb) == -1 ? "ok  " : "FAIL");
    int bad = NEMO_VOCAB;
    printf("%s bad token refused\n", nemo_model_prefill(&b, &bad, 1, hb) == -1 ? "ok  " : "FAIL");
    printf("%s context bound refused (max_pos 64)\n",
           (b.n_pos = 63, nemo_model_prefill(&b, toks, 2, hb) == -1) ? "ok  " : "FAIL");
    nemo_model_free(&a);
    nemo_model_free(&b);
    nemo_checkpoint_close(&cp);
    printf("RESULT: %s\n", fails ? "FAILURES" : "ALL PASS");
    return fails ? 1 : 0;
}
