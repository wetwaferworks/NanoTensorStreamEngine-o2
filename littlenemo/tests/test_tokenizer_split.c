/* test_tokenizer_split.c - regression checks for the
 * pre-tokenizer split + ignore_merges encoding (Codex Astra
 * review H2, 2026-11-25), run against a REAL checkpoint
 * tokenizer.json: the fixture tokenizer declares no
 * pre-tokenizer, so it cannot exercise the split path.
 *
 * Usage: test_tokenizer_split <tokenizer.json>
 * With no argument the test reports a skip and exits 0, in
 * the same spirit as test_simd's synthetic-only mode: the
 * split path only exists when a real checkpoint file is
 * there to configure it.
 *
 * Expected ids are the checkpoint's own vocabulary entries:
 * 1032 = 'Ġ', 1261 = 'Ġa', 1097 = 'a', 1289 = 'Ġb'. The
 * battery was verified token-for-token against the HF
 * tokenizers library (55/55 cases) when the fix landed. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/tokenizer.h"

static int checks = 0, failures = 0;

static void expect_ids(nemo_tokenizer *tk, const char *name,
                       const char *input, const int *want,
                       int wn) {
    int ids[256];
    int n = nemo_tokenizer_encode(tk, input, ids, 256);
    checks++;
    if (n != wn || memcmp(ids, want, (size_t)wn * sizeof(int)) != 0) {
        failures++;
        printf("FAIL %s: got %d ids [", name, n);
        for (int i = 0; i < n; i++) printf("%s%d", i ? " " : "", ids[i]);
        printf("], want [");
        for (int i = 0; i < wn; i++) printf("%s%d", i ? " " : "", want[i]);
        printf("]\n");
    } else {
        printf("ok   %s\n", name);
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("RESULT: skipped (no tokenizer.json given; "
               "split path unconfigured)\n");
        return 0;
    }
    nemo_tokenizer *tk = nemo_tokenizer_load(argv[1]);
    if (!tk) {
        printf("FAIL tokenizer did not load\n");
        return 1;
    }
    /* The mandated regression: two spaces + 'a' splits into
     * " " and " a", both direct vocabulary lookups. The old
     * whole-input encoder returned [1256, 1097]. */
    expect_ids(tk, "\"  a\" -> [1032, 1261]", "  a",
               (const int[]){1032, 1261}, 2);
    expect_ids(tk, "\"a  b\" -> [1097, 1032, 1289]", "a  b",
               (const int[]){1097, 1032, 1289}, 3);
    /* Digits are single-code-point pieces under the pattern. */
    {
        int ids[256];
        int n = nemo_tokenizer_encode(tk, "abc123", ids, 256);
        checks++;
        if (n != 4 || ids[1] != 1049 || ids[2] != 1050
                || ids[3] != 1051) {
            failures++;
            printf("FAIL digits split singly: n=%d\n", n);
        } else {
            printf("ok   digits split singly\n");
        }
    }
    /* M-I4: a byte-long but token-short input must encode
     * into a small id budget: "the " x 25 is 100 bytes but
     * 26 tokens (reference count), so a budget of 32 accepts
     * it. The old encoder rejected it before BPE ran because
     * 100 bytes exceeded the budget of 32. */
    {
        char buf[101];
        for (int i = 0; i < 25; i++) memcpy(buf + 4 * i, "the ", 4);
        buf[100] = '\0';
        int ids[32];
        int n = nemo_tokenizer_encode(tk, buf, ids, 32);
        checks++;
        if (n != 26) {
            failures++;
            printf("FAIL byte-cap regression: n=%d, want 26\n", n);
        } else {
            printf("ok   byte-long token-short input fits budget\n");
        }
    }
    /* ignore_merges segmentation: repeated characters take
     * the longest vocabulary match again and again, never a
     * merged longer token... ten 'a' bytes encode as five
     * "aa" entries (17498), per the reference tokenizer. */
    expect_ids(tk, "\"aaaaaaaaaa\" -> five \"aa\"", "aaaaaaaaaa",
               (const int[]){17498, 17498, 17498, 17498, 17498}, 5);
    printf("RESULT: %d/%d checks passed\n",
           checks - failures, checks);
    nemo_tokenizer_free(tk);
    return failures ? 1 : 0;
}
