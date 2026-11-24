/* test_tokenizer.c - JSON reader and BPE tokenizer checks against a
 * hand-built fixture in the real tokenizer.json format. Every expected
 * id sequence is computed by hand from the fixture's merges:
 *   vocab a0 b1 c2 sp3 ab4 abc5, merges [a b] rank 0, [ab c] rank 1,
 *   specials <|begin_of_text|> 8, <|end_of_text|> 9.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/json.h"
#include "../src/tokenizer.h"

static int checks = 0, failures = 0;

static void expect_true(const char *name, int cond) {
    checks++;
    if (!cond) {
        failures++;
        printf("FAIL %s\n", name);
    } else {
        printf("ok   %s\n", name);
    }
}

static void expect_ids(const char *name, const int *got, int n,
                       const int *want, int wn) {
    checks++;
    if (n != wn || memcmp(got, want, (size_t)wn * sizeof(int)) != 0) {
        failures++;
        printf("FAIL %s: got %d ids, want %d\n", name, n, wn);
    } else {
        printf("ok   %s\n", name);
    }
}

static void expect_str(const char *name, const char *got,
                       const char *want) {
    checks++;
    if (!got || strcmp(got, want) != 0) {
        failures++;
        printf("FAIL %s: got %s want %s\n", name, got ? got : "(null)",
               want);
    } else {
        printf("ok   %s\n", name);
    }
}

static void test_json(void) {
    const char *doc =
        "{\"a\": [1, 2.5, true, null, \"x\\u0041\\ud83d\\ude00\"], "
        "\"b\": {\"c\": \"line\\nbreak\"}}";
    nemo_json *j = nemo_json_parse(doc, strlen(doc));
    expect_true("json parses", j != NULL);
    if (j) {
        const nemo_json *a = nemo_json_get(j, "a");
        expect_true("json array len", a && a->count == 5);
        expect_true("json num", a && a->items[1]->num == 2.5);
        expect_true("json bool", a && a->items[2]->num == 1.0);
        expect_true("json null type",
                    a && a->items[3]->type == NEMO_JSON_NULL);
        /* "xA" + U+1F600 in UTF-8 (f0 9f 98 80). */
        expect_str("json \\u + surrogate pair", a ? a->items[4]->str : NULL,
                   "xA\xf0\x9f\x98\x80");
        const nemo_json *b = nemo_json_get(j, "b");
        const nemo_json *c = b ? nemo_json_get(b, "c") : NULL;
        expect_str("json nested + \\n", c ? c->str : NULL, "line\nbreak");
        nemo_json_free(j);
    }
    expect_true("json rejects trailing garbage",
                nemo_json_parse("{} x", 4) == NULL);
    expect_true("json rejects unterminated string",
                nemo_json_parse("{\"a\": \"x}", 9) == NULL);
    expect_true("json rejects lone surrogate",
                nemo_json_parse("\"\\ud83d\"", 8) == NULL);
    expect_true("json rejects empty", nemo_json_parse("", 0) == NULL);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("usage: test_tokenizer <fixture.json>\n");
        return 1;
    }
    test_json();

    nemo_tokenizer *tk = nemo_tokenizer_load(argv[1]);
    expect_true("tokenizer loads", tk != NULL);
    if (!tk) {
        printf("RESULT: %d/%d checks passed\n", checks - failures, checks);
        return 1;
    }
    expect_true("vocab size 10", nemo_tokenizer_vocab_size(tk) == 10);
    expect_true("special bos", nemo_tokenizer_special(tk, "<|begin_of_text|>") == 8);
    expect_true("special eos", nemo_tokenizer_special(tk, "<|end_of_text|>") == 9);
    expect_true("plain token is not special",
                nemo_tokenizer_special(tk, "a") == -1);
    expect_true("unknown special", nemo_tokenizer_special(tk, "<nope>") == -1);

    int ids[64];
    int n = nemo_tokenizer_encode(tk, "abc", ids, 64);
    expect_ids("encode abc -> [abc]", ids, n, (int[]){5}, 1);
    n = nemo_tokenizer_encode(tk, "ab a", ids, 64);
    expect_ids("encode 'ab a'", ids, n, (int[]){4, 3, 0}, 3);
    n = nemo_tokenizer_encode(tk, "c", ids, 64);
    expect_ids("encode c", ids, n, (int[]){2}, 1);
    n = nemo_tokenizer_encode(tk, "", ids, 64);
    expect_true("encode empty", n == 0);
    n = nemo_tokenizer_encode(tk, "abcab", ids, 64);
    expect_ids("encode abcab", ids, n, (int[]){5, 4}, 2);

    char *s = nemo_tokenizer_decode(tk, (int[]){5}, 1, 0);
    expect_str("decode [abc]", s, "abc");
    free(s);
    s = nemo_tokenizer_decode(tk, (int[]){4, 3, 0}, 3, 0);
    expect_str("decode 'ab a'", s, "ab a");
    free(s);
    /* Token 7 is the two-byte mapped string for UTF-8 e-acute. */
    s = nemo_tokenizer_decode(tk, (int[]){7}, 1, 0);
    expect_str("decode e-acute bytes", s, "\xc3\xa9");
    free(s);
    s = nemo_tokenizer_decode(tk, (int[]){8, 5, 9}, 3, 0);
    expect_str("decode with specials", s,
               "<|begin_of_text|>abc<|end_of_text|>");
    free(s);
    s = nemo_tokenizer_decode(tk, (int[]){8, 5, 9}, 3, 1);
    expect_str("decode skipping specials", s, "abc");
    free(s);

    nemo_tokenizer_free(tk);
    printf("RESULT: %d/%d checks passed\n", checks - failures, checks);
    return failures ? 1 : 0;
}
