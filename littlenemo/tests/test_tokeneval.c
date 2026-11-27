/* test_tokeneval.c - tokenizer-based evaluation for Little Nemo.
 *
 * Default (unit tests, no checkpoint):
 *     test_tokeneval <fixture_tokenizer.json>
 *
 * Tokenizer-only, against the committed passage and optional gold ids
 * (still no model; pass a local tokenizer.json):
 *     test_tokeneval --tokenizer tokenizer.json
 *                    --text tests/tokeneval/passage.txt
 *                    [--ids tests/tokeneval/passage.ids]
 *                    [--write-ids passage.ids]
 *
 * Teacher-forced live eval (heavy: one full forward per token; do not
 * run this as an all-core marathon while the machine is busy):
 *     test_tokeneval --checkpoint DIR --tokenizer tokenizer.json
 *                    --text tests/tokeneval/passage.txt
 *                    [--max-tokens N] [--threads 1] [--out report.txt]
 *
 * Quality is scored on the decoded text and on every next-token
 * decision along the passage, not on the top-1 of a single step.
 *
 * Direction by JD. Co-implementation by Cortana, built with Grok Bot
 * by SpaceX.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tokeneval/score.h"
#include "../src/tokenizer.h"
#include "../src/forward.h"

static int checks, failures;

static void expect_true(const char *name, int cond)
{
    checks++;
    if (!cond) {
        failures++;
        printf("FAIL %s\n", name);
    } else {
        printf("ok   %s\n", name);
    }
}

static void expect_near(const char *name, double got, double want, double eps)
{
    int ok = isfinite(got) && isfinite(want) && fabs(got - want) <= eps;
    expect_true(name, ok);
    if (!ok)
        printf("     got %.10g want %.10g\n", got, want);
}

static char *read_file(const char *path, size_t *len_out)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz < 0) { fclose(f); return NULL; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    char *b = (char *)malloc((size_t)sz + 1);
    if (!b) { fclose(f); return NULL; }
    size_t n = fread(b, 1, (size_t)sz, f);
    fclose(f);
    b[n] = 0;
    if (len_out) *len_out = n;
    return b;
}

static int *load_ids(const char *path, int *n_out)
{
    FILE *f = fopen(path, "r");
    if (!f) return NULL;
    int cap = 64, n = 0;
    int *ids = (int *)malloc((size_t)cap * sizeof(int));
    if (!ids) { fclose(f); return NULL; }
    char line[256];
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r')
            continue;
        int v;
        if (sscanf(line, "%d", &v) != 1) {
            free(ids);
            fclose(f);
            return NULL;
        }
        if (n == cap) {
            cap *= 2;
            int *nids = (int *)realloc(ids, (size_t)cap * sizeof(int));
            if (!nids) { free(ids); fclose(f); return NULL; }
            ids = nids;
        }
        ids[n++] = v;
    }
    fclose(f);
    *n_out = n;
    return ids;
}

static int write_ids(const char *path, const int *ids, int n)
{
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "# Little Nemo tokenizer ids for tests/tokeneval/passage.txt\n");
    fprintf(f, "# One id per line. Re-derive with test_tokeneval --tokenizer "
               "tokenizer.json --text tests/tokeneval/passage.txt --write-ids "
               "this-file\n");
    for (int i = 0; i < n; i++)
        fprintf(f, "%d\n", ids[i]);
    fclose(f);
    return 0;
}

static void test_score(void)
{
    const int V = 8;
    float z[8];
    nemo_te_stats s;
    int pred;

    /* A sharp correct peak: logit[3] = 20, others 0. */
    memset(z, 0, sizeof z);
    z[3] = 20.f;
    nemo_te_init(&s);
    expect_true("add sharp peak", nemo_te_add(&s, z, V, 3, &pred) == 0);
    expect_true("sharp argmax is 3", pred == 3);
    expect_true("sharp top-1", s.top1 == 1 && s.n == 1);
    expect_true("sharp top-5", s.top5 == 1);
    expect_true("sharp no divergence", s.first_div < 0);
    expect_near("sharp ppl near 1", nemo_te_ppl(&s), 1.0, 1e-6);

    /* Wrong argmax, but true token still in the top 5. */
    memset(z, 0, sizeof z);
    z[0] = 5.f;
    z[1] = 4.f;
    z[2] = 3.f;
    z[7] = 2.f; /* true */
    z[4] = 1.f;
    nemo_te_init(&s);
    nemo_te_add(&s, z, V, 7, &pred);
    expect_true("missed argmax is 0", pred == 0);
    expect_true("missed not top-1", s.top1 == 0);
    expect_true("missed still top-5", s.top5 == 1);
    expect_true("first div at 0", s.first_div == 0);

    /* True token is sixth: not top-5. */
    memset(z, 0, sizeof z);
    z[0] = 6.f; z[1] = 5.f; z[2] = 4.f; z[3] = 3.f; z[4] = 2.f;
    z[5] = 1.f; /* true, rank 5 */
    nemo_te_init(&s);
    nemo_te_add(&s, z, V, 5, &pred);
    expect_true("sixth is not top-5", s.top5 == 0 && s.top1 == 0);

    /* Tie: two equal maxes, lowest index wins. */
    memset(z, 0, sizeof z);
    z[2] = 1.f;
    z[5] = 1.f;
    nemo_te_init(&s);
    nemo_te_add(&s, z, V, 2, &pred);
    expect_true("tie argmax is lowest index", pred == 2);
    expect_true("tie lowest index is top-1", s.top1 == 1);
    nemo_te_init(&s);
    nemo_te_add(&s, z, V, 5, &pred);
    expect_true("tie higher index is not top-1", pred == 2 && s.top1 == 0);
    expect_true("tie higher index still top-5", s.top5 == 1);

    /* Two positions: first matches, second diverges; ppl is finite. */
    nemo_te_init(&s);
    memset(z, 0, sizeof z);
    z[1] = 10.f;
    nemo_te_add(&s, z, V, 1, NULL);
    memset(z, 0, sizeof z);
    z[0] = 10.f;
    nemo_te_add(&s, z, V, 1, NULL);
    expect_true("two-step counts", s.n == 2 && s.top1 == 1 && s.top5 == 2);
    expect_true("two-step first div at 1", s.first_div == 1);
    expect_true("two-step ppl finite and > 1",
                isfinite(nemo_te_ppl(&s)) && nemo_te_ppl(&s) > 1.0);

    /* Manual NLL: vocab-2, logits {0, 0}, true 0 -> p = 0.5, nll = log(2). */
    {
        float e[2] = {0.f, 0.f};
        nemo_te_init(&s);
        nemo_te_add(&s, e, 2, 0, NULL);
        expect_near("balanced two-class nll is log(2)", s.nll, log(2.0), 1e-9);
        expect_near("balanced two-class ppl is 2", nemo_te_ppl(&s), 2.0, 1e-9);
    }

    /* Sequence agreement. */
    {
        nemo_te_agree a;
        nemo_te_agree_init(&a);
        nemo_te_agree_add(&a, 10, 10);
        nemo_te_agree_add(&a, 11, 11);
        nemo_te_agree_add(&a, 12, 99);
        nemo_te_agree_add(&a, 13, 13);
        expect_true("agree 3 of 4", a.n == 4 && a.agree == 3);
        expect_true("agree first div at 2", a.first_div == 2);
    }

    nemo_te_init(&s);
    expect_true("reject bad true id", nemo_te_add(&s, z, V, -1, NULL) == -1);
}

static void test_fixture_tokenizer(const char *path)
{
    nemo_tokenizer *tk = nemo_tokenizer_load(path);
    expect_true("fixture tokenizer loads", tk != NULL);
    if (!tk) return;

    /* "a c" is two (or more) fixture tokens; "abc" merges to one. */
    const char *text = "a c";
    int ids[16];
    int n = nemo_tokenizer_encode(tk, text, ids, 16);
    expect_true("fixture encode a c", n > 0);
    if (n > 0) {
        char *dec = nemo_tokenizer_decode(tk, ids, n, 0);
        expect_true("fixture decode non-NULL", dec != NULL);
        if (dec) {
            expect_true("fixture encode/decode returns the source text",
                        strcmp(dec, text) == 0);
            free(dec);
        }
        /* Score a synthetic teacher-forced run on those ids, then
         * decode the predicted ids as text (the point of this eval). */
        if (n >= 2) {
            const int v = 16;
            float z[16];
            int pred[16];
            nemo_te_stats s;
            nemo_te_init(&s);
            int scored = 0;
            for (int i = 0; i < n - 1; i++) {
                memset(z, 0, sizeof z);
                if (ids[i + 1] >= 0 && ids[i + 1] < v)
                    z[ids[i + 1]] = 8.f;
                if (nemo_te_add(&s, z, v, ids[i + 1], &pred[i]) == 0)
                    scored++;
            }
            expect_true("fixture sequence scored", scored == n - 1);
            expect_true("fixture synthetic is all top-1",
                        s.top1 == scored && s.first_div < 0);
            char *from_pred = nemo_tokenizer_decode(tk, pred, scored, 0);
            char *from_true = nemo_tokenizer_decode(tk, ids + 1, scored, 0);
            expect_true("decoded prediction equals decoded true suffix",
                        from_pred && from_true &&
                        strcmp(from_pred, from_true) == 0);
            free(from_pred);
            free(from_true);
        }
    }
    nemo_tokenizer_free(tk);
}

static void test_passage_file(void)
{
    const char *path = "tests/tokeneval/passage.txt";
    size_t n = 0;
    char *t = read_file(path, &n);
    if (!t) {
        t = read_file("littlenemo/tests/tokeneval/passage.txt", &n);
        path = "littlenemo/tests/tokeneval/passage.txt";
    }
    expect_true("passage.txt is readable", t != NULL && n > 0);
    if (t) {
        expect_true("passage is UTF-8 text without NUL",
                    (int)strlen(t) == (int)n);
        expect_true("passage mentions tokenizer-based evaluation",
                    strstr(t, "tokenizer-based") != NULL);
        expect_true("passage is original to this repository",
                    strstr(t, "original to this repository") != NULL);
        free(t);
    }
}

static int unit_main(const char *fixture)
{
    test_score();
    test_fixture_tokenizer(fixture);
    test_passage_file();
    printf("RESULT: %d/%d checks passed\n", checks - failures, checks);
    return failures ? 1 : 0;
}

static void usage(void)
{
    fprintf(stderr,
        "usage: test_tokeneval <fixture_tokenizer.json>\n"
        "       test_tokeneval --tokenizer FILE --text FILE\n"
        "                      [--ids FILE] [--write-ids FILE]\n"
        "                      [--checkpoint DIR] [--max-tokens N]\n"
        "                      [--threads N] [--out FILE]\n");
}

static int eval_tokenizer_only(nemo_tokenizer *tk, const char *text,
                               const char *ids_path, const char *write_ids_path)
{
    int cap = (int)strlen(text) + 8;
    if (cap < 16) cap = 16;
    int *ids = (int *)malloc((size_t)cap * sizeof(int));
    if (!ids) { fprintf(stderr, "oom\n"); return 1; }
    int n = nemo_tokenizer_encode(tk, text, ids, cap);
    if (n < 0) {
        /* grow once for a dense byte encoding */
        cap = (int)strlen(text) * 4 + 16;
        int *nids = (int *)realloc(ids, (size_t)cap * sizeof(int));
        if (!nids) { free(ids); fprintf(stderr, "oom\n"); return 1; }
        ids = nids;
        n = nemo_tokenizer_encode(tk, text, ids, cap);
    }
    if (n < 0) {
        fprintf(stderr, "encode failed (buffer or unknown byte)\n");
        free(ids);
        return 1;
    }
    char *dec = nemo_tokenizer_decode(tk, ids, n, 0);
    int ident = dec && strcmp(dec, text) == 0;
    printf("tokens %d\n", n);
    printf("encode/decode identity: %s\n", ident ? "yes" : "no");
    if (!ident && dec)
        printf("decoded (%zu bytes) differs from the source text (%zu bytes)\n",
               strlen(dec), strlen(text));
    if (write_ids_path) {
        if (write_ids(write_ids_path, ids, n) != 0) {
            fprintf(stderr, "could not write %s\n", write_ids_path);
            free(dec);
            free(ids);
            return 1;
        }
        printf("wrote %s\n", write_ids_path);
    }
    if (ids_path) {
        int gn = 0;
        int *gold = load_ids(ids_path, &gn);
        if (!gold) {
            fprintf(stderr, "could not read gold ids %s\n", ids_path);
            free(dec);
            free(ids);
            return 1;
        }
        nemo_te_agree a;
        nemo_te_agree_init(&a);
        int m = n < gn ? n : gn;
        for (int i = 0; i < m; i++)
            nemo_te_agree_add(&a, ids[i], gold[i]);
        printf("gold ids: %d  got: %d  agree: %d/%d  first_div: %d\n",
               gn, n, a.agree, a.n, a.first_div);
        int ok = (n == gn && a.agree == a.n);
        printf("gold match: %s\n", ok ? "yes" : "no");
        free(gold);
        if (!ok) { free(dec); free(ids); return 1; }
    }
    free(dec);
    free(ids);
    return 0;
}

static int eval_live(nemo_tokenizer *tk, const char *text,
                     const char *ckpt, int max_tokens, int threads,
                     const char *out_path)
{
    int cap = (int)strlen(text) * 4 + 16;
    int *ids = (int *)malloc((size_t)cap * sizeof(int));
    if (!ids) { fprintf(stderr, "oom\n"); return 1; }
    int n = nemo_tokenizer_encode(tk, text, ids, cap);
    if (n < 0) {
        fprintf(stderr, "encode failed\n");
        free(ids);
        return 1;
    }
    if (max_tokens > 0 && n > max_tokens) n = max_tokens;
    if (n < 2) {
        fprintf(stderr, "need at least 2 tokens to score a next-token step\n");
        free(ids);
        return 1;
    }
    if (threads < 1) threads = 1;

    nemo_checkpoint cp;
    if (nemo_checkpoint_open(&cp, ckpt) != 0) {
        fprintf(stderr, "checkpoint open failed\n");
        free(ids);
        return 1;
    }
    nemo_model m;
    if (nemo_model_init_ex(&m, &cp, 0, n + 8, threads) != 0) {
        fprintf(stderr, "model init failed\n");
        nemo_checkpoint_close(&cp);
        free(ids);
        return 1;
    }

    float *logits = (float *)malloc((size_t)NEMO_VOCAB * sizeof(float));
    int *pred = (int *)malloc((size_t)(n - 1) * sizeof(int));
    if (!logits || !pred) {
        fprintf(stderr, "oom\n");
        free(pred); free(logits);
        nemo_model_free(&m);
        nemo_checkpoint_close(&cp);
        free(ids);
        return 1;
    }

    nemo_te_stats s;
    nemo_te_init(&s);
    int rc = 0;
    for (int i = 0; i < n - 1; i++) {
        rc = nemo_model_step(&m, ids[i], logits);
        if (rc != 0) {
            fprintf(stderr, "step %d failed (rc %d)\n", i, rc);
            break;
        }
        if (nemo_te_add(&s, logits, NEMO_VOCAB, ids[i + 1], &pred[i]) != 0) {
            fprintf(stderr, "score add failed at %d (id %d)\n", i, ids[i + 1]);
            rc = -1;
            break;
        }
    }

    char *true_txt = nemo_tokenizer_decode(tk, ids + 1, s.n, 0);
    char *pred_txt = nemo_tokenizer_decode(tk, pred, s.n, 0);
    int text_same = (true_txt && pred_txt && strcmp(true_txt, pred_txt) == 0);

    printf("tokens_scored %d  top1 %d/%d  top5 %d/%d  first_div %d  ppl %.6f\n",
           s.n, s.top1, s.n, s.top5, s.n, s.first_div, nemo_te_ppl(&s));
    printf("decoded_text_identical %s\n", text_same ? "yes" : "no");
    if (pred_txt)
        printf("predicted_text: %s\n", pred_txt);
    if (out_path) {
        FILE *f = fopen(out_path, "w");
        if (f) {
            fprintf(f, "tokens_scored %d\n", s.n);
            fprintf(f, "top1 %d\ntop5 %d\nfirst_div %d\n",
                    s.top1, s.top5, s.first_div);
            fprintf(f, "nll %.10g\nppl %.10g\n", s.nll, nemo_te_ppl(&s));
            fprintf(f, "decoded_text_identical %s\n", text_same ? "yes" : "no");
            fprintf(f, "predicted_text: %s\n", pred_txt ? pred_txt : "");
            fprintf(f, "true_suffix: %s\n", true_txt ? true_txt : "");
            fclose(f);
            printf("wrote %s\n", out_path);
        }
    }

    free(true_txt);
    free(pred_txt);
    free(pred);
    free(logits);
    nemo_model_free(&m);
    nemo_checkpoint_close(&cp);
    free(ids);
    return rc != 0 ? 1 : 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) { usage(); return 2; }
    if (argv[1][0] != '-')
        return unit_main(argv[1]);

    const char *tokenizer = NULL, *text_path = NULL, *ids_path = NULL;
    const char *write_ids_path = NULL, *ckpt = NULL, *out_path = NULL;
    int max_tokens = 0, threads = 1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--tokenizer") && i + 1 < argc)
            tokenizer = argv[++i];
        else if (!strcmp(argv[i], "--text") && i + 1 < argc)
            text_path = argv[++i];
        else if (!strcmp(argv[i], "--ids") && i + 1 < argc)
            ids_path = argv[++i];
        else if (!strcmp(argv[i], "--write-ids") && i + 1 < argc)
            write_ids_path = argv[++i];
        else if (!strcmp(argv[i], "--checkpoint") && i + 1 < argc)
            ckpt = argv[++i];
        else if (!strcmp(argv[i], "--max-tokens") && i + 1 < argc)
            max_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc)
            threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out") && i + 1 < argc)
            out_path = argv[++i];
        else {
            usage();
            return 2;
        }
    }
    if (!tokenizer || !text_path) { usage(); return 2; }
    if (threads < 1) threads = 1;

    char *text = read_file(text_path, NULL);
    if (!text) {
        fprintf(stderr, "could not read %s\n", text_path);
        return 1;
    }
    nemo_tokenizer *tk = nemo_tokenizer_load(tokenizer);
    if (!tk) {
        fprintf(stderr, "could not load tokenizer %s\n", tokenizer);
        free(text);
        return 1;
    }

    int rc;
    if (ckpt) {
        rc = eval_live(tk, text, ckpt, max_tokens, threads, out_path);
    } else {
        rc = eval_tokenizer_only(tk, text, ids_path, write_ids_path);
    }
    nemo_tokenizer_free(tk);
    free(text);
    return rc;
}
