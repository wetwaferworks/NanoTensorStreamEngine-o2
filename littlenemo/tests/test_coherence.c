/* test_coherence.c - Gizmo review fix pass (2026-11-24): whole
 * geometry validation and state-commitment coherence.
 *
 * Checks:
 *  - validate passes on the layers the partial checkpoint has;
 *  - validate names the first missing tensor (head, then layer 33);
 *  - validate refuses a wrong-dtype tensor with the dtype class;
 *  - a step whose tensors are missing is refused by PREFLIGHT and
 *    commits nothing: the state afterwards is bit-identical to a
 *    model that never attempted it;
 *  - a full step with a missing head is likewise refused before
 *    the token commits;
 *  - the invalidation machinery: state_valid 0 makes steps refuse
 *    with -2 until reset, and after reset the model reproduces a
 *    fresh model's output bit for bit.
 *
 *     test_coherence <checkpoint-dir> <dtype-fixture-dir>
 */
#include <stdio.h>
#include <string.h>

#include "../src/forward.h"

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

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("usage: test_coherence <checkpoint-dir> <dtype-dir>\n");
        return 1;
    }
    nemo_checkpoint cp;
    if (nemo_checkpoint_open(&cp, argv[1]) != 0) {
        printf("FAIL checkpoint open\n");
        return 1;
    }
    char bad[128];
    int code = 0;

    expect_true("validate layers 0..14, no head",
                nemo_validate_model(&cp, 15, 0, bad, sizeof bad, &code) ==
                    0);
    code = 0;
    bad[0] = 0;
    expect_true("validate with head refuses",
                nemo_validate_model(&cp, 15, 1, bad, sizeof bad, &code) !=
                    0);
    expect_true("head failure is missing (code 1)", code == 1);
    expect_true("head failure names norm_f",
                strcmp(bad, "backbone.norm_f.weight") == 0);
    code = 0;
    bad[0] = 0;
    expect_true("validate full stack refuses",
                nemo_validate_model(&cp, 56, 0, bad, sizeof bad, &code) !=
                    0);
    /* Layer 32 straddles the shard 2/3 boundary: its norm is in
     * shard 2, its in_proj is in shard 3. First missing wins. */
    expect_true("full failure names layer 32 in_proj",
                strcmp(bad, "backbone.layers.32.mixer.in_proj.weight") ==
                    0);

    /* Wrong-dtype fixture: embeddings fine, layer 0 norm is I32. */
    nemo_checkpoint dcp;
    if (nemo_checkpoint_open(&dcp, argv[2]) != 0) {
        printf("FAIL dtype fixture open\n");
        return 1;
    }
    code = 0;
    bad[0] = 0;
    expect_true("dtype fixture refuses",
                nemo_validate_model(&dcp, 1, 0, bad, sizeof bad, &code) !=
                    0);
    expect_true("dtype failure class is 3", code == 3);
    expect_true("dtype failure names the norm tensor",
                strcmp(bad, "backbone.layers.0.norm.weight") == 0);
    nemo_checkpoint_close(&dcp);

    static float hA[NEMO_HIDDEN], hB[NEMO_HIDDEN];
    const int seq[3] = {42, 1000, 7};

    /* Preflight refusal commits nothing. */
    nemo_model a;
    if (nemo_model_init(&a, &cp, 15, 16) != 0) return 1;
    int rc = 0;
    for (int i = 0; i < 3; i++) rc |= nemo_model_step_hidden(&a, seq[i], hA);
    expect_true("model A baseline steps", rc == 0);
    a.layer_limit = 40; /* layers 33..39 are not on this bench */
    rc = nemo_model_step_hidden(&a, 5, hA);
    expect_true("over-limit step refused", rc == -1);
    expect_true("preflight refusal keeps state valid",
                a.state_valid == 1);
    a.layer_limit = 15;
    rc = nemo_model_step_hidden(&a, 5, hA);
    expect_true("continuation step runs", rc == 0);

    nemo_model b;
    if (nemo_model_init(&b, &cp, 15, 16) != 0) return 1;
    rc = 0;
    for (int i = 0; i < 3; i++) rc |= nemo_model_step_hidden(&b, seq[i], hB);
    rc |= nemo_model_step_hidden(&b, 5, hB);
    expect_true("model B reference steps", rc == 0);
    expect_true("refused step committed nothing (bit-identical)",
                memcmp(hA, hB, sizeof hA) == 0);
    nemo_model_free(&a);
    nemo_model_free(&b);

    /* Full-step head preflight: refused before the token commits. */
    nemo_model e;
    if (nemo_model_init(&e, &cp, 15, 16) != 0) return 1;
    static float logits[NEMO_VOCAB];
    rc = nemo_model_step(&e, 42, logits);
    expect_true("full step refused on missing head", rc == -1);
    expect_true("head refusal keeps state valid", e.state_valid == 1);
    static float hE[NEMO_HIDDEN], hF[NEMO_HIDDEN];
    rc = nemo_model_step_hidden(&e, 42, hE);
    expect_true("hidden step after head refusal runs", rc == 0);
    nemo_model f;
    if (nemo_model_init(&f, &cp, 15, 16) != 0) return 1;
    rc = nemo_model_step_hidden(&f, 42, hF);
    expect_true("fresh model reference step", rc == 0);
    expect_true("head refusal committed nothing (bit-identical)",
                memcmp(hE, hF, sizeof hE) == 0);
    nemo_model_free(&e);
    nemo_model_free(&f);

    /* Invalidation machinery (the mid-commit failure path itself is
     * unreachable by missing tensors now that preflight exists; the
     * flag contract is what remains to prove). */
    nemo_model c;
    if (nemo_model_init(&c, &cp, 15, 16) != 0) return 1;
    rc = nemo_model_step_hidden(&c, 42, hA);
    expect_true("model C baseline step", rc == 0);
    c.state_valid = 0; /* as a mid-commit failure would leave it */
    rc = nemo_model_step_hidden(&c, 42, hA);
    expect_true("invalid state refuses with -2", rc == -2);
    nemo_model_reset(&c);
    expect_true("reset restores validity", c.state_valid == 1);
    rc = nemo_model_step_hidden(&c, 42, hA);
    expect_true("step after reset runs", rc == 0);
    expect_true("after reset, output matches fresh model",
                memcmp(hA, hF, sizeof hA) == 0);
    nemo_model_free(&c);

    nemo_checkpoint_close(&cp);
    printf("RESULT: %d/%d checks passed\n", checks - failures, checks);
    return failures ? 1 : 0;
}
