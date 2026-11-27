/* nemo_predict.c - see nemo_predict.h. Pure C, no locks: the optional
 * confidence lanes run on the zero-mutex nemo_pool (phase handshakes). */
#include "nemo_predict.h"
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

void nemo_predict_defaults(nemo_predict_cfg *c) {
    c->mode = 0;
    c->prompt_tokens = 96;
    c->complex_score = 6;
    c->longform = 1;
    c->margin = 0.5;
    c->entropy = 4.0;
}

/* case-insensitive substring */
static int has(const char *s, const char *w) {
    size_t n = strlen(w);
    for (; *s; s++) {
        size_t i = 0;
        while (i < n && s[i] && tolower((unsigned char)s[i]) == w[i]) i++;
        if (i == n) return 1;
    }
    return 0;
}

static const char *const LONGFORM[] = {
    "essay", "write a", "write an", "story", "poem", "article", "letter to",
    "paragraph", "explain", "describe", "summarize", "summarise", "compare",
    "step by step", "step-by-step", "prove", "derive", "in detail", "outline",
    "how does", "how do ", "why does", "why do ", "pros and cons", "plan for", NULL};

int nemo_predict_turn(const nemo_predict_cfg *c, const char *text, int ntok, int think,
                      char *reason, size_t rl) {
    if (c->mode == 1) { snprintf(reason, rl, "mode=dtc-only"); return 0; }
    if (c->mode == 2) { snprintf(reason, rl, "mode=always"); return 1; }
    if (ntok > c->prompt_tokens) { snprintf(reason, rl, "prompt_tokens=%d>%d", ntok, c->prompt_tokens); return 1; }
    if (c->longform && think) { snprintf(reason, rl, "task=reasoning(/think)"); return 1; }
    if (c->longform)
        for (int i = 0; LONGFORM[i]; i++)
            if (has(text, LONGFORM[i])) { snprintf(reason, rl, "task=longform(\"%s\")", LONGFORM[i]); return 1; }
    /* complexity: sentences, clauses, numbers and operators */
    int sent = 0, comma = 0, nums = 0, ops = 0, innum = 0;
    for (const char *p = text; *p; p++) {
        if (*p == '.' || *p == '?' || *p == '!' || *p == ';') sent++;
        if (*p == ',') comma++;
        if (isdigit((unsigned char)*p)) { if (!innum) nums++; innum = 1; } else innum = 0;
        if (strchr("+-*/^=<>%", *p)) ops++;
    }
    int score = (sent > 1 ? sent - 1 : 0) + comma / 2 + (nums > 2 ? nums - 2 : 0) + ops;
    if (score >= c->complex_score) { snprintf(reason, rl, "complexity=%d>=%d", score, c->complex_score); return 1; }
    snprintf(reason, rl, "short-factual(tokens=%d,complexity=%d)", ntok, score);
    return 0;
}

/* Confidence over NEMO_CONF_CHUNKS fixed chunks, combined in chunk order:
 * the result depends only on the logits, never on how many lanes ran the
 * chunks (1, 2, 4 or 8 give identical bits). Top-1, top-2 and the margin
 * are exact; the softmax sums are double, chunk by chunk. */
#define NEMO_CONF_CHUNKS 64
typedef struct {
    const float *z; int n;
    float m1; /* phase 2 input: the global max */
    int phase;
    float cm1[NEMO_CONF_CHUNKS], cm2[NEMO_CONF_CHUNKS]; int ca[NEMO_CONF_CHUNKS];
    double cs[NEMO_CONF_CHUNKS], ct[NEMO_CONF_CHUNKS];
} conf_job;
static void conf_chunk(void *ctx, int b, int e) {
    conf_job *j = (conf_job *)ctx;
    for (int c = b; c < e; c++) {
        int i0 = (int)((long long)j->n * c / NEMO_CONF_CHUNKS), i1 = (int)((long long)j->n * (c + 1) / NEMO_CONF_CHUNKS);
        if (j->phase == 0) {
            int a = i0; float m1 = i0 < i1 ? j->z[i0] : -INFINITY, m2 = -INFINITY;
            for (int i = i0 + 1; i < i1; i++) {
                if (j->z[i] > m1) { m2 = m1; m1 = j->z[i]; a = i; }
                else if (j->z[i] > m2) m2 = j->z[i];
            }
            j->cm1[c] = m1; j->cm2[c] = m2; j->ca[c] = a;
        } else {
            double s = 0.0, t = 0.0, m = j->m1;
            for (int i = i0; i < i1; i++) { double d = (double)j->z[i] - m, x = exp(d); s += x; t += x * d; }
            j->cs[c] = s; j->ct[c] = t;
        }
    }
}
void nemo_confidence_of_pool(const float *z, int n, nemo_confidence *k, nemo_pool *p) {
    static conf_job j; /* one caller (the engine's core loop) */
    j.z = z; j.n = n; j.phase = 0;
    if (p) nemo_pool_for(p, NEMO_CONF_CHUNKS, conf_chunk, &j); else conf_chunk(&j, 0, NEMO_CONF_CHUNKS);
    int a = j.ca[0]; float m1 = j.cm1[0], m2 = j.cm2[0];
    for (int c = 1; c < NEMO_CONF_CHUNKS; c++) {   /* same tie rule as one serial pass */
        if (j.cm1[c] > m1) { m2 = m1 > j.cm2[c] ? m1 : j.cm2[c]; m1 = j.cm1[c]; a = j.ca[c]; }
        else { float o = j.cm1[c]; if (o > m2) m2 = o; }
    }
    j.m1 = m1; j.phase = 1;
    if (p) nemo_pool_for(p, NEMO_CONF_CHUNKS, conf_chunk, &j); else conf_chunk(&j, 0, NEMO_CONF_CHUNKS);
    double s = 0.0, t = 0.0;
    for (int c = 0; c < NEMO_CONF_CHUNKS; c++) { s += j.cs[c]; t += j.ct[c]; }
    k->top1 = a;
    k->margin = (double)m1 - (double)m2;
    k->p1 = 1.0 / s;
    k->entropy = log(s) - t / s;
}
void nemo_confidence_of(const float *z, int n, nemo_confidence *k) { nemo_confidence_of_pool(z, n, k, NULL); }

int nemo_predict_step(const nemo_predict_cfg *c, const nemo_confidence *k, char *reason, size_t rl) {
    if (c->mode == 1) { snprintf(reason, rl, "mode=dtc-only"); return 0; }
    if (c->mode == 2) { snprintf(reason, rl, "mode=always"); return 1; }
    if (k->margin < c->margin) { snprintf(reason, rl, "margin=%.3f<%.3f", k->margin, c->margin); return 1; }
    if (k->entropy > c->entropy) { snprintf(reason, rl, "entropy=%.3f>%.3f", k->entropy, c->entropy); return 1; }
    snprintf(reason, rl, "confident(margin=%.3f,entropy=%.3f)", k->margin, k->entropy);
    return 0;
}
