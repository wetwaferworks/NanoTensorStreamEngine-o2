/* nemo_predict.c - see nemo_predict.h. Pure C, no threads, no locks. */
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

void nemo_confidence_of(const float *z, int n, nemo_confidence *k) {
    int a = 0; float m1 = z[0], m2 = -INFINITY;
    for (int i = 1; i < n; i++) {
        if (z[i] > m1) { m2 = m1; m1 = z[i]; a = i; }
        else if (z[i] > m2) m2 = z[i];
    }
    double s = 0.0, t = 0.0;
    for (int i = 0; i < n; i++) { double e = exp((double)z[i] - m1); s += e; t += e * ((double)z[i] - m1); }
    k->top1 = a;
    k->margin = (double)m1 - (double)m2;
    k->p1 = 1.0 / s;
    k->entropy = log(s) - t / s;
}

int nemo_predict_step(const nemo_predict_cfg *c, const nemo_confidence *k, char *reason, size_t rl) {
    if (c->mode == 1) { snprintf(reason, rl, "mode=dtc-only"); return 0; }
    if (c->mode == 2) { snprintf(reason, rl, "mode=always"); return 1; }
    if (k->margin < c->margin) { snprintf(reason, rl, "margin=%.3f<%.3f", k->margin, c->margin); return 1; }
    if (k->entropy > c->entropy) { snprintf(reason, rl, "entropy=%.3f>%.3f", k->entropy, c->entropy); return 1; }
    snprintf(reason, rl, "confident(margin=%.3f,entropy=%.3f)", k->margin, k->entropy);
    return 0;
}
