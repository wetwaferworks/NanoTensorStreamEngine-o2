/* nemo_predict.h - when to leave Dense Tensor Compute (DTC) and expand to
 * the full calculation. Pure C, no state beyond the caller's struct.
 *
 * DTC is the default. Two decisions:
 *   turn level  - from the prompt alone, before any compute: prompt
 *                 length, complexity and task type (short factual vs
 *                 long-form / reasoning). EXPAND here runs the whole
 *                 turn at full precision.
 *   step level  - from the DTC logits of each step: top-1 margin and
 *                 softmax entropy. Below the confidence threshold the
 *                 step is redone at full precision (the caller restores
 *                 the state snapshot and reruns the step).
 * Every decision comes with a short reason string for the log. */
#ifndef NEMO_PREDICT_H
#define NEMO_PREDICT_H
#include <stddef.h>
#include "pool.h"

typedef struct {
    int mode;            /* 0 auto, 1 never expand (DTC only), 2 always expand */
    int prompt_tokens;   /* user prompt longer than this -> expand the turn */
    int complex_score;   /* complexity score at or above this -> expand the turn */
    int longform;        /* 1: long-form / reasoning task type -> expand the turn */
    double margin;       /* step: top-1 minus top-2 logit below this -> expand */
    double entropy;      /* step: softmax entropy (nats) above this -> expand */
} nemo_predict_cfg;

typedef struct {
    int top1;
    double margin, entropy, p1;
} nemo_confidence;

void nemo_predict_defaults(nemo_predict_cfg *c);
/* Turn decision. text = the user's message, ntok = its token count,
 * think = 1 if the user asked for reasoning (/think). Returns 1 = EXPAND. */
int nemo_predict_turn(const nemo_predict_cfg *c, const char *text, int ntok, int think,
                      char *reason, size_t rl);
/* Confidence of one logits vector (n entries). */
void nemo_confidence_of(const float *logits, int n, nemo_confidence *out);
/* The same, with the chunks spread over a pool's lanes (p may be NULL).
 * Bit-identical for any lane count. */
void nemo_confidence_of_pool(const float *logits, int n, nemo_confidence *out, nemo_pool *p);
/* Step decision from that confidence. Returns 1 = EXPAND. */
int nemo_predict_step(const nemo_predict_cfg *c, const nemo_confidence *k, char *reason, size_t rl);
#endif
