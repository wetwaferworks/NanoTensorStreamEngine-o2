/* score.c - see score.h */
#include "score.h"

#include <math.h>

void nemo_te_init(nemo_te_stats *s)
{
    s->n = 0;
    s->top1 = 0;
    s->top5 = 0;
    s->first_div = -1;
    s->nll = 0.0;
}

static int argmax_f(const float *logits, int vocab)
{
    int best = 0;
    float m = logits[0];
    for (int i = 1; i < vocab; i++) {
        if (logits[i] > m) {
            m = logits[i];
            best = i;
        }
    }
    return best;
}

/* How many logits strictly beat `target`, plus how many equal it
 * with a lower index. That is the 0-based rank under our tie rule. */
static int rank_of(const float *logits, int vocab, int target)
{
    float t = logits[target];
    int rank = 0;
    for (int i = 0; i < vocab; i++) {
        if (logits[i] > t) rank++;
        else if (logits[i] == t && i < target) rank++;
    }
    return rank;
}

int nemo_te_add(nemo_te_stats *s, const float *logits, int vocab,
                int true_id, int *pred_out)
{
    if (!s || !logits || vocab < 1 || true_id < 0 || true_id >= vocab)
        return -1;
    int pred = argmax_f(logits, vocab);
    if (pred_out) *pred_out = pred;

    double nll;
    if (!isfinite((double)logits[true_id])) {
        nll = INFINITY;
    } else {
        double m = (double)logits[0];
        for (int i = 1; i < vocab; i++) {
            double v = (double)logits[i];
            if (v > m) m = v;
        }
        if (!isfinite(m)) {
            nll = INFINITY;
        } else {
            double z = 0.0;
            for (int i = 0; i < vocab; i++)
                z += exp((double)logits[i] - m);
            nll = -((double)logits[true_id] - m - log(z));
        }
    }

    s->n++;
    s->nll += nll;
    if (isfinite(nll) && pred == true_id) s->top1++;
    if (isfinite(nll) && rank_of(logits, vocab, true_id) < 5) s->top5++;
    if (pred != true_id && s->first_div < 0)
        s->first_div = s->n - 1;
    return 0;
}

double nemo_te_ppl(const nemo_te_stats *s)
{
    if (!s || s->n <= 0 || !isfinite(s->nll)) return INFINITY;
    return exp(s->nll / (double)s->n);
}

void nemo_te_agree_init(nemo_te_agree *a)
{
    a->n = 0;
    a->agree = 0;
    a->first_div = -1;
}

void nemo_te_agree_add(nemo_te_agree *a, int id_a, int id_b)
{
    a->n++;
    if (id_a == id_b) {
        a->agree++;
    } else if (a->first_div < 0) {
        a->first_div = a->n - 1;
    }
}
