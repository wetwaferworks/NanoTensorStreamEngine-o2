/* score.h - tokenizer-based eval metrics for Little Nemo.
 * Teacher-forced next-token scoring over a sequence: top-1, top-5,
 * first divergence, and perplexity from a double log-softmax.
 * Argmax and top-k ties go to the lowest index (strictly greater
 * logit wins). No model code lives here.
 */
#ifndef NEMO_TOKENEVAL_SCORE_H
#define NEMO_TOKENEVAL_SCORE_H

typedef struct {
    int n;          /* positions scored */
    int top1;       /* argmax equals the true token */
    int top5;       /* true token is among the top 5 logits */
    int first_div;  /* first n-index where argmax differs, or -1 */
    double nll;     /* sum of -log p(true) */
} nemo_te_stats;

typedef struct {
    int n;
    int agree;
    int first_div;  /* first index where a[i] != b[i], or -1 */
} nemo_te_agree;

void nemo_te_init(nemo_te_stats *s);

/* Score one teacher-forced position. Writes the argmax to *pred_out
 * when pred_out is not NULL. Returns 0, or -1 if vocab/true_id is
 * unusable. Non-finite logits at the true index make NLL infinite
 * and do not count as a top-1 or top-5 hit. */
int nemo_te_add(nemo_te_stats *s, const float *logits, int vocab,
                int true_id, int *pred_out);

/* exp(nll / n), or +inf when n is 0 or nll is not finite. */
double nemo_te_ppl(const nemo_te_stats *s);

void nemo_te_agree_init(nemo_te_agree *a);
void nemo_te_agree_add(nemo_te_agree *a, int id_a, int id_b);

#endif
