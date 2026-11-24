/* engine.c - the Little Nemo engine binary (protocol v2, see
 * littlenemo/PROTOCOL.md). Pure C. Loads a Nemotron-Nano-9B-v2
 * checkpoint directory plus a tokenizer.json, speaks the line
 * protocol on stdin/stdout, and generates greedily: deterministic
 * argmax decoding, the same discipline the PCHIE battery scores.
 *
 *   nemo-engine <checkpoint_dir> <tokenizer.json>
 *               [--layers N] [--max-pos N] [--max-new N]
 *
 * --layers truncates the stack (plumbing tests only, not a model
 * output). Chat-template wrapping arrives with tokenizer_config.json;
 * until then PROMPT text is encoded as-is.
 *
 * Telemetry honesty: ms_tok and tps are measured with a monotonic
 * clock; gbs is an ESTIMATE (weight bytes the step touches, summed
 * from the checkpoint index, over measured time) and is labeled an
 * estimate in the protocol. staged is 0: this engine consumes
 * serially and prefetches nothing yet.
 */
#define _POSIX_C_SOURCE 200809L
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "forward.h"
#include "loader.h"
#include "tokenizer.h"

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static long peak_kb(void) {
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256];
    long kb = 0;
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "VmHWM: %ld kB", &kb) == 1) break;
    }
    fclose(f);
    return kb;
}

/* ---- protocol text escaping ---- */

static void emit_escaped(const char *verb, const char *text, size_t n) {
    fputs(verb, stdout);
    fputc(' ', stdout);
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)text[i];
        if (c == '\\') fputs("\\\\", stdout);
        else if (c == '\n') fputs("\\n", stdout);
        else if (c == '\t') fputs("\\t", stdout);
        else fputc(c, stdout);
    }
    fputc('\n', stdout);
    fflush(stdout);
}

static void unescape(char *s) {
    char *r = s, *w = s;
    while (*r) {
        if (*r == '\\' && r[1]) {
            r++;
            *w++ = (*r == 'n') ? '\n' : (*r == 't') ? '\t' : *r;
            r++;
        } else {
            *w++ = *r++;
        }
    }
    *w = 0;
}

/* ---- layer telemetry hook ---- */

static const char *type_name(char t) {
    return t == 'M' ? "mamba" : t == '*' ? "attention" : "mlp";
}

static void on_layer(void *user, int layer, char t) {
    (void)user;
    printf("LAYER %d %s\n", layer, type_name(t));
    fflush(stdout);
}

/* ---- think/answer stream splitting ---- */

#define THINK_OPEN "<think>"
#define THINK_CLOSE "</think>"

typedef struct {
    char buf[8192]; /* decoded bytes not yet emitted */
    size_t len;
    int thinking;   /* current mode */
} splitter;

static size_t utf8_prefix(const char *s, size_t n) {
    /* Longest prefix of s[0..n) ending on a UTF-8 boundary. */
    if (n == 0) return 0;
    size_t i = n - 1;
    while (i > 0 && ((unsigned char)s[i] & 0xC0) == 0x80) i--;
    unsigned char lead = (unsigned char)s[i];
    size_t need = lead < 0x80 ? 1 : lead < 0xE0 ? 2 : lead < 0xF0 ? 3 : 4;
    return (i + need <= n) ? n : i;
}

static void split_flush(splitter *sp, int final) {
    for (;;) {
        const char *delim = sp->thinking ? THINK_CLOSE : THINK_OPEN;
        size_t dlen = strlen(delim);
        /* find the delimiter in the pending bytes */
        size_t at = (size_t)-1;
        if (sp->len >= dlen) {
            for (size_t i = 0; i + dlen <= sp->len; i++) {
                if (memcmp(sp->buf + i, delim, dlen) == 0) { at = i; break; }
            }
        }
        if (at != (size_t)-1) {
            if (at > 0)
                emit_escaped(sp->thinking ? "THINK" : "TOK", sp->buf, at);
            memmove(sp->buf, sp->buf + at + dlen, sp->len - at - dlen);
            sp->len -= at + dlen;
            sp->thinking = !sp->thinking;
            continue;
        }
        /* No delimiter: hold back a tail that could be a delimiter
         * prefix split across tokens, plus any partial UTF-8 tail. */
        size_t hold = 0;
        if (!final) {
            size_t maxk = sp->len < dlen - 1 ? sp->len : dlen - 1;
            for (size_t k = maxk; k > 0; k--) {
                if (memcmp(sp->buf + sp->len - k, delim, k) == 0) {
                    hold = k;
                    break;
                }
            }
        }
        size_t emit = utf8_prefix(sp->buf, sp->len - hold);
        if (emit == 0) return;
        emit_escaped(sp->thinking ? "THINK" : "TOK", sp->buf, emit);
        memmove(sp->buf, sp->buf + emit, sp->len - emit);
        sp->len -= emit;
        return;
    }
}

static void split_feed(splitter *sp, const char *bytes) {
    size_t n = strlen(bytes);
    if (sp->len + n >= sizeof sp->buf) split_flush(sp, 1);
    if (n >= sizeof sp->buf) { /* pathological single token: emit raw */
        emit_escaped(sp->thinking ? "THINK" : "TOK", bytes, n);
        return;
    }
    memcpy(sp->buf + sp->len, bytes, n);
    sp->len += n;
    split_flush(sp, 0);
}

/* ---- weight bytes per token (for the gbs estimate) ---- */

static uint64_t tensor_bytes(const nemo_checkpoint *cp, const char *name) {
    const nemo_tensor_info *t = nemo_find(cp, name);
    return t ? t->count * (uint64_t)t->elem_size : 0;
}

static uint64_t bytes_per_token(const nemo_checkpoint *cp, int limit) {
    char name[96];
    uint64_t sum = tensor_bytes(cp, "lm_head.weight") +
                   (uint64_t)NEMO_HIDDEN * 2; /* one embedding row */
    for (int i = 0; i < limit; i++) {
        char t = nemo_layer_type(i);
        snprintf(name, sizeof name, "backbone.layers.%d.norm.weight", i);
        sum += tensor_bytes(cp, name);
        if (t == 'M') {
            const char *leaves[] = {"mixer.in_proj.weight",
                                    "mixer.out_proj.weight",
                                    "mixer.conv1d.weight",
                                    "mixer.conv1d.bias", "mixer.A_log",
                                    "mixer.D", "mixer.dt_bias",
                                    "mixer.norm.weight"};
            for (size_t k = 0; k < sizeof leaves / sizeof *leaves; k++) {
                snprintf(name, sizeof name, "backbone.layers.%d.%s", i,
                         leaves[k]);
                sum += tensor_bytes(cp, name);
            }
        } else if (t == '*') {
            const char *leaves[] = {"mixer.q_proj.weight",
                                    "mixer.k_proj.weight",
                                    "mixer.v_proj.weight",
                                    "mixer.o_proj.weight"};
            for (size_t k = 0; k < sizeof leaves / sizeof *leaves; k++) {
                snprintf(name, sizeof name, "backbone.layers.%d.%s", i,
                         leaves[k]);
                sum += tensor_bytes(cp, name);
            }
        } else {
            snprintf(name, sizeof name, "backbone.layers.%d.mixer.up_proj.weight", i);
            sum += tensor_bytes(cp, name);
            snprintf(name, sizeof name, "backbone.layers.%d.mixer.down_proj.weight", i);
            sum += tensor_bytes(cp, name);
        }
    }
    return sum;
}

/* ---- generation ---- */

typedef struct {
    nemo_checkpoint *cp;
    nemo_tokenizer *tk;
    nemo_model *model;
    int max_new;
    int eos_id;
    uint64_t bytes_tok;
    char *pending_cmd; /* command read while generating (STOP handled) */
} engine;

/* Returns 1 if a command line was waiting on stdin (it is consumed
 * into e->pending_cmd; the caller checks for STOP). */
static int poll_command(engine *e) {
    struct pollfd pfd;
    pfd.fd = 0;
    pfd.events = POLLIN;
    if (poll(&pfd, 1, 0) <= 0 || !(pfd.revents & POLLIN)) return 0;
    char *line = NULL;
    size_t cap = 0;
    if (getline(&line, &cap, stdin) < 0) {
        free(line);
        e->pending_cmd = NULL;
        return 0;
    }
    e->pending_cmd = line;
    return 1;
}

static int argmax(const float *logits) {
    int best = 0;
    float bv = logits[0];
    for (int i = 1; i < NEMO_VOCAB; i++) {
        if (logits[i] > bv) {
            bv = logits[i];
            best = i;
        }
    }
    return best;
}

static void run_turn(engine *e, const char *prompt_text) {
    int ids[8192];
    int n = nemo_tokenizer_encode(e->tk, prompt_text, ids, 8192);
    if (n < 0) {
        printf("ERROR prompt could not be tokenized\n");
        fflush(stdout);
        return;
    }
    float *logits = (float *)malloc((size_t)NEMO_VOCAB * sizeof(float));
    float *hidden = (float *)malloc((size_t)NEMO_HIDDEN * sizeof(float));
    if (!logits || !hidden) {
        free(logits);
        free(hidden);
        printf("ERROR out of memory\n");
        fflush(stdout);
        return;
    }
    double turn_t0 = now_ms();
    /* Prompt ingestion: full steps for the state, lm_head only on the
     * last prompt token (it does not affect the state trajectory). */
    for (int i = 0; i < n; i++) {
        int rc = (i + 1 == n)
                     ? nemo_model_step(e->model, ids[i], logits)
                     : nemo_model_step_hidden(e->model, ids[i], hidden);
        if (rc != 0) {
            printf("ERROR forward pass failed during prompt\n");
            fflush(stdout);
            free(logits);
            free(hidden);
            return;
        }
    }
    splitter sp;
    memset(&sp, 0, sizeof sp);
    int generated = 0;
    double gen_ms = 0.0;
    int stopped = 0;
    if (n == 0) { /* empty prompt: nothing to condition on */
        printf("ERROR empty prompt\n");
        fflush(stdout);
        free(logits);
        free(hidden);
        return;
    }
    while (generated < e->max_new) {
        double t0 = now_ms();
        int next = argmax(logits);
        if (next == e->eos_id) break;
        generated++;
        char *piece = nemo_tokenizer_decode(e->tk, &next, 1, 1);
        if (piece) {
            split_feed(&sp, piece);
            free(piece);
        }
        /* Step the chosen token to produce the next logits, unless
         * the turn is done: an extra step would advance the state
         * past the last emitted token and corrupt the next turn. */
        if (generated < e->max_new &&
            nemo_model_step(e->model, next, logits) != 0) {
            printf("ERROR forward pass failed during generation\n");
            fflush(stdout);
            free(logits);
            free(hidden);
            return;
        }
        double ms = now_ms() - t0;
        gen_ms += ms;
        printf("TICK mode=%s layer=%d next=-1 active=1 staged=0 tok=%d "
               "ms_tok=%.1f gbs=%.3f\n",
               sp.thinking ? "thinking" : "answering",
               e->model->layer_limit - 1, generated, ms,
               ms > 0 ? (double)e->bytes_tok / (ms / 1000.0) / 1e9 : 0.0);
        fflush(stdout);
        if (poll_command(e)) {
            if (e->pending_cmd &&
                strncmp(e->pending_cmd, "STOP", 4) == 0) {
                free(e->pending_cmd);
                e->pending_cmd = NULL;
                stopped = 1;
                break;
            }
            break; /* another command: end the turn, main loop reads it */
        }
    }
    (void)stopped;
    split_flush(&sp, 1);
    double total_ms = now_ms() - turn_t0;
    printf("STATS tokens=%d ms=%.0f tps=%.3f ms_per_tok=%.1f "
           "gbs_avg=%.3f peak_mb=%.1f\n",
           generated, total_ms,
           gen_ms > 0 ? generated / (gen_ms / 1000.0) : 0.0,
           generated ? gen_ms / generated : 0.0,
           gen_ms > 0
               ? (double)e->bytes_tok * generated / (gen_ms / 1000.0) / 1e9
               : 0.0,
           peak_kb() / 1024.0);
    printf("DONE\n");
    fflush(stdout);
    free(logits);
    free(hidden);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: nemo-engine <checkpoint_dir> <tokenizer.json> "
                "[--layers N] [--max-pos N] [--max-new N]\n");
        return 2;
    }
    int layers = 0, max_pos = 4096, max_new = 256;
    for (int i = 3; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--layers")) layers = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--max-pos")) max_pos = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--max-new")) max_new = atoi(argv[i + 1]);
        else {
            fprintf(stderr, "nemo-engine: unknown option %s\n", argv[i]);
            return 2;
        }
    }
    nemo_checkpoint cp;
    if (nemo_checkpoint_open(&cp, argv[1]) != 0) {
        fprintf(stderr, "nemo-engine: cannot open checkpoint %s\n", argv[1]);
        return 1;
    }
    nemo_tokenizer *tk = nemo_tokenizer_load(argv[2]);
    if (!tk) {
        fprintf(stderr, "nemo-engine: cannot load tokenizer %s\n", argv[2]);
        return 1;
    }
    nemo_model model;
    if (nemo_model_init(&model, &cp, layers, max_pos) != 0) {
        fprintf(stderr, "nemo-engine: model init failed\n");
        return 1;
    }
    model.on_layer = on_layer;
    engine e;
    memset(&e, 0, sizeof e);
    e.cp = &cp;
    e.tk = tk;
    e.model = &model;
    e.max_new = max_new;
    e.eos_id = nemo_tokenizer_special(tk, "<|end_of_text|>");
    if (e.eos_id < 0) e.eos_id = 12; /* arch study: EOS 12 */
    e.bytes_tok = bytes_per_token(&cp, model.layer_limit);
    printf("READY nemo-engine 2 model=Nemotron-Nano-9B-v2 weights=raw "
           "layers=%d\n",
           model.layer_limit);
    fflush(stdout);

    char *line = NULL;
    size_t cap = 0;
    for (;;) {
        if (e.pending_cmd) {
            line = e.pending_cmd;
            e.pending_cmd = NULL;
        } else if (getline(&line, &cap, stdin) < 0) {
            break;
        }
        if (!strncmp(line, "PROMPT ", 7)) {
            char *text = line + 7;
            size_t L = strlen(text);
            while (L && (text[L - 1] == '\n' || text[L - 1] == '\r'))
                text[--L] = 0;
            unescape(text);
            run_turn(&e, text);
        } else if (!strncmp(line, "RESET", 5)) {
            nemo_model_reset(&model);
        } else if (!strncmp(line, "STOP", 4)) {
            /* no generation running: nothing to stop */
        } else if (!strncmp(line, "QUIT", 4)) {
            free(line);
            line = NULL;
            break;
        }
        free(line);
        line = NULL;
        cap = 0;
    }
    free(line);
    nemo_model_free(&model);
    nemo_tokenizer_free(tk);
    nemo_checkpoint_close(&cp);
    return 0;
}
