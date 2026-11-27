/* sn_engine.c - the SuperNemo engine: NTSE chat protocol v3 over sn_model (pure C, zero mutexes).
 *
 *   sn-engine --pack PREFIX [--shapes TSV] --config config.json --tokenizer tokenizer.json
 *             [--threads N] [--max-pos N] [--max-new N] [--think] [--below-normal]
 *             [--ask "text"]      one-shot: print the reply and the measured speed, no protocol
 *
 * Speaks the same line protocol as nemo-engine (littlenemo/PROTOCOL.md): PROMPT, STOP, RESET, QUIT in;
 * READY, HW, PREFILL, LAYER, MOE, THINK/TOK, TICK, STATS, MEM, DONE out. MEDIA lines are refused with a
 * plain reason until the RADIO vision tower lands. Greedy decoding, the same discipline as Little Nemo.
 * The chat template is the checkpoint's own (chat_template.jinja): ChatML with <think> handling.
 *
 * Direction by JD. Co-implementation by Cortana, built with Grok Bot by SpaceX. */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#include "sn_model.h"
#include "../../littlenemo/src/pool.h"
#include "../../littlenemo/src/tokenizer.h"

#if defined(_WIN32)
#include <conio.h>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
static long eng_getline(char **lineptr, size_t *n, FILE *stream) {
    if (!lineptr || !n) return -1;
    if (!*lineptr || *n == 0) { *n = 128; *lineptr = (char *)realloc(*lineptr, *n); if (!*lineptr) return -1; }
    size_t len = 0; int c;
    while ((c = fgetc(stream)) != EOF) {
        if (len + 2 > *n) { *n *= 2; char *nb = (char *)realloc(*lineptr, *n); if (!nb) return -1; *lineptr = nb; }
        (*lineptr)[len++] = (char)c;
        if (c == '\n') break;
    }
    if (len == 0) return -1;
    (*lineptr)[len] = '\0';
    return (long)len;
}
#define getline eng_getline
#else
#include <poll.h>
#endif

static double now_ms(void)
{
#if defined(_WIN32)
    static LARGE_INTEGER freq; LARGE_INTEGER t;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart * 1000.0 / (double)freq.QuadPart;
#else
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
#endif
}

static long peak_kb(void)
{
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc)) return (long)(pmc.PeakWorkingSetSize / 1024);
    return 0;
#else
    FILE *f = fopen("/proc/self/status", "r"); if (!f) return 0;
    char line[256]; long kb = 0;
    while (fgets(line, sizeof line, f)) if (sscanf(line, "VmHWM: %ld kB", &kb) == 1) break;
    fclose(f); return kb;
#endif
}

#include "sn_engine_text.inc"

/* ---- engine state ---- */
typedef struct {
    sn_model *m;
    nemo_tokenizer *tk;
    int max_new, max_pos, think_default, turns, pos;
    int eos_a, eos_b;
    int quiet;          /* --ask: no protocol lines */
    char *pending_cmd;
    float *logits;
} engine;

static void on_layer_cb(void *u, int layer, char t)
{
    (void)u;
    printf("LAYER %d %s\n", layer, t == 'M' ? "mamba" : t == 'A' ? "attention" : "moe");
}

static void on_moe_cb(void *u, int layer, const int *ex, int k)
{
    (void)u;
    printf("MOE layer=%d experts=", layer);
    for (int i = 0; i < k; i++) printf("%s%d", i ? "%2C" : "", ex[i]);
    printf(" k=%d of=512\n", k);
}

static int poll_command(engine *e)
{
#if defined(_WIN32)
    HANDLE h = GetStdHandle(STD_INPUT_HANDLE); DWORD avail = 0;
    DWORD type = (h == INVALID_HANDLE_VALUE) ? 0 : GetFileType(h);
    if (type == FILE_TYPE_PIPE) { if (!PeekNamedPipe(h, NULL, 0, NULL, &avail, NULL) || avail == 0) return 0; }
    else if (type == FILE_TYPE_CHAR) { if (!_kbhit()) return 0; }
    else return 0;
#else
    struct pollfd pfd; pfd.fd = 0; pfd.events = POLLIN;
    if (poll(&pfd, 1, 0) <= 0 || !(pfd.revents & POLLIN)) return 0;
#endif
    char *line = NULL; size_t cap = 0;
    if (getline(&line, &cap, stdin) < 0) { free(line); e->pending_cmd = NULL; return 0; }
    e->pending_cmd = line;
    return 1;
}

static int argmax(const float *l, int n)
{
    int b = 0; float v = l[0];
    for (int i = 1; i < n; i++) if (l[i] > v) { v = l[i]; b = i; }
    return b;
}

static void engine_reset(engine *e) { sn_model_reset(e->m); e->turns = 0; e->pos = 0; }

static int strip_tag(char *s, const char *tag)
{
    size_t tl = strlen(tag); int found = 0; char *p;
    while ((p = strstr(s, tag)) != NULL) { memmove(p, p + tl, strlen(p + tl) + 1); found = 1; }
    return found;
}

/* Feed an already-templated text through the model and generate. */
static int run_templated(engine *e, const char *text, int think_on)
{
    int V = sn_model_geom(e->m)->vocab;
    int *ids = (int *)malloc(sizeof(int) * 16384);
    if (!ids) return -1;
    int n = nemo_tokenizer_encode(e->tk, text, ids, 16384);
    if (n <= 0) { free(ids); printf("ERROR prompt could not be tokenized\n"); fflush(stdout); return -1; }
    if (e->pos + n + 8 > e->max_pos) {
        free(ids); engine_reset(e);
        printf("ERROR the context is full (limit %d tokens); the conversation was reset, send your message again\n", e->max_pos);
        fflush(stdout); return -1;
    }
    double t0 = now_ms();
    for (int i = 0; i < n; i++) {
        int rc = sn_model_step(e->m, ids[i], i == n - 1 ? e->logits : NULL);
        if (rc != 0) { free(ids); engine_reset(e); printf("ERROR forward pass failed during prompt; state reset\n"); fflush(stdout); return -1; }
    }
    double prefill_ms = now_ms() - t0;
    e->pos += n;
    if (!e->quiet) { printf("PREFILL tokens=%d ms=%.0f tps=%.3f\n", n, prefill_ms, prefill_ms > 0 ? n / (prefill_ms / 1000.0) : 0.0); fflush(stdout); }
    free(ids);
    splitter sp; memset(&sp, 0, sizeof sp);
    sp.thinking = think_on ? 1 : 0;
    int generated = 0, stopped = 0;
    double gen_ms = 0.0;
    while (generated < e->max_new) {
        int next = argmax(e->logits, V);
        if (next == e->eos_a || next == e->eos_b) {
            /* commit the end-of-message token so the state holds the whole reply */
            sn_model_step(e->m, next, NULL); e->pos++;
            break;
        }
        char *piece = nemo_tokenizer_decode(e->tk, &next, 1, 0);
        if (piece) { split_feed(&sp, piece); free(piece); }
        generated++;
        double t1 = now_ms();
        int rc = sn_model_step(e->m, next, e->logits);
        double ms = now_ms() - t1;
        gen_ms += ms; e->pos++;
        if (rc != 0) { engine_reset(e); printf("ERROR forward pass failed during generation; state reset\n"); fflush(stdout); return -1; }
        if (!e->quiet) printf("TICK mode=%s layer=%d next=-1 active=1 staged=0 tok=%d ms_tok=%.1f gbs=0\n", sp.thinking ? "thinking" : "answering", sn_model_geom(e->m)->n_layers - 1, generated, ms);
        fflush(stdout);
        if (e->pos + 2 >= e->max_pos) break;
        if (!e->quiet && poll_command(e)) {
            if (e->pending_cmd && strncmp(e->pending_cmd, "STOP", 4) == 0) { free(e->pending_cmd); e->pending_cmd = NULL; stopped = 1; break; }
            break;
        }
    }
    split_flush(&sp, 1);
    double total_ms = now_ms() - t0;
    printf("STATS tokens=%d ms=%.0f tps=%.4f ms_per_tok=%.1f gbs_avg=0 peak_mb=%.1f prompt_tokens=%d prompt_ms=%.0f prompt_tps=%.3f stopped=%d lane=cpu gen_tps=%.4f ttft_ms=%.0f\n",
           generated, total_ms, gen_ms > 0 ? generated / (gen_ms / 1000.0) : 0.0, generated ? gen_ms / generated : 0.0,
           peak_kb() / 1024.0, n, prefill_ms, prefill_ms > 0 ? n / (prefill_ms / 1000.0) : 0.0, stopped,
           gen_ms > 0 ? generated / (gen_ms / 1000.0) : 0.0, prefill_ms);
    printf("MEM peak_mb=%.1f\n", peak_kb() / 1024.0);
    e->turns++;
    printf("DONE\n");
    fflush(stdout);
    return 0;
}

static void run_turn(engine *e, const char *user_text)
{
    size_t ul = strlen(user_text);
    char *u = (char *)malloc(ul + 1), *tpl = (char *)malloc(ul + 512);
    if (!u || !tpl) { free(u); free(tpl); printf("ERROR out of memory\n"); fflush(stdout); return; }
    memcpy(u, user_text, ul + 1);
    int think_on = e->think_default;
    if (strip_tag(u, "/no_think")) think_on = 0; else if (strip_tag(u, "/think")) think_on = 1;
    char *b = u; while (*b == ' ' || *b == '\t' || *b == '\n') b++;
    size_t L = strlen(b); while (L && (b[L - 1] == ' ' || b[L - 1] == '\t' || b[L - 1] == '\n')) b[--L] = 0;
    if (L == 0) { free(u); free(tpl); printf("ERROR empty prompt\n"); fflush(stdout); return; }
    /* chat_template.jinja: an (empty) system block first, then user, then the generation prompt.
     * Later turns open with the "\n" that follows the previous <|im_end|>; the reply itself is already in the state. */
    snprintf(tpl, ul + 512, "%s<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n%s",
             e->turns == 0 ? "<|im_start|>system\n<|im_end|>\n" : "\n", b, think_on ? "<think>\n" : "<think></think>");
    run_templated(e, tpl, think_on);
    free(u); free(tpl);
}

int main(int argc, char **argv)
{
    const char *pack = NULL, *shapes = NULL, *config = NULL, *tokf = NULL, *ask = NULL;
    int threads = 0, max_pos = 4096, max_new = 256, think = 0, below = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--pack") && i + 1 < argc) pack = argv[++i];
        else if (!strcmp(argv[i], "--shapes") && i + 1 < argc) shapes = argv[++i];
        else if (!strcmp(argv[i], "--config") && i + 1 < argc) config = argv[++i];
        else if (!strcmp(argv[i], "--tokenizer") && i + 1 < argc) tokf = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-pos") && i + 1 < argc) max_pos = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-new") && i + 1 < argc) max_new = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ask") && i + 1 < argc) ask = argv[++i];
        else if (!strcmp(argv[i], "--think")) think = 1;
        else if (!strcmp(argv[i], "--below-normal")) below = 1;
        else if (!strcmp(argv[i], "--chat") || !strcmp(argv[i], "--serve-stdio")) { /* accepted for GUI parity */ }
        else { fprintf(stderr, "sn-engine: unknown argument %s\n", argv[i]); return 2; }
    }
    if (!pack || !config || !tokf) {
        fprintf(stderr, "usage: sn-engine --pack PREFIX [--shapes TSV] --config config.json --tokenizer tokenizer.json [--threads N] [--max-pos N] [--max-new N] [--think] [--below-normal] [--ask TEXT]\n");
        return 2;
    }
#if defined(_WIN32)
    if (below) SetPriorityClass(GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS);
#else
    (void)below;
#endif
    char shapes_buf[1200];
    if (!shapes) { snprintf(shapes_buf, sizeof shapes_buf, "%s.shapes.tsv", pack); shapes = shapes_buf; }
    engine e; memset(&e, 0, sizeof e);
    e.max_new = max_new; e.max_pos = max_pos; e.think_default = think; e.quiet = ask != NULL;
    e.tk = nemo_tokenizer_load(tokf);
    if (!e.tk) { fprintf(stderr, "sn-engine: cannot load tokenizer %s\n", tokf); return 1; }
    e.eos_a = nemo_tokenizer_special(e.tk, "<|im_end|>");
    e.eos_b = nemo_tokenizer_special(e.tk, "</s>");
    if (e.eos_b < 0) e.eos_b = e.eos_a;
    e.m = sn_model_open(pack, shapes, config, max_pos, threads);
    if (!e.m) return 1;
    const sn_geom *g = sn_model_geom(e.m);
    e.logits = (float *)malloc((size_t)g->vocab * 4);
    if (!e.logits) return 1;
    int cores = nemo_cores();
    if (!e.quiet) sn_model_set_hooks(e.m, on_layer_cb, on_moe_cb, NULL);
    printf("READY nemo-engine 3 model=SuperNemo weights=packed layers=%d cores=%d threads=%d mode=custom simd=0 accel=cpu chat=1 think=%d moe=%d\n",
           g->n_layers, cores, sn_model_threads(e.m), think, g->moe_experts);
    printf("HW cores=%d threads=%d accel=cpu\n", cores, sn_model_threads(e.m));
    fflush(stdout);
    if (ask) { run_turn(&e, ask); return 0; }
    char *line = NULL; size_t cap = 0;
    for (;;) {
        if (e.pending_cmd) { line = e.pending_cmd; e.pending_cmd = NULL; }
        else {
            if (getline(&line, &cap, stdin) < 0) break;
        }
        size_t L = strlen(line); while (L && (line[L - 1] == '\n' || line[L - 1] == '\r')) line[--L] = 0;
        if (!strncmp(line, "QUIT", 4)) break;
        else if (!strncmp(line, "RESET", 5)) { engine_reset(&e); printf("DONE\n"); fflush(stdout); }
        else if (!strncmp(line, "STOP", 4)) { /* nothing running */ }
        else if (!strncmp(line, "MEDIA", 5)) {
            int id = 0; const char *p = strstr(line, "id="); if (p) id = atoi(p + 3);
            printf("MEDIA_ERR id=%d reason=the%%20SuperNemo%%20vision%%20tower%%20is%%20not%%20built%%20yet%%3B%%20text%%20only\n", id); fflush(stdout);
        }
        else if (!strncmp(line, "FRAME", 5)) { /* ignored */ }
        else if (!strncmp(line, "PROMPT ", 7)) { unescape(line + 7); run_turn(&e, line + 7); }
        else if (L > 0) { printf("ERROR unknown command\n"); fflush(stdout); }
        if (line != e.pending_cmd && !e.pending_cmd) { /* keep buffer for reuse */ }
    }
    free(line);
    sn_model_close(e.m);
    return 0;
}
