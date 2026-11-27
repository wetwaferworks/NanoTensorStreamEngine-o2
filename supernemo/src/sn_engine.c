/* sn_engine.c - the SuperNemo engine: NTSE chat protocol v3 over sn_model (pure C, zero mutexes).
 *
 *   sn-engine --pack PREFIX [--shapes TSV] --config config.json --tokenizer tokenizer.json
 *             [--threads N] [--max-pos N] [--max-new N] [--think] [--below-normal]
 *             [--ask "text"]      one-shot: print the reply and the measured speed, no protocol
 *
 * Speaks the same line protocol as nemo-engine (littlenemo/PROTOCOL.md): PROMPT, STOP, RESET, QUIT in;
 * READY, HW, PREFILL, LAYER, MOE, THINK/TOK, TICK, STATS, MEM, DONE out. MEDIA lines are refused with a
 * plain reason when the pack carries no vision tensors; images go through the RADIO tower (sn_vision) at PROMPT time. Greedy decoding, the same discipline as Little Nemo.
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
#include "sn_vision.h"
#include "sn_image.h"
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
#define MAX_MEDIA 8
typedef struct {
    int id, wp, hp, tokens;
    unsigned char *rgb; int w, h;
} media_t;

typedef struct {
    sn_model *m;
    sn_vision *vis;
    int max_patches, img_id;
    media_t media[MAX_MEDIA]; int n_media;
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

static void media_clear(engine *e)
{
    for (int i = 0; i < e->n_media; i++) free(e->media[i].rgb);
    e->n_media = 0;
}

static void engine_reset(engine *e) { sn_model_reset(e->m); e->turns = 0; e->pos = 0; media_clear(e); }

/* percent-decode in place (RFC 3986, UTF-8 bytes) */
static void pct_decode(char *s)
{
    char *o = s;
    while (*s) {
        if (*s == '%' && s[1] && s[2]) {
            char h[3] = { s[1], s[2], 0 }; char *end;
            long v = strtol(h, &end, 16);
            if (*end == 0) { *o++ = (char)v; s += 3; continue; }
        }
        *o++ = *s++;
    }
    *o = 0;
}
static void pct_encode(const char *in, char *out, size_t cap)
{
    size_t n = 0;
    for (; *in && n + 4 < cap; in++) {
        unsigned char c = (unsigned char)*in;
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' || c == '.' || c == '_' || c == '~') out[n++] = (char)c;
        else n += (size_t)sprintf(out + n, "%%%02X", c);
    }
    out[n] = 0;
}
static void media_err(int id, const char *reason)
{
    char q[512]; pct_encode(reason, q, sizeof q);
    printf("MEDIA_ERR id=%d reason=%s\n", id, q); fflush(stdout);
}

/* MEDIA id=<n> type=<image|video> mime=<m> bytes=<n> [w= h=] name=<pct> path=<pct> */
static void handle_media(engine *e, char *line)
{
    int id = 0; long bytes = -1; char type[16] = "", path[1100] = "";
    for (char *tok = strtok(line + 5, " "); tok; tok = strtok(NULL, " ")) {
        if (!strncmp(tok, "id=", 3)) id = atoi(tok + 3);
        else if (!strncmp(tok, "type=", 5)) { strncpy(type, tok + 5, sizeof type - 1); type[sizeof type - 1] = 0; }
        else if (!strncmp(tok, "bytes=", 6)) bytes = atol(tok + 6);
        else if (!strncmp(tok, "path=", 5)) { strncpy(path, tok + 5, sizeof path - 1); path[sizeof path - 1] = 0; pct_decode(path); }
    }
    if (!e->vis) { media_err(id, "this SuperNemo pack has no vision tensors; text only"); return; }
    if (strcmp(type, "image")) { media_err(id, "this engine accepts images only; video is not supported yet"); return; }
    if (e->n_media >= MAX_MEDIA) { media_err(id, "too many attachments for one turn (limit 8)"); return; }
    for (int i = 0; i < e->n_media; i++) if (e->media[i].id == id) { media_err(id, "duplicate attachment id"); return; }
    FILE *f = path[0] ? fopen(path, "rb") : NULL;
    if (!f) { media_err(id, "the file does not exist on the engine's machine (it needs a shared path)"); return; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fclose(f);
    if (bytes >= 0 && sz != bytes) { media_err(id, "the file size does not match bytes="); return; }
    media_t *mt = &e->media[e->n_media];
    memset(mt, 0, sizeof *mt);
    char why[256];
    if (sn_image_load(path, &mt->rgb, &mt->w, &mt->h, why, sizeof why) != 0) { media_err(id, why); return; }
    const sn_vgeom *vg = sn_vision_geom(e->vis);
    /* the checkpoint's own rule first (budget 13312 patches, at least 1024); only when the image itself is bigger than
     * this machine should chew on (--max-patches) is it planned again with the smaller budget (small images keep the
     * reference's scale-up to about 1024 patches) */
    sn_image_plan(mt->w, mt->h, vg->patch, vg->shuffle, 1024, 13312, &mt->wp, &mt->hp);
    long natural = (long)(mt->w / vg->patch + 1) * (long)(mt->h / vg->patch + 1);   /* about the image's own patch count */
    if (mt->wp * mt->hp > e->max_patches && natural > e->max_patches)
        sn_image_plan(mt->w, mt->h, vg->patch, vg->shuffle, 1024, e->max_patches > 64 ? e->max_patches : 64, &mt->wp, &mt->hp);
    mt->id = id; mt->tokens = mt->wp * mt->hp / (vg->shuffle * vg->shuffle);
    e->n_media++;
    printf("MEDIA_OK id=%d type=image tokens=%d\n", id, mt->tokens); fflush(stdout);
}

static int strip_tag(char *s, const char *tag)
{
    size_t tl = strlen(tag); int found = 0; char *p;
    while ((p = strstr(s, tag)) != NULL) { memmove(p, p + tl, strlen(p + tl) + 1); found = 1; }
    return found;
}

/* Feed an already-templated text through the model and generate. */
static int run_templated(engine *e, const char *text, int think_on, const float *embeds, int n_embeds)
{
    int V = sn_model_geom(e->m)->vocab;
    int cap_ids = (int)strlen(text) + 16;
    int *ids = (int *)malloc(sizeof(int) * (size_t)cap_ids);
    if (!ids) return -1;
    int n = nemo_tokenizer_encode(e->tk, text, ids, cap_ids);
    if (n <= 0) { free(ids); printf("ERROR prompt could not be tokenized\n"); fflush(stdout); return -1; }
    if (e->pos + n + 8 > e->max_pos) {
        free(ids); engine_reset(e);
        printf("ERROR the context is full (limit %d tokens); the conversation was reset, send your message again\n", e->max_pos);
        fflush(stdout); return -1;
    }
    double t0 = now_ms();
    /* batched, layer-major prefill: each weight matrix and each routed expert is read once per 128-token chunk */
    int *erow = (int *)malloc(sizeof(int) * (size_t)n);
    if (!erow) { free(ids); return -1; }
    int ei = 0;
    for (int i = 0; i < n; i++) erow[i] = (n_embeds > 0 && ids[i] == e->img_id && ei < n_embeds) ? ei++ : -1;
    int prc = sn_model_prefill(e->m, ids, erow, embeds, n, e->logits);
    free(erow);
    if (prc != 0) { free(ids); engine_reset(e); printf("ERROR forward pass failed during prompt; state reset\n"); fflush(stdout); return -1; }
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
    printf("STATS tokens=%d ms=%.0f tps=%.4f ms_per_tok=%.1f gbs_avg=0 peak_mb=%.1f prompt_tokens=%d prompt_ms=%.0f prompt_tps=%.3f stopped=%d lane=%s gen_tps=%.4f ttft_ms=%.0f\n",
           generated, total_ms, gen_ms > 0 ? generated / (gen_ms / 1000.0) : 0.0, generated ? gen_ms / generated : 0.0,
           peak_kb() / 1024.0, n, prefill_ms, prefill_ms > 0 ? n / (prefill_ms / 1000.0) : 0.0, stopped, sn_model_accel(e->m),
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
    if (L == 0 && e->n_media == 0) { free(u); free(tpl); printf("ERROR empty prompt\n"); fflush(stdout); return; }
    /* Images first, then the text (chat_template.jinja render_content). Each <image> placeholder becomes
     * <img> + one <image> token per projected patch group + </img>, as the checkpoint's processor does. */
    const sn_vgeom *vg = e->vis ? sn_vision_geom(e->vis) : NULL;
    int Hh = sn_model_geom(e->m)->hidden;
    size_t need = ul + 1024, total_tok = 0;
    for (int k = 0; k < e->n_media; k++) { need += 64 + (size_t)e->media[k].tokens * 8; total_tok += (size_t)e->media[k].tokens; }
    free(tpl); tpl = (char *)malloc(need);
    float *embeds = NULL;
    if (!tpl) { free(u); media_clear(e); printf("ERROR out of memory\n"); fflush(stdout); return; }
    if (e->n_media > 0) {
        embeds = (float *)malloc(total_tok * (size_t)Hh * sizeof(float));
        if (!embeds) { free(u); free(tpl); media_clear(e); printf("ERROR out of memory for image embeddings\n"); fflush(stdout); return; }
        size_t at = 0;
        for (int k = 0; k < e->n_media; k++) {
            media_t *mt = &e->media[k];
            const float mean[3] = { 0.48145466f, 0.4578275f, 0.40821073f }, sd[3] = { 0.26862954f, 0.26130258f, 0.27577711f };
            double t0 = now_ms();
            float *pix = sn_image_prepare(mt->rgb, mt->w, mt->h, mt->wp, mt->hp, vg->patch, mean, sd);
            int rc = pix ? sn_vision_embed(e->vis, pix, mt->hp * vg->patch, mt->wp * vg->patch, embeds + at * (size_t)Hh) : -1;
            free(pix);
            if (rc != 0) { free(u); free(tpl); free(embeds); media_clear(e); printf("ERROR the vision tower failed on an image\n"); fflush(stdout); return; }
            at += (size_t)mt->tokens;
            printf("EXPAND tensor=vision.%d ms=%.0f tokens=%d\n", mt->id, now_ms() - t0, mt->tokens);
            fflush(stdout);
        }
    }
    size_t w = 0;
    w += (size_t)sprintf(tpl + w, "%s<|im_start|>user\n", e->turns == 0 ? "<|im_start|>system\n<|im_end|>\n" : "\n");
    for (int k = 0; k < e->n_media; k++) {
        if (e->n_media > 1) w += (size_t)sprintf(tpl + w, "<image %d>", k + 1);
        w += (size_t)sprintf(tpl + w, "<img>");
        for (int t = 0; t < e->media[k].tokens; t++) { memcpy(tpl + w, "<image>", 7); w += 7; }
        w += (size_t)sprintf(tpl + w, "</img>");
    }
    w += (size_t)sprintf(tpl + w, "%s<|im_end|>\n<|im_start|>assistant\n%s", b, think_on ? "<think>\n" : "<think></think>");
    run_templated(e, tpl, think_on, embeds, (int)total_tok);
    free(embeds);
    media_clear(e);
    free(u); free(tpl);
}

int main(int argc, char **argv)
{
    const char *pack = NULL, *shapes = NULL, *config = NULL, *tokf = NULL, *ask = NULL;
    int threads = 0, max_pos = 4096, max_new = 256, think = 0, below = 0, max_patches = 1024, no_vision = 0, gpu_mb = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--pack") && i + 1 < argc) pack = argv[++i];
        else if (!strcmp(argv[i], "--shapes") && i + 1 < argc) shapes = argv[++i];
        else if (!strcmp(argv[i], "--config") && i + 1 < argc) config = argv[++i];
        else if (!strcmp(argv[i], "--tokenizer") && i + 1 < argc) tokf = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-pos") && i + 1 < argc) max_pos = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-new") && i + 1 < argc) max_new = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ask") && i + 1 < argc) ask = argv[++i];
        else if (!strcmp(argv[i], "--max-patches") && i + 1 < argc) max_patches = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-vision")) no_vision = 1;
        else if (!strcmp(argv[i], "--gpu-mb") && i + 1 < argc) gpu_mb = atoi(argv[++i]);
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
    if (gpu_mb > 0) {
        char gwhy[400] = "";
        int nres = sn_model_cuda(e.m, (long long)gpu_mb << 20, gwhy, sizeof gwhy);
        if (nres < 0) fprintf(stderr, "sn-engine: CUDA lane off: %s\n", gwhy);
        else fprintf(stderr, "sn-engine: CUDA lane on %s: %d tensors, %.0f MB resident\n", sn_model_gpu_name(e.m), nres, sn_model_gpu_bytes(e.m) / 1048576.0);
    }
    e.max_patches = max_patches;
    e.img_id = nemo_tokenizer_special(e.tk, "<image>");
    if (!no_vision && !ask) {
        sn_vgeom vg;
        if (e.img_id >= 0 && sn_vgeom_load(&vg, config, g->hidden) == 0) {
            e.vis = sn_vision_open(sn_model_pack(e.m), sn_model_pool(e.m), sn_model_threads(e.m), &vg);
            if (!e.vis) fprintf(stderr, "sn-engine: vision tower not available; text only\n");
        }
    }
    int cores = nemo_cores();
    if (!e.quiet) sn_model_set_hooks(e.m, on_layer_cb, on_moe_cb, NULL);
    printf("READY nemo-engine 3 model=SuperNemo weights=packed layers=%d cores=%d threads=%d mode=custom simd=0 accel=%s chat=1 think=%d moe=%d%s\n",
           g->n_layers, cores, sn_model_threads(e.m), sn_model_accel(e.m), think, g->moe_experts, e.vis ? " media=image" : "");
    printf("HW cores=%d threads=%d accel=%s\n", cores, sn_model_threads(e.m), sn_model_accel(e.m));
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
        else if (!strncmp(line, "MEDIA", 5)) handle_media(&e, line);
        else if (!strncmp(line, "FRAME", 5)) { /* ignored */ }
        else if (!strncmp(line, "PROMPT ", 7)) { unescape(line + 7); run_turn(&e, line + 7); }
        else if (L > 0) { printf("ERROR unknown command\n"); fflush(stdout); }
        if (line != e.pending_cmd && !e.pending_cmd) { /* keep buffer for reuse */ }
    }
    free(line);
    media_clear(&e);
    sn_vision_close(e.vis);
    sn_model_close(e.m);
    return 0;
}
