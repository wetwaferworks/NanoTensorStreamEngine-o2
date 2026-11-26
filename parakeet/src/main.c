/* parakeet-c.exe - native Windows speech-to-text for NVIDIA Parakeet models.
 *   parakeet-c.exe [--model DIR] [--threads N] [--timing] file.wav [...]
 *   parakeet-c.exe --mic [--seconds N]      live dictation in the console
 *   parakeet-c.exe --stdio                  line protocol for the NTSE chat GUI
 *   parakeet-c.exe --mic-test [N]           mic level test (no model needed)
 * Single main thread does all I/O and mic polling; the math pool is started
 * and joined around each decode (phase boundaries). No mutexes. */
#include "pk.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <conio.h>
#include <io.h>
#include <fcntl.h>
static void sleep_ms(int ms) { Sleep(ms); }
#else
#include <unistd.h>
#include <poll.h>
static void sleep_ms(int ms) { usleep(ms * 1000); }
#endif

static int g_threads = 0;
static int default_threads(void) {
#ifdef _WIN32
    DWORD n = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
#endif
    int t = (int)n; if (t > 8) t = 8; if (t < 1) t = 1; return t;
}
static char *run_transcribe(const pk_model *m, const float *x, int n, pk_timing *tm) {
    pk_pool_start(g_threads);
    char *s = pk_transcribe(m, x, n, tm);
    pk_pool_stop();
    return s;
}

/* ------------------------------------------------------------ stdin polling (single thread) */
static char g_in[8192]; static int g_inn;
static int poll_line(char *out, int cap) {
    for (;;) {
        for (int i = 0; i < g_inn; i++) if (g_in[i] == '\n') {
            int l = i; if (l && g_in[l - 1] == '\r') l--;
            if (l >= cap) l = cap - 1;
            memcpy(out, g_in, l); out[l] = 0;
            memmove(g_in, g_in + i + 1, g_inn - i - 1); g_inn -= i + 1;
            return 1;
        }
        if (g_inn >= (int)sizeof g_in - 1) g_inn = 0;
#ifdef _WIN32
        HANDLE h = GetStdHandle(STD_INPUT_HANDLE); DWORD avail = 0, got = 0;
        DWORD type = GetFileType(h);
        if (type == FILE_TYPE_PIPE) {
            if (!PeekNamedPipe(h, NULL, 0, NULL, &avail, NULL)) return -1; /* pipe closed */
            if (!avail) return 0;
            DWORD want = (DWORD)(sizeof g_in - 1 - g_inn); if (avail < want) want = avail;
            if (!ReadFile(h, g_in + g_inn, want, &got, NULL) || !got) return -1;
            g_inn += got;
        } else if (type == FILE_TYPE_CHAR) {
            if (!_kbhit()) return 0;
            if (!fgets(g_in + g_inn, sizeof g_in - g_inn, stdin)) return -1;
            g_inn += (int)strlen(g_in + g_inn);
        } else {
            if (!fgets(g_in + g_inn, sizeof g_in - g_inn, stdin)) return -1;
            g_inn += (int)strlen(g_in + g_inn);
        }
#else
        struct pollfd p = { 0, POLLIN, 0 };
        if (poll(&p, 1, 0) <= 0) return 0;
        ssize_t r = read(0, g_in + g_inn, sizeof g_in - 1 - g_inn);
        if (r <= 0) return -1;
        g_inn += (int)r;
#endif
    }
}

/* ------------------------------------------------------------ streaming dictation
 * Rolling segment: partials re-decode the open segment every hop; a segment
 * is finalized at a pause (>= pause_ms silence after speech) or when it gets
 * long (cut at the quietest 100 ms of its last few seconds). */
typedef void (*emit_fn)(const char *kind, int segid, const char *text, void *u);
typedef struct {
    const pk_model *m;
    float *seg; int n, cap;
    int last_dec_n, speech, last_speech_end, speech_frames;
    double noise_db, fe; int fn;
    int segid; char *partial;
    float level_db; int speaking;
    int hop, pause, maxlen, preroll, tail;
    emit_fn emit; void *u;
    pk_timing last;
    double last_dec_s;
} dict;

static void dict_init(dict *d, const pk_model *m, emit_fn e, void *u) {
    memset(d, 0, sizeof *d);
    d->m = m; d->emit = e; d->u = u;
    d->cap = 60 * PK_SR; d->seg = malloc(sizeof(float) * d->cap);
    d->noise_db = -60; d->level_db = -90;
    d->hop = PK_SR * 4 / 10;      /* partial every 0.4 s of new audio (adaptive, see dict_step) */
    d->pause = PK_SR * 7 / 10;    /* 0.7 s pause finalizes */
    d->maxlen = PK_SR * 12;       /* hard cap per segment (keeps partials < ~1 s) */
    d->preroll = PK_SR * 4 / 10;  /* keep 0.4 s before speech */
    d->tail = PK_SR * 3 / 10;     /* keep 0.3 s after speech */
    const char *e1 = getenv("PARAKEET_HOP_MS"); if (e1) d->hop = atoi(e1) * PK_SR / 1000;
    const char *e2 = getenv("PARAKEET_PAUSE_MS"); if (e2) d->pause = atoi(e2) * PK_SR / 1000;
}
static void dict_reset(dict *d) { d->speech_frames = 0; d->n = 0; d->last_dec_n = 0; d->speech = 0; d->last_speech_end = 0; free(d->partial); d->partial = NULL; d->fe = 0; d->fn = 0; }

static int is_filler(const char *s) {
    static const char *f[] = { "uh", "um", "uhm", "mm", "mhm", "hmm", "hm", "ah", "eh", "er", NULL };
    char w[16]; int n = 0;
    for (const char *c = s; *c; c++) { if ((*c >= 'a' && *c <= 'z') || (*c >= 'A' && *c <= 'Z')) { if (n < 15) w[n++] = (char)(*c | 32); } else if (*c != ' ' && *c != '.' && *c != ',' && *c != '?' && *c != '!') return 0; else if (*c == ' ' && n) return 0; }
    w[n] = 0; if (!n) return 0;
    for (int i = 0; f[i]; i++) if (!strcmp(w, f[i])) return 1;
    return 0;
}
static void dict_decode(dict *d, int end, int final) {
    pk_timing tm;
    char *s = run_transcribe(d->m, d->seg, end, &tm);
    d->last = tm; d->last_dec_s = tm.total_s;
    if (final) {
        if (d->speech_frames < 10 || is_filler(s)) s[0] = 0;   /* < 0.2 s of speech or a lone "uh": drop */
        d->speech_frames = 0;
        d->emit("FINAL", d->segid, s, d->u);
        d->segid++;
        free(s); free(d->partial); d->partial = NULL;
    } else {
        if (!d->partial || strcmp(d->partial, s)) d->emit("PARTIAL", d->segid, s, d->u);
        free(d->partial); d->partial = s;
    }
    char st[200];
    snprintf(st, sizeof st, "audio=%.2f mel=%.3f enc=%.3f dec=%.3f total=%.3f rtf=%.3f final=%d",
             tm.audio_s, tm.mel_s, tm.enc_s, tm.dec_s, tm.total_s, tm.audio_s > 0 ? tm.total_s / tm.audio_s : 0, final);
    d->emit("STAT", d->segid, st, d->u);
}
static void dict_finalize(dict *d, int end, int keep_speech) {
    if (end > d->n) end = d->n;
    if (end > PK_SR / 4) dict_decode(d, end, 1);
    else if (d->partial) { d->emit("FINAL", d->segid, "", d->u); d->segid++; free(d->partial); d->partial = NULL; }
    memmove(d->seg, d->seg + end, sizeof(float) * (d->n - end));
    d->n -= end; d->last_dec_n = 0;
    d->speech = keep_speech; d->last_speech_end = keep_speech ? d->n : 0;
}

static void dict_push(dict *d, const float *x, int k) {
    for (int i = 0; i < k; i++) {
        if (d->n == d->cap) dict_finalize(d, d->n, 0);
        d->seg[d->n++] = x[i];
        d->fe += (double)x[i] * x[i];
        if (++d->fn == 320) { /* 20 ms frame */
            double db = 10 * log10(d->fe / 320 + 1e-12);
            d->fe = 0; d->fn = 0;
            if (db < d->noise_db) d->noise_db += (db - d->noise_db) * 0.3; else d->noise_db += (db - d->noise_db) * 0.005;
            if (d->noise_db < -80) d->noise_db = -80;
            double thr = d->noise_db + 10; if (thr < -50) thr = -50;
            d->level_db = (float)db;
            d->speaking = db > thr;
            if (d->speaking) { d->speech = 1; d->last_speech_end = d->n; d->speech_frames++; }
            else if (!d->speech && d->n > d->preroll) { /* trim leading silence */
                int drop = d->n - d->preroll;
                memmove(d->seg, d->seg + drop, sizeof(float) * d->preroll); d->n = d->preroll; d->last_dec_n = 0;
            }
        }
    }
}
/* call after pushing; may run one decode */
static void dict_step(dict *d) {
    if (!d->speech) return;
    if (d->n - d->last_speech_end >= d->pause && d->n >= PK_SR * 8 / 10) {
        int end = d->last_speech_end + d->tail; if (end > d->n) end = d->n;
        dict_finalize(d, end, 0);
        return;
    }
    if (d->n >= d->maxlen) {
        int win = PK_SR / 10, lo = d->n - 5 * PK_SR, hi = d->n - PK_SR, cut = d->n - 2 * PK_SR; double best = 1e30;
        for (int s = lo; s + win <= hi; s += win / 4) { double e = 0; for (int i = 0; i < win; i++) e += (double)d->seg[s + i] * d->seg[s + i]; if (e < best) { best = e; cut = s + win / 2; } }
        dict_finalize(d, cut, 1);
        return;
    }
    int hop = d->hop, adapt = (int)(d->last_dec_s * 1.25 * PK_SR); if (adapt > hop) hop = adapt; /* never queue up behind the CPU */
    if (d->n - d->last_dec_n >= hop && d->n >= PK_SR * 6 / 10) { d->last_dec_n = d->n; dict_decode(d, d->n, 0); }
}
static void dict_stop(dict *d) {
    if (d->speech) { int end = d->last_speech_end + d->tail; if (end > d->n) end = d->n; dict_finalize(d, end, 0); }
    else if (d->partial) dict_finalize(d, 0, 0);
    dict_reset(d);
}

/* ------------------------------------------------------------ output helpers */
static void out_line(const char *kind, const char *text) {
    fputs(kind, stdout);
    if (text) { fputc(' ', stdout); for (const char *c = text; *c; c++) fputc((*c == '\n' || *c == '\r') ? ' ' : *c, stdout); }
    fputc('\n', stdout); fflush(stdout);
}
static void emit_stdio(const char *kind, int segid, const char *text, void *u) {
    (void)u; char head[64]; snprintf(head, sizeof head, "%s %d", kind, segid); out_line(head, text);
}

static void warm_pages(const pk_model *m) {
    volatile unsigned s = 0; for (size_t i = 0; i < m->st.size; i += 4096) s += m->st.base[i]; (void)s;
}

static const char *find_model_dir(const char *arg, char *buf, int cap) {
    if (arg) return arg;
    const char *e = getenv("PARAKEET_MODEL"); if (e) return e;
    const char *c[] = { "models\\parakeet-tdt-0.6b-v2", "../models/parakeet-tdt-0.6b-v2", NULL };
    for (int i = 0; c[i]; i++) {
        snprintf(buf, cap, "%s%cmodel_weights.safetensors", c[i],
#ifdef _WIN32
                 '\\'
#else
                 '/'
#endif
        );
        FILE *f = fopen(buf, "rb"); if (f) { fclose(f); snprintf(buf, cap, "%s", c[i]); return buf; }
    }
    return c[0];
}

/* ------------------------------------------------------------ modes */
static int mode_stdio(pk_model *m) {
    dict d; dict_init(&d, m, emit_stdio, NULL);
    char err[256]; pk_mic *mic = NULL; int rec = 0;
    float *buf = malloc(sizeof(float) * PK_SR * 4);
    char line[4096]; double last_level = 0; int warned_zero = 0, nonzero = 0, zero_n = 0;
    pk_audio sim = { NULL, 0 }; int sim_pos = 0; double sim_t0 = 0;
    {   /* warm up: page in weights and run a short decode so the first click is fast */
        double t0 = pk_now(); warm_pages(m);
        float *z = calloc(PK_SR, sizeof(float)); pk_timing tm; free(run_transcribe(m, z, PK_SR, &tm)); free(z);
        char info[256]; snprintf(info, sizeof info, "model=%dL d%d %s vocab=%d threads=%d warm=%.2fs",
                                 m->n_layers, m->d_model, m->is_tdt == 1 ? "TDT" : m->is_tdt == 2 ? "CTC" : "RNNT", m->vocab, g_threads, pk_now() - t0);
        out_line("READY", info);
    }
    for (;;) {
        int r = poll_line(line, sizeof line);
        if (r < 0) break;
        if (r > 0) {
            if (!strcmp(line, "QUIT")) break;
            else if (!strcmp(line, "PING")) out_line("PONG", NULL);
            else if (!strcmp(line, "START")) {
                if (!mic && !(mic = pk_mic_open(err, sizeof err))) { out_line("ERR", err); continue; }
                dict_reset(&d);
                if (pk_mic_start(mic)) { out_line("ERR", "mic start failed"); continue; }
                rec = 1; warned_zero = 0; nonzero = 0; zero_n = 0; out_line("REC", pk_mic_describe(mic));
            } else if (!strcmp(line, "STOP") || !strcmp(line, "CANCEL")) {
                if (rec == 2) { rec = 0; if (line[0] == 'S') dict_stop(&d); else dict_reset(&d); }
                else if (rec) {
                    int k = pk_mic_read(mic, buf, PK_SR * 4); if (k > 0 && line[0] == 'S') dict_push(&d, buf, k);
                    pk_mic_stop(mic); rec = 0;
                    if (line[0] == 'S') dict_stop(&d); else dict_reset(&d);
                }
                out_line("DONE", NULL);
            } else if (!strncmp(line, "SIMFILE ", 8)) {   /* test hook: replay a WAV as if it were the mic */
                free(sim.x); sim.x = NULL;
                if (pk_wav_read(line + 8, &sim)) { sim.x = NULL; out_line("ERR", "cannot read wav"); continue; }
                dict_reset(&d); sim_pos = 0; sim_t0 = pk_now(); rec = 2; out_line("REC", "simulated mic (WAV replay)");
            } else if (!strncmp(line, "FILE ", 5)) {
                pk_audio a; if (pk_wav_read(line + 5, &a)) { out_line("ERR", "cannot read wav"); continue; }
                pk_timing tm; char *s = run_transcribe(m, a.x, a.n, &tm);
                out_line("RESULT", s); free(s); free(a.x);
                char st[160]; snprintf(st, sizeof st, "-1 audio=%.2f total=%.3f rtf=%.3f", tm.audio_s, tm.total_s, tm.total_s / (tm.audio_s + 1e-9)); out_line("STAT", st);
            } else out_line("ERR", "unknown command");
        }
        if (rec == 2) {   /* WAV replay at real-time pace, then silence until STOP */
            int due = (int)((pk_now() - sim_t0) * PK_SR);
            if (due > sim_pos) {
                int real_end = due < sim.n ? due : sim.n;
                if (real_end > sim_pos) { dict_push(&d, sim.x + sim_pos, real_end - sim_pos); sim_pos = real_end; }
                static const float z[320];
                while (due - sim_pos >= 320) { dict_push(&d, z, 320); sim_pos += 320; }
                dict_step(&d);
            }
            double now = pk_now();
            if (now - last_level > 0.08) { char lv[64]; float l = (d.level_db + 60) / 60; if (l < 0) l = 0; if (l > 1) l = 1; snprintf(lv, sizeof lv, "%.3f %d", l, d.speaking); out_line("LEVEL", lv); last_level = now; }
            sleep_ms(10);
        } else if (rec) {
            int k = pk_mic_read(mic, buf, PK_SR * 4);
            if (k > 0) {
                if (!warned_zero) { for (int i = 0; i < k; i++) if (buf[i] != 0.f) { nonzero = 1; break; } zero_n += k;
                    if (!nonzero && zero_n > 2 * PK_SR) { char w[400]; snprintf(w, sizeof w, "mic sends pure digital silence: %s. If you are talking, check that device (NVIDIA Broadcast source/mute) or set PARAKEET_MIC to another mic", pk_mic_describe(mic)); out_line("WARN", w); warned_zero = 1; } }
                dict_push(&d, buf, k); dict_step(&d);
            }
            double now = pk_now();
            if (now - last_level > 0.08) {
                char lv[64]; float l = (d.level_db + 60) / 60; if (l < 0) l = 0; if (l > 1) l = 1;
                snprintf(lv, sizeof lv, "%.3f %d", l, d.speaking); out_line("LEVEL", lv); last_level = now;
            }
            sleep_ms(10);
        } else sleep_ms(15);
    }
    if (mic) { if (rec) pk_mic_stop(mic); pk_mic_close(mic); }
    free(buf);
    return 0;
}

static double g_t0;
static void emit_timed(const char *kind, int segid, const char *text, void *u) {
    (void)u; printf("%7.2fs %s %d %s\n", pk_now() - g_t0, kind, segid, text); fflush(stdout);
}
/* feed a WAV through the live dictation path at real-time pace (mic stand-in) */
static int mode_stream_test(pk_model *m, const char *path) {
    pk_audio a; if (pk_wav_read(path, &a)) { fprintf(stderr, "cannot read %s\n", path); return 2; }
    warm_pages(m);
    dict d; dict_init(&d, m, emit_timed, NULL);
    int chunk = PK_SR / 50, pos = 0; g_t0 = pk_now();
    while (pos < a.n) {
        double due = (double)pos / PK_SR;            /* audio time already delivered */
        while (pk_now() - g_t0 < due) sleep_ms(2);
        int k = a.n - pos < chunk ? a.n - pos : chunk;
        dict_push(&d, a.x + pos, k); pos += k; dict_step(&d);
    }
    printf("%7.2fs END-OF-AUDIO (%.2fs)\n", pk_now() - g_t0, a.n / (double)PK_SR);
    dict_stop(&d);
    printf("%7.2fs DONE\n", pk_now() - g_t0);
    free(a.x); return 0;
}

static void emit_console(const char *kind, int segid, const char *text, void *u) {
    (void)segid; int *timing = u;
    if (!strcmp(kind, "PARTIAL")) { fprintf(stderr, "\r\x1b[2K  ... %s", text); fflush(stderr); }
    else if (!strcmp(kind, "FINAL")) { fprintf(stderr, "\r\x1b[2K"); if (*text) { printf("%s\n", text); fflush(stdout); } }
    else if (!strcmp(kind, "STAT") && *timing) fprintf(stderr, "\r\x1b[2K  [%s]\n", text);
}
static int mode_mic(pk_model *m, double seconds, int timing) {
    char err[256]; pk_mic *mic = pk_mic_open(err, sizeof err);
    if (!mic) { fprintf(stderr, "mic: %s\n", err); return 2; }
    fprintf(stderr, "mic: %s\nSpeak. Press Enter to stop.\n", pk_mic_describe(mic));
    warm_pages(m);
    dict d; dict_init(&d, m, emit_console, &timing);
    float *buf = malloc(sizeof(float) * PK_SR * 4);
    pk_mic_start(mic);
    double t0 = pk_now(); char line[256];
    for (;;) {
        int k = pk_mic_read(mic, buf, PK_SR * 4);
        if (k > 0) { dict_push(&d, buf, k); dict_step(&d); }
        if (seconds > 0 && pk_now() - t0 >= seconds) break;
        if (poll_line(line, sizeof line) != 0) break;
        sleep_ms(10);
    }
    pk_mic_stop(mic); dict_stop(&d); pk_mic_close(mic); free(buf);
    return 0;
}
static int g_save;
static int mode_mic_test(double seconds) {
    char err[256]; pk_mic *mic = pk_mic_open(err, sizeof err);
    if (!mic) { fprintf(stderr, "mic: %s\n", err); return 2; }
    printf("mic: %s\n", pk_mic_describe(mic));
    int cap = (int)(seconds * PK_SR) + PK_SR, n = 0; float *all = malloc(sizeof(float) * cap);
    pk_mic_start(mic); double t0 = pk_now(), next = 0.1; int from = 0; double peak_all = 0;
    while (pk_now() - t0 < seconds) {
        int k = pk_mic_read(mic, all + n, cap - n); n += k;
        if (pk_now() - t0 >= next && n > from) {
            double e = 0, pk = 0; for (int i = from; i < n; i++) { e += all[i] * all[i]; if (fabs(all[i]) > pk) pk = fabs(all[i]); }
            double db = 10 * log10(e / (n - from) + 1e-12); if (pk > peak_all) peak_all = pk;
            int bars = (int)((db + 70) / 2); if (bars < 0) bars = 0; if (bars > 35) bars = 35;
            printf("%5.1fs %6.1f dBFS peak %.3f |%.*s\n", pk_now() - t0, db, pk, bars, "###################################");
            from = n; next += 0.1;
        }
        sleep_ms(10);
    }
    pk_mic_stop(mic); pk_mic_close(mic);
    double e = 0; for (int i = 0; i < n; i++) e += all[i] * all[i];
    printf("captured %d samples (%.2fs at 16 kHz), rms %.1f dBFS, peak %.3f\n", n, n / (double)PK_SR, 10 * log10(e / (n + 1) + 1e-12), peak_all);
    if (g_save) { pk_wav_write16("mic-test.wav", all, n); printf("wrote mic-test.wav\n"); }
    free(all); return n > 0 ? 0 : 1;
}

int main(int argc, char **argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    HANDLE ho = GetStdHandle(STD_ERROR_HANDLE); DWORD mode;
    if (GetConsoleMode(ho, &mode)) SetConsoleMode(ho, mode | 0x0004);
#endif
    const char *model_arg = NULL; int stdio_mode = 0, mic = 0, mictest = 0, timing = 0, bench = 0; double secs = 0; const char *stream_file = NULL;
    const char *files[64]; int nf = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model_arg = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) g_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--stdio")) stdio_mode = 1;
        else if (!strcmp(argv[i], "--mic")) mic = 1;
        else if (!strcmp(argv[i], "--mic-test")) { mictest = 1; if (i + 1 < argc && argv[i + 1][0] != '-') secs = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--seconds") && i + 1 < argc) secs = atof(argv[++i]);
        else if (!strcmp(argv[i], "--timing")) timing = 1;
        else if (!strcmp(argv[i], "--save")) g_save = 1;
        else if (!strcmp(argv[i], "--mic-list")) return pk_mic_list() ? 2 : 0;
        else if (!strcmp(argv[i], "--bench") && i + 1 < argc) bench = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--stream-test") && i + 1 < argc) stream_file = argv[++i];
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            printf("parakeet-c: native C runtime for NVIDIA Parakeet TDT/CTC speech recognition\n"
                   "  parakeet-c [--model DIR] [--threads N] [--timing] [--bench N] file.wav ...\n"
                   "  parakeet-c --mic [--seconds N]   live dictation, partials on stderr, finals on stdout\n"
                   "  parakeet-c --stdio               GUI protocol (START/STOP/CANCEL/FILE path/PING/QUIT)\n"
                   "  parakeet-c --mic-test [N] [--save] mic level test (--save writes mic-test.wav)\n"
                   "  parakeet-c --stream-test f.wav   replay a WAV through the live dictation path in real time\n"
                   "  env PARAKEET_THREADS=N overrides the default thread count (8)\n"
                   "  parakeet-c --mic-list           list microphones\n"
                   "env: PARAKEET_MIC=<name part>, PARAKEET_MODEL, PARAKEET_HOP_MS, PARAKEET_PAUSE_MS, PARAKEET_MIC_ROLE=communications\n");
            return 0;
        } else if (nf < 64) files[nf++] = argv[i];
    }
    if (g_threads <= 0 && getenv("PARAKEET_THREADS")) g_threads = atoi(getenv("PARAKEET_THREADS"));
    if (g_threads <= 0) g_threads = default_threads();
    if (mictest) return mode_mic_test(secs > 0 ? secs : 5);
    char mbuf[1024]; const char *dir = find_model_dir(model_arg, mbuf, sizeof mbuf);
    pk_model m; double t0 = pk_now();
    if (pk_model_load(&m, dir)) { if (stdio_mode) out_line("ERR", "model load failed"); fprintf(stderr, "cannot load model from %s\n", dir); return 2; }
    if (timing) fprintf(stderr, "model: %s  %d layers d%d heads %d ff %d %s vocab %d mel %d  (%.3fs)\n", dir, m.n_layers, m.d_model, m.n_heads, m.d_ff,
                        m.is_tdt == 1 ? "TDT" : m.is_tdt == 2 ? "CTC" : "RNNT", m.vocab, m.mel.nmel, pk_now() - t0);
    if (stdio_mode) return mode_stdio(&m);
    if (stream_file) return mode_stream_test(&m, stream_file);
    if (mic) return mode_mic(&m, secs, timing);
    if (!nf) { fprintf(stderr, "usage: parakeet-c file.wav | --mic | --stdio | --help\n"); return 1; }
    for (int i = 0; i < nf; i++) {
        pk_audio a; if (pk_wav_read(files[i], &a)) { fprintf(stderr, "cannot read %s (WAV PCM/float only)\n", files[i]); continue; }
        int reps = bench > 0 ? bench : 1;
        for (int r = 0; r < reps; r++) {
            pk_timing tm; char *s = run_transcribe(&m, a.x, a.n, &tm);
            if (r == reps - 1) { printf("%s\n", s); fflush(stdout); }
            if (timing || bench) fprintf(stderr, "[%s] audio %.2fs  mel %.3f  enc %.3f  dec %.3f  total %.3fs  RTF %.3f  frames %d tokens %d threads %d\n",
                                         files[i], tm.audio_s, tm.mel_s, tm.enc_s, tm.dec_s, tm.total_s, tm.total_s / tm.audio_s, tm.frames, tm.tokens, g_threads);
            free(s);
        }
        free(a.x);
    }
    pk_model_free(&m);
    return 0;
}
