/* parakeet-c: native Windows (and portable) pure-C runtime for NVIDIA
 * Parakeet TDT/CTC FastConformer speech recognition.
 * Pure C11. No Python, no WSL, no mutexes: one main thread for I/O and a
 * lock-free fork/join worker pool (atomics + phase boundaries) for math. */
#ifndef PK_H
#define PK_H
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define PK_SR 16000
#define PK_HOP 160
#define PK_WIN 400
#define PK_NFFT 512
#define PK_NBIN 257

/* ---------- utils ---------- */
double pk_now(void);                        /* seconds, monotonic */
void  *pk_alloc(size_t n);                  /* 64-byte aligned, zeroed */
void   pk_free(void *p);
float  pk_bf16_to_f32(uint16_t h);

/* ---------- thread pool (lock-free fork/join) ---------- */
typedef void (*pk_task_fn)(void *ctx, int task);
void pk_pool_start(int nthreads);           /* phase boundary: spawn */
void pk_pool_stop(void);                    /* phase boundary: join  */
void pk_parallel_for(int ntasks, pk_task_fn fn, void *ctx);
int  pk_pool_threads(void);

/* ---------- safetensors (memory mapped, read only) ---------- */
typedef enum { PK_F32 = 0, PK_BF16 = 1, PK_F16 = 2, PK_I64 = 3, PK_OTHER = 9 } pk_dtype;
typedef struct {
    char name[160];
    pk_dtype dtype;
    int ndim;
    int64_t shape[4];
    const uint8_t *data;
    size_t nbytes;
} pk_tensor;
typedef struct {
    void *map_handle, *file_handle;  /* Win32 handles (or fd) */
    const uint8_t *base;
    size_t size;
    pk_tensor *t;
    int n;
} pk_st;
int  pk_st_open(pk_st *st, const char *path);
void pk_st_close(pk_st *st);
const pk_tensor *pk_st_get(const pk_st *st, const char *name);
int64_t pk_tensor_numel(const pk_tensor *t);
/* copy a tensor out as f32 (any of f32/bf16/f16); returns malloc'd array */
float *pk_tensor_f32(const pk_tensor *t);

/* ---------- audio ---------- */
typedef struct { float *x; int n; } pk_audio;
int  pk_wav_read(const char *path, pk_audio *out);  /* any PCM16/24/32/float, any rate/ch -> 16k mono */
int  pk_wav_write16(const char *path, const float *x, int n);
/* windowed-sinc resampler, mono */
float *pk_resample(const float *x, int n, int sr_in, int sr_out, int *n_out);

/* ---------- mel front end (NeMo AudioToMelSpectrogramPreprocessor) ---------- */
typedef struct {
    int nmel;            /* 80 or 128 */
    float *fb;           /* nmel x 257 */
    float *window;       /* 400 */
    float preemph;       /* 0.97 */
    float log_guard;     /* 2^-24 */
} pk_mel;
/* out: [T][nmel] (time major), returns T = n / 160 (NeMo get_seq_len) */
int  pk_mel_compute(const pk_mel *m, const float *x, int n, float **out);
void pk_mel_default_fb(pk_mel *m, int nmel);   /* librosa slaney mel + periodic=False hann */
void pk_fft_real512(const float *in, float *re, float *im); /* exposed for tests */

/* ---------- GEMM: C[M,N] = A[M,K] * W[N,K]^T (+bias) ---------- */
void pk_gemm_f32w(float *C, const float *A, const float *W, const float *bias, int M, int N, int K);
void pk_gemm_bf16w(float *C, const float *A, const uint16_t *W, const float *bias, int M, int N, int K);
void pk_gemm_ref(float *C, const float *A, const float *W, const float *bias, int M, int N, int K);

/* ---------- model ---------- */
typedef struct {
    const uint16_t *w; /* bf16 [N][K] mapped */
    const float *wf;   /* or f32 [N][K] (owned) */
    float *b;          /* bias or NULL */
    int N, K;
} pk_lin;

typedef struct {
    float *ln_ff1_w, *ln_ff1_b, *ln_att_w, *ln_att_b, *ln_conv_w, *ln_conv_b,
          *ln_ff2_w, *ln_ff2_b, *ln_out_w, *ln_out_b;
    pk_lin ff1_1, ff1_2, ff2_1, ff2_2;
    pk_lin q, k, v, o, pos;
    float *bias_u, *bias_v;   /* [H][dk] */
    pk_lin pw1, pw2;
    float *dw;                /* [d][ks] */
    float *dw_b;              /* or NULL */
    float *bn_scale, *bn_shift; /* folded batch norm */
} pk_layer;

/* Largest shapes the runtime accepts. The decode path keeps its LSTM
 * state, gates, joint vectors and attention rows in fixed stack buffers
 * sized by these; a model beyond them is rejected at load, never decoded. */
#define PK_MAX_PRED 1024      /* LSTM hidden and joint width: lstm_state, g/gp/hsum */
#define PK_MAX_HEAD_DIM 256   /* d_model / n_heads: qu/qv in att_task */
#define PK_MAX_DUR 8          /* TDT durations: pk_model.durations */
#define PK_MAX_DIM 8192
#define PK_MAX_FF 32768
#define PK_MAX_LAYERS 256
#define PK_MAX_VOCAB 262144
#define PK_CHECK_ENCODER 1
#define PK_CHECK_DECODER 2
typedef struct {
    int d_model, n_layers, n_heads, d_ff, conv_ks, feat_in, sub_ch;
    int vocab;       /* 1024 tokens; blank = vocab */
    int n_dur; int durations[PK_MAX_DUR];
    int pred_hidden, joint_hidden, pred_layers;
    int is_tdt;      /* 1 TDT, 0 plain RNNT, 2 CTC */
    /* pre_encode (dw_striding, factor 8) */
    float *c0_w, *c0_b;        /* [ch][3][3] */
    float *c2_w, *c2_b;        /* dw [ch][3][3] */
    float *c3_w, *c3_b;        /* pw [ch][ch] */
    float *c5_w, *c5_b, *c6_w, *c6_b;
    pk_lin pre_out;            /* [d][ch*F/8] */
    pk_layer *L;
    /* decoder / joint */
    float *embed;              /* [vocab+1][pred_hidden] */
    float *lstm_wih[2], *lstm_whh[2], *lstm_b[2]; /* b = b_ih + b_hh */
    pk_lin j_enc, j_pred, j_out;
    pk_lin ctc;                /* for CTC models */
    pk_mel mel;
    char **pieces; int n_pieces;
    pk_st st;
} pk_model;

int  pk_model_check_dims(const pk_model *m, int stage);   /* 0 when the shapes fit the runtime */
int  pk_model_check_layer(const pk_model *m, const pk_layer *L);
int  pk_model_load(pk_model *m, const char *dir);  /* dir with model_weights.safetensors + tokenizer.model */
void pk_model_free(pk_model *m);
/* encoder: feats [T][feat_in] -> enc [T'][d_model]; returns T' */
int  pk_encode(const pk_model *m, const float *feats, int T, float **enc_out);
/* TDT / RNNT greedy decode; returns n tokens into *tok (malloc), frame idx into *tstamp */
int  pk_decode_greedy(const pk_model *m, const float *enc, int Te, int **tok, int **tstamp);
/* tokens -> UTF-8 text (malloc) */
char *pk_detok(const pk_model *m, const int *tok, int n);
/* whole pipeline: 16k mono float audio -> text (malloc). timing out optional */
typedef struct { double mel_s, enc_s, dec_s, total_s, audio_s; int frames, tokens; } pk_timing;
char *pk_transcribe(const pk_model *m, const float *x, int n, pk_timing *tm);

/* ---------- tokenizer ---------- */
int pk_spm_load(const char *path, char ***pieces, int *n);   /* SentencePiece ModelProto */
char *pk_spm_detok(char **pieces, int npieces, const int *ids, int n);

/* ---------- mic (WASAPI shared, event-free polling, single thread) ---------- */
typedef struct pk_mic pk_mic;
pk_mic *pk_mic_open(char *err, int errlen);       /* default capture endpoint */
int     pk_mic_start(pk_mic *m);
/* drains whatever the endpoint has, appends 16k mono floats; returns count */
int     pk_mic_read(pk_mic *m, float *dst, int cap);
void    pk_mic_stop(pk_mic *m);
void    pk_mic_close(pk_mic *m);
const char *pk_mic_describe(pk_mic *m);
int     pk_mic_list(void);   /* prints active capture endpoints */

#endif
