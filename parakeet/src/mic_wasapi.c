/* parakeet-c mic capture: WASAPI shared mode, default capture endpoint,
 * polled from the caller's single thread (no capture thread, no locks).
 * Asks Windows for 16 kHz mono float with AUTOCONVERTPCM; if the endpoint
 * refuses, captures the mix format and runs our own streaming windowed-sinc
 * resampler + downmix. */
#include "pk.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _WIN32
#define COBJMACROS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <initguid.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <functiondiscoverykeys_devpkey.h>
#include <mmreg.h>
#include <ks.h>
#include <ksmedia.h>

static const GUID PK_SUBTYPE_FLOAT = { 0x00000003, 0x0000, 0x0010, { 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71 } };

struct pk_mic {
    IMMDeviceEnumerator *en; IMMDevice *dev; IAudioClient *ac; IAudioCaptureClient *cc;
    int native_rate, native_ch, is_float, bits, converting;
    /* streaming resampler state (native mono -> 16k) */
    float *hist; int hn, hcap; double tpos;
    char desc[256];
};

static void fill_fmt(WAVEFORMATEX *w, int rate, int ch) {
    memset(w, 0, sizeof *w);
    w->wFormatTag = WAVE_FORMAT_IEEE_FLOAT; w->nChannels = (WORD)ch; w->nSamplesPerSec = rate;
    w->wBitsPerSample = 32; w->nBlockAlign = (WORD)(4 * ch); w->nAvgBytesPerSec = rate * 4 * ch; w->cbSize = 0;
}

static void dev_name(IMMDevice *d, char *out, int cap) {
    IPropertyStore *ps = NULL; snprintf(out, cap, "?");
    if (SUCCEEDED(IMMDevice_OpenPropertyStore(d, STGM_READ, &ps))) {
        PROPVARIANT pv; PropVariantInit(&pv);
        if (SUCCEEDED(IPropertyStore_GetValue(ps, &PKEY_Device_FriendlyName, &pv)) && pv.vt == VT_LPWSTR)
            WideCharToMultiByte(CP_UTF8, 0, pv.pwszVal, -1, out, cap, NULL, NULL);
        PropVariantClear(&pv); IPropertyStore_Release(ps);
    }
}
static int ci_contains(const char *h, const char *n) {
    size_t ln = strlen(n);
    for (; *h; h++) { size_t i = 0; while (i < ln && h[i] && ((h[i] | 32) == (n[i] | 32))) i++; if (i == ln) return 1; }
    return 0;
}
static HRESULT pick_by_name(IMMDeviceEnumerator *en, const char *want, IMMDevice **out) {
    IMMDeviceCollection *col = NULL; UINT n = 0; HRESULT res = E_FAIL;
    if (FAILED(IMMDeviceEnumerator_EnumAudioEndpoints(en, eCapture, DEVICE_STATE_ACTIVE, &col))) return E_FAIL;
    IMMDeviceCollection_GetCount(col, &n);
    for (UINT i = 0; i < n && FAILED(res); i++) {
        IMMDevice *d = NULL; char nm[256];
        if (FAILED(IMMDeviceCollection_Item(col, i, &d))) continue;
        dev_name(d, nm, sizeof nm);
        if (ci_contains(nm, want)) { *out = d; res = S_OK; } else IMMDevice_Release(d);
    }
    IMMDeviceCollection_Release(col);
    return res;
}
int pk_mic_list(void) {
    if (FAILED(CoInitializeEx(NULL, COINIT_APARTMENTTHREADED))) {}
    IMMDeviceEnumerator *en = NULL; IMMDeviceCollection *col = NULL; UINT n = 0;
    if (FAILED(CoCreateInstance(&CLSID_MMDeviceEnumerator, NULL, CLSCTX_ALL, &IID_IMMDeviceEnumerator, (void **)&en))) return -1;
    IMMDevice *def = NULL; char defn[256] = "";
    if (SUCCEEDED(IMMDeviceEnumerator_GetDefaultAudioEndpoint(en, eCapture, eConsole, &def))) { dev_name(def, defn, sizeof defn); IMMDevice_Release(def); }
    IMMDeviceEnumerator_EnumAudioEndpoints(en, eCapture, DEVICE_STATE_ACTIVE, &col);
    if (col) IMMDeviceCollection_GetCount(col, &n);
    for (UINT i = 0; i < n; i++) {
        IMMDevice *d = NULL; char nm[256];
        if (FAILED(IMMDeviceCollection_Item(col, i, &d))) continue;
        dev_name(d, nm, sizeof nm);
        printf("%s %s\n", strcmp(nm, defn) ? " " : "*", nm);
        IMMDevice_Release(d);
    }
    if (col) IMMDeviceCollection_Release(col);
    IMMDeviceEnumerator_Release(en);
    printf("(* = Windows default; set PARAKEET_MIC=<part of name> to pick another)\n");
    return 0;
}

pk_mic *pk_mic_open(char *err, int errlen) {
    HRESULT hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
    if (FAILED(hr) && hr != RPC_E_CHANGED_MODE) { snprintf(err, errlen, "CoInitializeEx 0x%08lx", (unsigned long)hr); return NULL; }
    pk_mic *m = calloc(1, sizeof *m);
    hr = CoCreateInstance(&CLSID_MMDeviceEnumerator, NULL, CLSCTX_ALL, &IID_IMMDeviceEnumerator, (void **)&m->en);
    if (FAILED(hr)) { snprintf(err, errlen, "MMDeviceEnumerator 0x%08lx", (unsigned long)hr); free(m); return NULL; }
    const char *role = getenv("PARAKEET_MIC_ROLE");
    ERole r = (role && !strcmp(role, "communications")) ? eCommunications : eConsole;
    const char *want_name = getenv("PARAKEET_MIC");
    hr = E_FAIL;
    if (want_name && *want_name) hr = pick_by_name(m->en, want_name, &m->dev);
    if (FAILED(hr)) hr = IMMDeviceEnumerator_GetDefaultAudioEndpoint(m->en, eCapture, r, &m->dev);
    if (FAILED(hr)) { snprintf(err, errlen, "no default microphone (0x%08lx)", (unsigned long)hr); pk_mic_close(m); return NULL; }
    IPropertyStore *ps = NULL;
    snprintf(m->desc, sizeof m->desc, "default microphone");
    if (SUCCEEDED(IMMDevice_OpenPropertyStore(m->dev, STGM_READ, &ps))) {
        PROPVARIANT pv; PropVariantInit(&pv);
        if (SUCCEEDED(IPropertyStore_GetValue(ps, &PKEY_Device_FriendlyName, &pv)) && pv.vt == VT_LPWSTR)
            WideCharToMultiByte(CP_UTF8, 0, pv.pwszVal, -1, m->desc, sizeof m->desc, NULL, NULL);
        PropVariantClear(&pv); IPropertyStore_Release(ps);
    }
    hr = IMMDevice_Activate(m->dev, &IID_IAudioClient, CLSCTX_ALL, NULL, (void **)&m->ac);
    if (FAILED(hr)) { snprintf(err, errlen, "IAudioClient 0x%08lx", (unsigned long)hr); pk_mic_close(m); return NULL; }
    REFERENCE_TIME buf = 20000000; /* 2 s, enough to ride out a decode */
    WAVEFORMATEX want; fill_fmt(&want, PK_SR, 1);
    const DWORD conv = 0x80000000 /*AUTOCONVERTPCM*/ | 0x08000000 /*SRC_DEFAULT_QUALITY*/;
    hr = IAudioClient_Initialize(m->ac, AUDCLNT_SHAREMODE_SHARED, conv, buf, 0, &want, NULL);
    if (SUCCEEDED(hr)) { m->native_rate = PK_SR; m->native_ch = 1; m->is_float = 1; m->bits = 32; m->converting = 0; }
    else {
        WAVEFORMATEX *mix = NULL;
        IAudioClient_Release(m->ac); m->ac = NULL;
        hr = IMMDevice_Activate(m->dev, &IID_IAudioClient, CLSCTX_ALL, NULL, (void **)&m->ac);
        if (FAILED(hr) || FAILED(IAudioClient_GetMixFormat(m->ac, &mix))) { snprintf(err, errlen, "GetMixFormat failed"); pk_mic_close(m); return NULL; }
        m->native_rate = mix->nSamplesPerSec; m->native_ch = mix->nChannels; m->bits = mix->wBitsPerSample;
        m->is_float = mix->wFormatTag == WAVE_FORMAT_IEEE_FLOAT ||
            (mix->wFormatTag == WAVE_FORMAT_EXTENSIBLE && IsEqualGUID(&((WAVEFORMATEXTENSIBLE *)mix)->SubFormat, &PK_SUBTYPE_FLOAT));
        hr = IAudioClient_Initialize(m->ac, AUDCLNT_SHAREMODE_SHARED, 0, buf, 0, mix, NULL);
        CoTaskMemFree(mix);
        if (FAILED(hr)) { snprintf(err, errlen, "IAudioClient_Initialize 0x%08lx", (unsigned long)hr); pk_mic_close(m); return NULL; }
        m->converting = 1;
        m->hcap = 1 << 18; m->hist = malloc(sizeof(float) * m->hcap);
    }
    hr = IAudioClient_GetService(m->ac, &IID_IAudioCaptureClient, (void **)&m->cc);
    if (FAILED(hr)) { snprintf(err, errlen, "IAudioCaptureClient 0x%08lx", (unsigned long)hr); pk_mic_close(m); return NULL; }
    size_t dl = strlen(m->desc);
    snprintf(m->desc + dl, sizeof m->desc - dl, " (%d Hz %dch %s%s)", m->native_rate, m->native_ch,
             m->is_float ? "float" : "pcm", m->converting ? ", own resampler" : ", Windows SRC");
    return m;
}
const char *pk_mic_describe(pk_mic *m) { return m->desc; }
int pk_mic_start(pk_mic *m) { m->hn = 0; m->tpos = 0; return FAILED(IAudioClient_Start(m->ac)) ? -1 : 0; }
void pk_mic_stop(pk_mic *m) { IAudioClient_Stop(m->ac); IAudioClient_Reset(m->ac); }

/* windowed-sinc streaming resample of m->hist into dst */
static int resample_drain(pk_mic *m, float *dst, int cap) {
    const double ratio = (double)PK_SR / m->native_rate, cutoff = ratio < 1 ? ratio : 1;
    const int r = (int)ceil(16 / cutoff);
    int n = 0;
    while (n < cap) {
        double t = m->tpos; int c = (int)floor(t);
        if (c + r >= m->hn) break;
        double acc = 0;
        for (int k = c - r + 1; k <= c + r; k++) {
            if (k < 0) continue;
            double d = (t - k) * cutoff, s = fabs(d) < 1e-9 ? 1.0 : sin(M_PI * d) / (M_PI * d);
            double wx = (t - k) / r, win = fabs(wx) >= 1 ? 0 : 0.5 + 0.5 * cos(M_PI * wx);
            acc += m->hist[k] * s * win;
        }
        dst[n++] = (float)(acc * cutoff);
        m->tpos += 1.0 / ratio;
    }
    int drop = (int)floor(m->tpos) - r; /* keep r samples of history */
    if (drop > 0) { memmove(m->hist, m->hist + drop, sizeof(float) * (m->hn - drop)); m->hn -= drop; m->tpos -= drop; }
    return n;
}

int pk_mic_read(pk_mic *m, float *dst, int cap) {
    int n = 0; UINT32 pkt = 0;
    while (SUCCEEDED(IAudioCaptureClient_GetNextPacketSize(m->cc, &pkt)) && pkt > 0) {
        BYTE *data; UINT32 frames; DWORD flags;
        if (FAILED(IAudioCaptureClient_GetBuffer(m->cc, &data, &frames, &flags, NULL, NULL))) break;
        const int ch = m->native_ch, silent = (flags & AUDCLNT_BUFFERFLAGS_SILENT) != 0;
        for (UINT32 i = 0; i < frames; i++) {
            float s = 0;
            if (!silent) for (int c = 0; c < ch; c++) {
                if (m->is_float) s += ((float *)data)[i * ch + c];
                else if (m->bits == 16) s += ((int16_t *)data)[i * ch + c] / 32768.f;
                else if (m->bits == 32) s += ((int32_t *)data)[i * ch + c] / 2147483648.f;
                else if (m->bits == 24) { const uint8_t *q = data + (i * ch + c) * 3; s += ((int32_t)((uint32_t)q[0] << 8 | (uint32_t)q[1] << 16 | (uint32_t)q[2] << 24) >> 8) / 8388608.f; }
            }
            s /= ch;
            if (!m->converting) { if (n < cap) dst[n++] = s; }
            else { if (m->hn == m->hcap) { m->hcap *= 2; m->hist = realloc(m->hist, sizeof(float) * m->hcap); } m->hist[m->hn++] = s; }
        }
        IAudioCaptureClient_ReleaseBuffer(m->cc, frames);
        if (m->converting) n += resample_drain(m, dst + n, cap - n);
    }
    return n;
}
void pk_mic_close(pk_mic *m) {
    if (!m) return;
    if (m->cc) IAudioCaptureClient_Release(m->cc);
    if (m->ac) IAudioClient_Release(m->ac);
    if (m->dev) IMMDevice_Release(m->dev);
    if (m->en) IMMDeviceEnumerator_Release(m->en);
    free(m->hist); free(m);
}
#else
struct pk_mic { int dummy; };
pk_mic *pk_mic_open(char *err, int errlen) { snprintf(err, errlen, "mic capture is WASAPI (Windows) only"); return NULL; }
int pk_mic_list(void) { printf("mic capture is WASAPI (Windows) only\n"); return -1; }
int pk_mic_start(pk_mic *m) { (void)m; return -1; }
int pk_mic_read(pk_mic *m, float *d, int c) { (void)m; (void)d; (void)c; return 0; }
void pk_mic_stop(pk_mic *m) { (void)m; }
void pk_mic_close(pk_mic *m) { (void)m; }
const char *pk_mic_describe(pk_mic *m) { (void)m; return "none"; }
#endif
