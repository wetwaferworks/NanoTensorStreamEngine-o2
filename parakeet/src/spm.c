/* parakeet-c SentencePiece: reads the ModelProto (protobuf) piece table from
 * tokenizer.model and detokenizes ids (U+2581 -> space, <0xNN> byte pieces,
 * control/unknown pieces dropped). No protobuf library needed. */
#include "pk.h"
#include <stdlib.h>
#include <string.h>

static uint64_t varint(const uint8_t **p, const uint8_t *e) {
    uint64_t v = 0; int s = 0;
    while (*p < e) { uint8_t b = *(*p)++; v |= (uint64_t)(b & 0x7f) << s; if (!(b & 0x80)) break; s += 7; }
    return v;
}
static int skip_field(const uint8_t **p, const uint8_t *e, int wt) {
    if (wt == 0) varint(p, e); else if (wt == 1) *p += 8; else if (wt == 5) *p += 4;
    else if (wt == 2) { uint64_t l = varint(p, e); *p += l; } else return -1;
    return *p <= e ? 0 : -1;
}

/* piece types: 1 normal, 2 unknown, 3 control, 4 user defined, 6 byte */
static int *g_types; /* parallel to pieces, owned by first load (single model) */

int pk_spm_load(const char *path, char ***pieces, int *n) {
    FILE *f = fopen(path, "rb"); if (!f) return -1;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc(sz); if (fread(b, 1, sz, f) != (size_t)sz) { fclose(f); free(b); return -1; } fclose(f);
    int cap = 256, cnt = 0; char **out = malloc(sizeof(char *) * cap); int *types = malloc(sizeof(int) * cap);
    const uint8_t *p = b, *e = b + sz;
    while (p < e) {
        uint64_t key = varint(&p, e); int fn = (int)(key >> 3), wt = (int)(key & 7);
        if (fn == 1 && wt == 2) {
            uint64_t l = varint(&p, e); const uint8_t *q = p, *qe = p + l; p += l;
            char *piece = NULL; int type = 1;
            while (q < qe) {
                uint64_t k2 = varint(&q, qe); int f2 = (int)(k2 >> 3), w2 = (int)(k2 & 7);
                if (f2 == 1 && w2 == 2) { uint64_t sl = varint(&q, qe); piece = malloc(sl + 1); memcpy(piece, q, sl); piece[sl] = 0; q += sl; }
                else if (f2 == 3 && w2 == 0) type = (int)varint(&q, qe);
                else if (skip_field(&q, qe, w2)) break;
            }
            if (cnt == cap) { cap *= 2; out = realloc(out, sizeof(char *) * cap); types = realloc(types, sizeof(int) * cap); }
            out[cnt] = piece ? piece : calloc(1, 1); types[cnt] = type; cnt++;
        } else if (skip_field(&p, e, wt)) break;
    }
    free(b);
    if (!cnt) { free(out); free(types); return -2; }
    g_types = types;
    *pieces = out; *n = cnt;
    return 0;
}

char *pk_spm_detok(char **pieces, int npieces, const int *ids, int n) {
    size_t cap = 64; for (int i = 0; i < n; i++) if (ids[i] >= 0 && ids[i] < npieces) cap += strlen(pieces[ids[i]]) + 1;
    char *o = malloc(cap); size_t w = 0;
    for (int i = 0; i < n; i++) {
        int id = ids[i]; if (id < 0 || id >= npieces) continue;
        int type = g_types ? g_types[id] : 1;
        const char *s = pieces[id];
        if (type == 2 || type == 3) continue;
        if (type == 6 || (s[0] == '<' && s[1] == '0' && s[2] == 'x' && strlen(s) == 6)) { o[w++] = (char)strtol(s + 3, NULL, 16); continue; }
        for (const char *c = s; *c; ) {
            if ((uint8_t)c[0] == 0xE2 && (uint8_t)c[1] == 0x96 && (uint8_t)c[2] == 0x81) { o[w++] = ' '; c += 3; }
            else o[w++] = *c++;
        }
    }
    o[w] = 0;
    /* strip leading space, collapse doubled spaces */
    char *r = o; while (*r == ' ') r++;
    size_t k = 0; for (char *c = r; *c; c++) { if (*c == ' ' && k && o[k - 1] == ' ') continue; o[k++] = *c; }
    o[k] = 0;
    while (k && o[k - 1] == ' ') o[--k] = 0;
    return o;
}
