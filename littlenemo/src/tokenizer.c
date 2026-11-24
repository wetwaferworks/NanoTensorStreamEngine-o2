/* tokenizer.c - see tokenizer.h. */
#include "tokenizer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"

/* --- byte <-> unicode mapping (GPT-2 byte-level BPE) --- */
static unsigned byte_to_cp[256];
static int cp_to_byte[512];
static int mapping_ready = 0;

static void build_mapping(void) {
    if (mapping_ready) return;
    for (int i = 0; i < 512; i++) cp_to_byte[i] = -1;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        unsigned cp;
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) ||
            (b >= 174 && b <= 255))
            cp = (unsigned)b;
        else
            cp = 256u + (unsigned)n++;
        byte_to_cp[b] = cp;
        cp_to_byte[cp] = b;
    }
    mapping_ready = 1;
}

static int utf8_encode_cp(char *dst, unsigned cp) {
    if (cp < 0x80) {
        dst[0] = (char)cp;
        return 1;
    }
    if (cp < 0x800) {
        dst[0] = (char)(0xc0 | (cp >> 6));
        dst[1] = (char)(0x80 | (cp & 0x3f));
        return 2;
    }
    dst[0] = (char)(0xe0 | (cp >> 12));
    dst[1] = (char)(0x80 | ((cp >> 6) & 0x3f));
    dst[2] = (char)(0x80 | (cp & 0x3f));
    return 3;
}

/* Decode one UTF-8 code point; returns bytes consumed or -1. */
static int utf8_decode_cp(const unsigned char *s, size_t len, unsigned *cp) {
    if (len < 1) return -1;
    unsigned c = s[0];
    if (c < 0x80) {
        *cp = c;
        return 1;
    }
    if ((c & 0xe0) == 0xc0 && len >= 2) {
        *cp = ((c & 0x1f) << 6) | (s[1] & 0x3f);
        return 2;
    }
    if ((c & 0xf0) == 0xe0 && len >= 3) {
        *cp = ((c & 0x0f) << 12) | ((s[1] & 0x3f) << 6) | (s[2] & 0x3f);
        return 3;
    }
    if ((c & 0xf8) == 0xf0 && len >= 4) {
        *cp = ((c & 0x07) << 18) | ((s[1] & 0x3f) << 12) |
              ((s[2] & 0x3f) << 6) | (s[3] & 0x3f);
        return 4;
    }
    return -1;
}

/* --- string -> int hash (open addressing) --- */
typedef struct {
    char **keys;
    int *vals;
    int cap, used;
} smap;

static uint64_t hash_str(const char *s) {
    uint64_t h = 14695981039346656037ull;
    while (*s) {
        h ^= (unsigned char)*s++;
        h *= 1099511628211ull;
    }
    return h;
}

static int smap_init(smap *m, int cap) {
    m->keys = (char **)calloc((size_t)cap, sizeof(char *));
    m->vals = (int *)malloc((size_t)cap * sizeof(int));
    if (!m->keys || !m->vals) {
        free(m->keys);
        free(m->vals);
        return -1;
    }
    m->cap = cap;
    m->used = 0;
    return 0;
}

static void smap_free(smap *m) {
    if (m->keys)
        for (int i = 0; i < m->cap; i++) free(m->keys[i]);
    free(m->keys);
    free(m->vals);
}

static int smap_put(smap *m, const char *key, int val) {
    if (m->used * 2 >= m->cap) return -1; /* sized generously up front */
    uint64_t h = hash_str(key) % (uint64_t)m->cap;
    while (m->keys[h]) {
        if (strcmp(m->keys[h], key) == 0) {
            m->vals[h] = val;
            return 0;
        }
        h = (h + 1) % (uint64_t)m->cap;
    }
    m->keys[h] = (char *)malloc(strlen(key) + 1);
    if (!m->keys[h]) return -1;
    strcpy(m->keys[h], key);
    m->vals[h] = val;
    m->used++;
    return 0;
}

static int smap_get(const smap *m, const char *key) {
    if (!m->keys) return -1;
    uint64_t h = hash_str(key) % (uint64_t)m->cap;
    while (m->keys[h]) {
        if (strcmp(m->keys[h], key) == 0) return m->vals[h];
        h = (h + 1) % (uint64_t)m->cap;
    }
    return -1;
}

/* --- pair -> rank hash --- */
typedef struct {
    uint64_t *keys;
    int *vals;
    unsigned char *taken;
    int cap;
} pmap;

static int pmap_init(pmap *m, int cap) {
    m->keys = (uint64_t *)malloc((size_t)cap * sizeof(uint64_t));
    m->vals = (int *)malloc((size_t)cap * sizeof(int));
    m->taken = (unsigned char *)calloc((size_t)cap, 1);
    if (!m->keys || !m->vals || !m->taken) {
        free(m->keys);
        free(m->vals);
        free(m->taken);
        return -1;
    }
    m->cap = cap;
    return 0;
}

static void pmap_free(pmap *m) {
    free(m->keys);
    free(m->vals);
    free(m->taken);
}

static void pmap_put(pmap *m, int a, int b, int rank) {
    uint64_t key = ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
    uint64_t h = (key * 0x9e3779b97f4a7c15ull) % (uint64_t)m->cap;
    while (m->taken[h]) h = (h + 1) % (uint64_t)m->cap;
    m->taken[h] = 1;
    m->keys[h] = key;
    m->vals[h] = rank;
}

static int pmap_get(const pmap *m, int a, int b) {
    if (!m->taken) return -1;
    uint64_t key = ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
    uint64_t h = (key * 0x9e3779b97f4a7c15ull) % (uint64_t)m->cap;
    while (m->taken[h]) {
        if (m->keys[h] == key) return m->vals[h];
        h = (h + 1) % (uint64_t)m->cap;
    }
    return -1;
}

/* --- the tokenizer --- */
struct nemo_tokenizer {
    smap vocab;      /* token string -> id */
    char **id_str;   /* id -> token string (owned) */
    char *is_special;
    int n_ids;
    pmap merges; /* (idA, idB) -> rank */
};

static char *read_file(const char *path, size_t *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseeko(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    off_t sz = ftello(f);
    if (sz <= 0 || sz > (off_t)(256u << 20)) {
        fclose(f);
        return NULL;
    }
    if (fseeko(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return NULL;
    }
    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    buf[sz] = '\0';
    *len_out = (size_t)sz;
    return buf;
}

nemo_tokenizer *nemo_tokenizer_load(const char *path) {
    build_mapping();
    size_t len = 0;
    char *text = read_file(path, &len);
    if (!text) return NULL;
    nemo_json *root = nemo_json_parse(text, len);
    free(text);
    if (!root) return NULL;
    nemo_tokenizer *tk = (nemo_tokenizer *)calloc(1, sizeof *tk);
    if (!tk) {
        nemo_json_free(root);
        return NULL;
    }
    const nemo_json *model = nemo_json_get(root, "model");
    const nemo_json *vocab = model ? nemo_json_get(model, "vocab") : NULL;
    const nemo_json *merges = model ? nemo_json_get(model, "merges") : NULL;
    const nemo_json *added = nemo_json_get(root, "added_tokens");
    if (!vocab || vocab->type != NEMO_JSON_OBJ) goto fail;

    int max_id = -1;
    for (int i = 0; i < vocab->count; i++) {
        int id = (int)vocab->items[i]->num;
        if (id > max_id) max_id = id;
    }
    int n_added = (added && added->type == NEMO_JSON_ARR) ? added->count : 0;
    for (int i = 0; i < n_added; i++) {
        const nemo_json *idj = nemo_json_get(added->items[i], "id");
        if (idj && (int)idj->num > max_id) max_id = (int)idj->num;
    }
    tk->n_ids = max_id + 1;
    tk->id_str = (char **)calloc((size_t)tk->n_ids, sizeof(char *));
    tk->is_special = (char *)calloc((size_t)tk->n_ids, 1);
    if (!tk->id_str || !tk->is_special) goto fail;
    if (smap_init(&tk->vocab, 1 << 19) != 0) goto fail;
    for (int i = 0; i < vocab->count; i++) {
        const char *s = vocab->keys[i];
        int id = (int)vocab->items[i]->num;
        tk->id_str[id] = (char *)malloc(strlen(s) + 1);
        if (!tk->id_str[id]) goto fail;
        strcpy(tk->id_str[id], s);
        if (smap_put(&tk->vocab, s, id) != 0) goto fail;
    }
    for (int i = 0; i < n_added; i++) {
        const nemo_json *a = added->items[i];
        const nemo_json *idj = nemo_json_get(a, "id");
        const nemo_json *content = nemo_json_get(a, "content");
        if (!idj || !content || content->type != NEMO_JSON_STR) continue;
        int id = (int)idj->num;
        if (!tk->id_str[id]) {
            tk->id_str[id] = (char *)malloc(strlen(content->str) + 1);
            if (!tk->id_str[id]) goto fail;
            strcpy(tk->id_str[id], content->str);
        }
        tk->is_special[id] = 1;
        if (smap_put(&tk->vocab, content->str, id) != 0) goto fail;
    }
    if (merges && merges->type == NEMO_JSON_ARR) {
        if (pmap_init(&tk->merges, 1 << 19) != 0) goto fail;
        for (int r = 0; r < merges->count; r++) {
            const nemo_json *mj = merges->items[r];
            const char *a = NULL, *b = NULL;
            char *left = NULL;
            if (mj->type == NEMO_JSON_STR) {
                const char *sp = strchr(mj->str, ' ');
                if (!sp) continue;
                size_t ln = (size_t)(sp - mj->str);
                left = (char *)malloc(ln + 1);
                if (!left) goto fail;
                memcpy(left, mj->str, ln);
                left[ln] = '\0';
                a = left;
                b = sp + 1;
            } else if (mj->type == NEMO_JSON_ARR && mj->count == 2 &&
                       mj->items[0]->type == NEMO_JSON_STR &&
                       mj->items[1]->type == NEMO_JSON_STR) {
                a = mj->items[0]->str;
                b = mj->items[1]->str;
            } else {
                continue;
            }
            int ia = smap_get(&tk->vocab, a);
            int ib = smap_get(&tk->vocab, b);
            if (ia >= 0 && ib >= 0) pmap_put(&tk->merges, ia, ib, r);
            free(left);
        }
    }
    nemo_json_free(root);
    return tk;

fail:
    nemo_json_free(root);
    nemo_tokenizer_free(tk);
    return NULL;
}

void nemo_tokenizer_free(nemo_tokenizer *tk) {
    if (!tk) return;
    smap_free(&tk->vocab);
    pmap_free(&tk->merges);
    if (tk->id_str)
        for (int i = 0; i < tk->n_ids; i++) free(tk->id_str[i]);
    free(tk->id_str);
    free(tk->is_special);
    free(tk);
}

int nemo_tokenizer_vocab_size(const nemo_tokenizer *tk) {
    return tk ? tk->n_ids : 0;
}

int nemo_tokenizer_special(const nemo_tokenizer *tk, const char *content) {
    if (!tk || !content) return -1;
    int id = smap_get(&tk->vocab, content);
    if (id < 0 || id >= tk->n_ids || !tk->is_special[id]) return -1;
    return id;
}

int nemo_tokenizer_encode(const nemo_tokenizer *tk, const char *utf8,
                          int *ids_out, int max_ids) {
    if (!tk || !utf8 || !ids_out) return -1;
    size_t nbytes = strlen(utf8);
    if (nbytes > (size_t)max_ids) return -1;
    int n = 0;
    for (size_t i = 0; i < nbytes; i++) {
        char sym[4];
        int w = utf8_encode_cp(sym, byte_to_cp[(unsigned char)utf8[i]]);
        sym[w] = '\0';
        int id = smap_get(&tk->vocab, sym);
        if (id < 0) return -1;
        ids_out[n++] = id;
    }
    /* Greedy lowest-rank merging. */
    for (;;) {
        int best_rank = -1, best_pos = -1;
        for (int i = 0; i + 1 < n; i++) {
            int r = pmap_get(&tk->merges, ids_out[i], ids_out[i + 1]);
            if (r >= 0 && (best_rank < 0 || r < best_rank)) {
                best_rank = r;
                best_pos = i;
            }
        }
        if (best_pos < 0) break;
        /* The merged token's string is the concatenation; it must be
         * in the vocab for a well-formed tokenizer file. */
        const char *sa = tk->id_str[ids_out[best_pos]];
        const char *sb = tk->id_str[ids_out[best_pos + 1]];
        size_t la = strlen(sa), lb = strlen(sb);
        char *cat = (char *)malloc(la + lb + 1);
        if (!cat) return -1;
        memcpy(cat, sa, la);
        memcpy(cat + la, sb, lb + 1);
        int merged = smap_get(&tk->vocab, cat);
        free(cat);
        if (merged < 0) return -1;
        /* Merge every non-overlapping occurrence of this pair. The
         * pair's ids are captured first: compaction overwrites the
         * array in place, so indexing best_pos inside the loop would
         * read already-merged values. */
        const int pa = ids_out[best_pos], pb = ids_out[best_pos + 1];
        int w = 0;
        for (int i = 0; i < n; i++) {
            if (i + 1 < n && ids_out[i] == pa && ids_out[i + 1] == pb &&
                pmap_get(&tk->merges, ids_out[i], ids_out[i + 1]) ==
                    best_rank) {
                ids_out[w++] = merged;
                i++;
            } else {
                ids_out[w++] = ids_out[i];
            }
        }
        n = w;
    }
    return n;
}

char *nemo_tokenizer_decode(const nemo_tokenizer *tk, const int *ids, int n,
                            int skip_specials) {
    if (!tk || !ids || n < 0) return NULL;
    /* Scratch holds mapped-char text awaiting byte unmapping. */
    size_t cap = 256, len = 0, ocap = 256, olen = 0;
    char *scratch = (char *)malloc(cap);
    char *out = (char *)malloc(ocap);
    if (!scratch || !out) {
        free(scratch);
        free(out);
        return NULL;
    }
    for (int i = 0; i < n; i++) {
        int id = ids[i];
        if (id < 0 || id >= tk->n_ids || !tk->id_str[id]) continue;
        const char *s = tk->id_str[id];
        size_t sl = strlen(s);
        if (tk->is_special[id]) {
            if (skip_specials) continue;
            /* Flush pending mapped text, then emit literally. */
            size_t p = 0;
            while (p < len) {
                unsigned cp;
                int w = utf8_decode_cp((const unsigned char *)scratch + p,
                                       len - p, &cp);
                if (w < 0) {
                    p++;
                    continue;
                }
                p += (size_t)w;
                int b = (cp < 512) ? cp_to_byte[cp] : -1;
                if (b < 0) continue;
                if (olen + 2 > ocap) {
                    ocap *= 2;
                    char *no = (char *)realloc(out, ocap);
                    if (!no) {
                        free(scratch);
                        free(out);
                        return NULL;
                    }
                    out = no;
                }
                out[olen++] = (char)b;
            }
            len = 0;
            if (olen + sl + 1 > ocap) {
                while (olen + sl + 1 > ocap) ocap *= 2;
                char *no = (char *)realloc(out, ocap);
                if (!no) {
                    free(scratch);
                    free(out);
                    return NULL;
                }
                out = no;
            }
            memcpy(out + olen, s, sl);
            olen += sl;
        } else {
            if (len + sl + 1 > cap) {
                while (len + sl + 1 > cap) cap *= 2;
                char *ns = (char *)realloc(scratch, cap);
                if (!ns) {
                    free(scratch);
                    free(out);
                    return NULL;
                }
                scratch = ns;
            }
            memcpy(scratch + len, s, sl);
            len += sl;
        }
    }
    size_t p = 0;
    while (p < len) {
        unsigned cp;
        int w = utf8_decode_cp((const unsigned char *)scratch + p, len - p,
                               &cp);
        if (w < 0) {
            p++;
            continue;
        }
        p += (size_t)w;
        int b = (cp < 512) ? cp_to_byte[cp] : -1;
        if (b < 0) continue;
        if (olen + 2 > ocap) {
            ocap *= 2;
            char *no = (char *)realloc(out, ocap);
            if (!no) {
                free(scratch);
                free(out);
                return NULL;
            }
            out = no;
        }
        out[olen++] = (char)b;
    }
    free(scratch);
    out[olen] = '\0';
    return out;
}
