/* tokenizer.c - see tokenizer.h. */
#define _FILE_OFFSET_BITS 64 /* 64-bit off_t on Windows too (Archie P0-2) */
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
        /* Clear the pointers on the way out: the loader's
         * common cleanup path runs smap_free on this map after
         * a failed init, and it must find nothing to free a
         * second time (Codex Astra review M-I7, 2026-11-25). */
        free(m->keys);
        free(m->vals);
        m->keys = NULL;
        m->vals = NULL;
        m->cap = 0;
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
    m->keys = NULL;
    m->vals = NULL;
    m->cap = 0;
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
    int used;
} pmap;

static int pmap_init(pmap *m, int cap) {
    m->keys = (uint64_t *)malloc((size_t)cap * sizeof(uint64_t));
    m->vals = (int *)malloc((size_t)cap * sizeof(int));
    m->taken = (unsigned char *)calloc((size_t)cap, 1);
    if (!m->keys || !m->vals || !m->taken) {
        /* Pointer clearing as in smap_init (M-I7). */
        free(m->keys);
        free(m->vals);
        free(m->taken);
        m->keys = NULL;
        m->vals = NULL;
        m->taken = NULL;
        m->cap = 0;
        return -1;
    }
    m->cap = cap;
    return 0;
}

static void pmap_free(pmap *m) {
    free(m->keys);
    free(m->vals);
    free(m->taken);
    m->keys = NULL;
    m->vals = NULL;
    m->taken = NULL;
    m->cap = 0;
}

static int pmap_put(pmap *m, int a, int b, int rank) {
    /* Load-factor guard, mirroring smap_put (Archie P1-2): with
     * no occupancy bound, a full table makes the probe loop
     * below spin forever on a hostile tokenizer.json. */
    if (m->used * 2 >= m->cap) return -1;
    uint64_t key = ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
    uint64_t h = (key * 0x9e3779b97f4a7c15ull) % (uint64_t)m->cap;
    while (m->taken[h]) h = (h + 1) % (uint64_t)m->cap;
    m->taken[h] = 1;
    m->keys[h] = key;
    m->vals[h] = rank;
    m->used++;
    return 0;
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
    /* Pre-tokenizer configuration from tokenizer.json (H2):
     * use_split is set when the file declares the Split
     * pre-tokenizer this engine implements (the Nemotron
     * checkpoint's pattern, matched exactly in the loader);
     * ignore_merges mirrors model.ignore_merges. Tokenizers
     * without the declaration keep the legacy whole-input
     * encoding the fixture suite is written against. */
    int use_split;
    int ignore_merges;
};

/* The one Split pattern this engine implements: the
 * Nemotron checkpoint's pre-tokenizer regex (Llama-style
 * word/number/punctuation splitting), matched verbatim
 * against the decoded JSON string. A tokenizer declaring a
 * different Split pattern is refused at load rather than
 * silently encoded with the wrong splits. */
static const char NEMO_SPLIT_PATTERN[] =
    "[^\\r\\n\\p{L}\\p{N}]?"
    "[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*"
    "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+"
    "|[^\\r\\n\\p{L}\\p{N}]?"
    "[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+"
    "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*"
    "|\\p{N}"
    "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*"
    "|\\s*[\\r\\n]+"
    "|\\s+(?!\\S)"
    "|\\s+";

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

/* A token id from the file is trusted only if it is a JSON
 * number, integral, and within [0, NEMO_MAX_TOKEN_ID]: a negative
 * or oversized id would index id_str out of bounds or force a
 * hostile allocation (Archie P1-2). */
#define NEMO_MAX_TOKEN_ID (1 << 22)
static int json_id(const nemo_json *j, int *out) {
    if (!j || j->type != NEMO_JSON_NUM) return -1;
    if (j->num < 0.0 || j->num > (double)NEMO_MAX_TOKEN_ID) return -1;
    if (j->num != (double)(int)j->num) return -1;
    *out = (int)j->num;
    return 0;
}

/* Smallest power-of-two table capacity that holds n entries at
 * the maps' half-load discipline. n is clamped first: a hostile
 * or malformed tokenizer.json could otherwise drive 2 * n into
 * signed overflow (Archie Round 2 P2-D). The 1 << 22 ceiling is
 * preserved; past it the maps reject cleanly at half load. */
static int map_cap_for(int n) {
    if (n <= 0) return 1 << 10;
    if (n > (1 << 21)) n = 1 << 21;
    int cap = 1 << 10;
    while (cap < 2 * n && cap < (1 << 22)) cap <<= 1;
    return cap;
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

    /* Pre-tokenizer configuration (H2). A declared Split step
     * must carry the one pattern this engine implements; any
     * other pattern fails the load rather than encoding with
     * the wrong splits. No Split step: legacy encoding. */
    {
        const nemo_json *pt = nemo_json_get(root, "pre_tokenizer");
        const nemo_json *im = nemo_json_get(model, "ignore_merges");
        if (im && im->type == NEMO_JSON_BOOL && im->num != 0.0)
            tk->ignore_merges = 1;
        if (pt && pt->type == NEMO_JSON_OBJ) {
            const nemo_json *seq = nemo_json_get(pt, "pretokenizers");
            int nsteps = 0;
            if (seq && seq->type == NEMO_JSON_ARR)
                nsteps = seq->count;
            for (int s = 0; s < nsteps; s++) {
                const nemo_json *step = seq->items[s];
                const nemo_json *sty = nemo_json_get(step, "type");
                if (!sty || sty->type != NEMO_JSON_STR
                    || strcmp(sty->str, "Split") != 0)
                    continue;
                const nemo_json *pat = nemo_json_get(step, "pattern");
                const nemo_json *rx = pat
                    ? nemo_json_get(pat, "Regex") : NULL;
                if (!rx || rx->type != NEMO_JSON_STR
                    || strcmp(rx->str, NEMO_SPLIT_PATTERN) != 0)
                    goto fail;
                tk->use_split = 1;
            }
        }
    }

    int max_id = -1;
    for (int i = 0; i < vocab->count; i++) {
        int id;
        if (json_id(vocab->items[i], &id) != 0) goto fail;
        if (id > max_id) max_id = id;
    }
    int n_added = (added && added->type == NEMO_JSON_ARR) ? added->count : 0;
    for (int i = 0; i < n_added; i++) {
        const nemo_json *idj = nemo_json_get(added->items[i], "id");
        int id;
        if (idj && json_id(idj, &id) == 0 && id > max_id) max_id = id;
    }
    if (max_id < 0) goto fail; /* a tokenizer with no vocab at all */
    tk->n_ids = max_id + 1;
    tk->id_str = (char **)calloc((size_t)tk->n_ids, sizeof(char *));
    tk->is_special = (char *)calloc((size_t)tk->n_ids, 1);
    if (!tk->id_str || !tk->is_special) goto fail;
    if (smap_init(&tk->vocab, map_cap_for(vocab->count + n_added)) != 0)
        goto fail;
    for (int i = 0; i < vocab->count; i++) {
        const char *s = vocab->keys[i];
        int id;
        if (json_id(vocab->items[i], &id) != 0) goto fail;
        free(tk->id_str[id]); /* duplicate id: last wins, no leak */
        tk->id_str[id] = (char *)malloc(strlen(s) + 1);
        if (!tk->id_str[id]) goto fail;
        strcpy(tk->id_str[id], s);
        if (smap_put(&tk->vocab, s, id) != 0) goto fail;
    }
    for (int i = 0; i < n_added; i++) {
        const nemo_json *a = added->items[i];
        const nemo_json *idj = nemo_json_get(a, "id");
        const nemo_json *content = nemo_json_get(a, "content");
        int id;
        if (!idj || json_id(idj, &id) != 0
            || !content || content->type != NEMO_JSON_STR) continue;
        if (!tk->id_str[id]) {
            tk->id_str[id] = (char *)malloc(strlen(content->str) + 1);
            if (!tk->id_str[id]) goto fail;
            strcpy(tk->id_str[id], content->str);
        }
        tk->is_special[id] = 1;
        if (smap_put(&tk->vocab, content->str, id) != 0) goto fail;
    }
    if (merges && merges->type == NEMO_JSON_ARR) {
        if (pmap_init(&tk->merges, map_cap_for(merges->count)) != 0)
            goto fail;
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
            int bad = ia >= 0 && ib >= 0
                      && pmap_put(&tk->merges, ia, ib, r) != 0;
            free(left);
            if (bad) goto fail;
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

/* Greedy lowest-rank merging over ids[0..n); returns the new
 * count, or -1 on error. Shared by the legacy whole-input path
 * and the per-piece split path: merges never cross a piece
 * boundary, because the checkpoint's Split is Isolated. */
static int bpe_merge(const nemo_tokenizer *tk, int *ids, int n) {
    for (;;) {
        int best_rank = -1, best_pos = -1;
        for (int i = 0; i + 1 < n; i++) {
            int r = pmap_get(&tk->merges, ids[i], ids[i + 1]);
            if (r >= 0 && (best_rank < 0 || r < best_rank)) {
                best_rank = r;
                best_pos = i;
            }
        }
        if (best_pos < 0) break;
        /* The merged token's string is the concatenation; it must be
         * in the vocab for a well-formed tokenizer file. */
        const char *sa = tk->id_str[ids[best_pos]];
        const char *sb = tk->id_str[ids[best_pos + 1]];
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
        const int pa = ids[best_pos], pb = ids[best_pos + 1];
        int w = 0;
        for (int i = 0; i < n; i++) {
            if (i + 1 < n && ids[i] == pa && ids[i + 1] == pb &&
                pmap_get(&tk->merges, ids[i], ids[i + 1]) ==
                    best_rank) {
                ids[w++] = merged;
                i++;
            } else {
                ids[w++] = ids[i];
            }
        }
        n = w;
    }
    return n;
}

/* --- pre-tokenizer split (H2) --------------------------------
 * A hand-written scanner for NEMO_SPLIT_PATTERN over decoded
 * code points, following the regex's alternation order and
 * greedy backtracking. Unicode classes come from range tables
 * over the major blocks: exact for ASCII, Latin, Greek,
 * Cyrillic, Hebrew, Arabic, the major Indic blocks, Thai,
 * Lao, Tibetan, Myanmar, Khmer, CJK, Hiragana, Katakana and
 * Hangul. Code points outside the tables fall to CC_OTHER
 * (the punctuation/symbol bucket), which is where the pattern
 * effectively puts unlisted scripts' separators; letters of
 * an unlisted script may split at case boundaries the tables
 * cannot see, and that limitation is accepted and stated
 * rather than hidden. */
enum {
    CC_OTHER = 0, CC_UPPER, CC_LOWER, CC_LETTER, CC_MARK,
    CC_NUMBER, CC_SPACE
};

static int in_rng(unsigned cp, unsigned lo, unsigned hi) {
    return cp >= lo && cp <= hi;
}

static int cp_class(unsigned cp) {
    if (cp == 0x20 || (cp >= 0x09 && cp <= 0x0d) || cp == 0x85
        || cp == 0xa0 || cp == 0x1680
        || (cp >= 0x2000 && cp <= 0x200a) || cp == 0x2028
        || cp == 0x2029 || cp == 0x202f || cp == 0x205f
        || cp == 0x3000)
        return CC_SPACE;
    if ((cp >= '0' && cp <= '9')
        || in_rng(cp, 0x660, 0x669) || in_rng(cp, 0x6f0, 0x6f9)
        || in_rng(cp, 0x7c0, 0x7c9) || in_rng(cp, 0x966, 0x96f)
        || in_rng(cp, 0x9e6, 0x9ef) || in_rng(cp, 0xa66, 0xa6f)
        || in_rng(cp, 0xae6, 0xaef) || in_rng(cp, 0xb66, 0xb6f)
        || in_rng(cp, 0xbe6, 0xbef) || in_rng(cp, 0xc66, 0xc6f)
        || in_rng(cp, 0xce6, 0xcef) || in_rng(cp, 0xd66, 0xd6f)
        || in_rng(cp, 0xde6, 0xdef) || in_rng(cp, 0xe50, 0xe59)
        || in_rng(cp, 0xed0, 0xed9) || in_rng(cp, 0xf20, 0xf29)
        || in_rng(cp, 0x1040, 0x1049) || in_rng(cp, 0x1090, 0x1099)
        || in_rng(cp, 0x17e0, 0x17e9) || in_rng(cp, 0x1810, 0x1819)
        || in_rng(cp, 0x1946, 0x194f) || in_rng(cp, 0x19d0, 0x19d9)
        || in_rng(cp, 0x1b50, 0x1b59) || in_rng(cp, 0x1c40, 0x1c49)
        || in_rng(cp, 0x1c50, 0x1c59) || in_rng(cp, 0xa620, 0xa629)
        || in_rng(cp, 0xa900, 0xa909) || in_rng(cp, 0xa9d0, 0xa9d9)
        || in_rng(cp, 0xaa50, 0xaa59) || in_rng(cp, 0xff10, 0xff19)
        || in_rng(cp, 0x2160, 0x2188))
        return CC_NUMBER;
    if (in_rng(cp, 0x300, 0x36f) || in_rng(cp, 0x483, 0x489)
        || in_rng(cp, 0x591, 0x5bd) || cp == 0x5bf
        || in_rng(cp, 0x5c1, 0x5c2) || in_rng(cp, 0x5c4, 0x5c7)
        || in_rng(cp, 0x610, 0x61a) || in_rng(cp, 0x64b, 0x65f)
        || cp == 0x670 || in_rng(cp, 0x6d6, 0x6dc)
        || in_rng(cp, 0x6df, 0x6e4) || in_rng(cp, 0x6e7, 0x6e8)
        || in_rng(cp, 0x6ea, 0x6ed) || in_rng(cp, 0x901, 0x903)
        || in_rng(cp, 0x93a, 0x954) || in_rng(cp, 0x981, 0x983)
        || in_rng(cp, 0x1dc0, 0x1dff) || in_rng(cp, 0x20d0, 0x20f0)
        || in_rng(cp, 0xfe00, 0xfe0f) || in_rng(cp, 0xfe20, 0xfe2f))
        return CC_MARK;
    if ((cp >= 'A' && cp <= 'Z')
        || (in_rng(cp, 0xc0, 0xde) && cp != 0xd7)
        || in_rng(cp, 0x391, 0x3a1) || in_rng(cp, 0x3a3, 0x3ab)
        || in_rng(cp, 0x400, 0x42f) || in_rng(cp, 0x531, 0x556)
        || in_rng(cp, 0x10a0, 0x10c5) || in_rng(cp, 0xff21, 0xff3a)
        || (in_rng(cp, 0x100, 0x131) && !(cp & 1)))
        return CC_UPPER;
    if ((cp >= 'a' && cp <= 'z')
        || (in_rng(cp, 0xdf, 0xff) && cp != 0xf7)
        || in_rng(cp, 0x3b1, 0x3c1) || in_rng(cp, 0x3c3, 0x3cb)
        || cp == 0x3c2
        || in_rng(cp, 0x430, 0x44f) || in_rng(cp, 0x561, 0x586)
        || in_rng(cp, 0xff41, 0xff5a)
        || (in_rng(cp, 0x100, 0x131) && (cp & 1)))
        return CC_LOWER;
    if (in_rng(cp, 0x132, 0x24f) || in_rng(cp, 0x250, 0x2ff)
        || in_rng(cp, 0x374, 0x3ff) || in_rng(cp, 0x450, 0x52f)
        || in_rng(cp, 0x5d0, 0x5ea) || in_rng(cp, 0x5f0, 0x5f2)
        || in_rng(cp, 0x620, 0x64a) || in_rng(cp, 0x66e, 0x6d3)
        || in_rng(cp, 0x710, 0x74a)
        || in_rng(cp, 0x900, 0x97f) || in_rng(cp, 0x980, 0x9ff)
        || in_rng(cp, 0xa00, 0xa7f) || in_rng(cp, 0xa80, 0xaff)
        || in_rng(cp, 0xb00, 0xb7f) || in_rng(cp, 0xb80, 0xbff)
        || in_rng(cp, 0xc00, 0xc7f) || in_rng(cp, 0xc80, 0xcff)
        || in_rng(cp, 0xd00, 0xd7f) || in_rng(cp, 0xd80, 0xdff)
        || in_rng(cp, 0xe00, 0xe7f) || in_rng(cp, 0xe80, 0xeff)
        || in_rng(cp, 0xf00, 0xfff) || in_rng(cp, 0x1000, 0x109f)
        || in_rng(cp, 0x10d0, 0x10ff) || in_rng(cp, 0x1100, 0x11ff)
        || in_rng(cp, 0x1200, 0x137f) || in_rng(cp, 0x1780, 0x17ff)
        || in_rng(cp, 0x1800, 0x18af)
        || in_rng(cp, 0x3040, 0x309f) || in_rng(cp, 0x30a0, 0x30ff)
        || in_rng(cp, 0x3100, 0x312f) || in_rng(cp, 0x3130, 0x318f)
        || in_rng(cp, 0x31f0, 0x31ff)
        || in_rng(cp, 0x3400, 0x4dbf) || in_rng(cp, 0x4e00, 0x9fff)
        || in_rng(cp, 0xac00, 0xd7af) || in_rng(cp, 0xf900, 0xfaff)
        || in_rng(cp, 0xff66, 0xff9d))
        return CC_LETTER;
    return CC_OTHER;
}

/* The pattern's letter classes: the "upper" run set is
 * Lu/Lt/Lm/Lo/M and the "lower" run set is Ll/Lm/Lo/M. Lm and
 * Lo are CC_LETTER here, Lt folds into CC_UPPER. */
static int cls_is_letter(int c) {
    return c == CC_UPPER || c == CC_LOWER || c == CC_LETTER;
}
static int cls_in_U(int c) {
    return c == CC_UPPER || c == CC_LETTER || c == CC_MARK;
}
static int cls_in_Lw(int c) {
    return c == CC_LOWER || c == CC_LETTER || c == CC_MARK;
}

/* Length in code points of the piece starting at i (>= 1),
 * following the pattern's alternation order. */
static int split_piece_len(const unsigned *cps, const int *cls,
                           int ncp, int i) {
    /* Alternatives A and B share the optional prefix: one code
     * point that is not a letter, not a number, and not CR/LF. */
    int pref = (!cls_is_letter(cls[i]) && cls[i] != CC_NUMBER
                && cps[i] != '\r' && cps[i] != '\n') ? 1 : 0;
    for (int use_pref = pref; use_pref >= 0; use_pref--) {
        int j = i + use_pref;
        int urun = 0;
        while (j + urun < ncp && cls_in_U(cls[j + urun])) urun++;
        /* A: U* Lw+ ... the longest U run that still leaves an
         * Lw code point for the mandatory tail. */
        for (int u = urun; u >= 0; u--) {
            int lw = 0;
            while (j + u + lw < ncp && cls_in_Lw(cls[j + u + lw]))
                lw++;
            if (lw >= 1) return use_pref + u + lw;
        }
        /* B: U+ Lw* */
        if (urun >= 1) {
            int lw = 0;
            while (j + urun + lw < ncp
                   && cls_in_Lw(cls[j + urun + lw]))
                lw++;
            return use_pref + urun + lw;
        }
    }
    /* C: a single number code point */
    if (cls[i] == CC_NUMBER) return 1;
    /* D: optional literal space, a run of other/mark code
     * points, then any trailing CR/LF/slash run */
    {
        int j = i;
        if (j < ncp && cps[j] == ' ') j++;
        int run = 0;
        while (j + run < ncp
               && (cls[j + run] == CC_OTHER
                   || cls[j + run] == CC_MARK))
            run++;
        if (run >= 1) {
            int t = 0;
            while (j + run + t < ncp
                   && (cps[j + run + t] == '\r'
                       || cps[j + run + t] == '\n'
                       || cps[j + run + t] == '/'))
                t++;
            return (j - i) + run + t;
        }
    }
    /* E, F, G work on the whitespace run */
    if (cls[i] == CC_SPACE) {
        int run = 0;
        while (i + run < ncp && cls[i + run] == CC_SPACE) run++;
        /* E: \s*[\r\n]+ ends just past the last CR/LF in the
         * run (the greedy \s* leaves exactly the trailing
         * newline sequence for the +). */
        for (int k = run - 1; k >= 0; k--)
            if (cps[i + k] == '\r' || cps[i + k] == '\n')
                return k + 1;
        /* F: \s+(?!\S) ... the whole run at end of input, else
         * the run minus its last code point (the lookahead
         * then faces a whitespace). */
        if (i + run >= ncp) return run;
        if (run >= 2) return run - 1;
        /* G: \s+ */
        return run;
    }
    /* Every class is covered above; never drop input. */
    return 1;
}

int nemo_tokenizer_encode(const nemo_tokenizer *tk, const char *utf8,
                          int *ids_out, int max_ids) {
    if (!tk || !utf8 || !ids_out || max_ids < 0) return -1;
    size_t nbytes = strlen(utf8);
    /* Independent work storage (Codex Astra review M-I4,
     * 2026-11-25): the byte sequence can be longer than the
     * caller's token budget because merging only shrinks it.
     * The old code used the caller's buffer for the byte
     * sequence and rejected any input whose BYTE count
     * exceeded max_ids before BPE ran; the final token count
     * is what must fit, and it is checked at the end. */
    int *work = (int *)malloc((nbytes + 1) * sizeof(int));
    if (!work) return -1;
    int n = 0;
    if (!tk->use_split) {
        for (size_t i = 0; i < nbytes; i++) {
            char sym[4];
            int w = utf8_encode_cp(sym, byte_to_cp[(unsigned char)utf8[i]]);
            sym[w] = '\0';
            int id = smap_get(&tk->vocab, sym);
            if (id < 0) { free(work); return -1; }
            work[n++] = id;
        }
        n = bpe_merge(tk, work, n);
        if (n < 0) { free(work); return -1; }
    } else {
        unsigned *cps = (unsigned *)malloc((nbytes + 1)
                                           * sizeof(unsigned));
        int *cls = (int *)malloc((nbytes + 1) * sizeof(int));
        size_t *coff = (size_t *)malloc((nbytes + 1)
                                        * sizeof(size_t));
        if (!cps || !cls || !coff) {
            free(cps); free(cls); free(coff); free(work);
            return -1;
        }
        int ncp = 0;
        for (size_t i = 0; i < nbytes; ) {
            unsigned cp;
            int w = utf8_decode_cp(
                (const unsigned char *)utf8 + i, nbytes - i, &cp);
            if (w < 0) { /* stray byte: its own code point */
                cp = (unsigned char)utf8[i];
                w = 1;
            }
            cps[ncp] = cp;
            cls[ncp] = cp_class(cp);
            coff[ncp] = i;
            ncp++;
            i += (size_t)w;
        }
        int ci = 0, failed = 0;
        while (ci < ncp && !failed) {
            int plen_cp = split_piece_len(cps, cls, ncp, ci);
            size_t b0 = coff[ci];
            size_t b1 = (ci + plen_cp < ncp)
                        ? coff[ci + plen_cp] : nbytes;
            /* Byte-level map the piece (H2). */
            char *mapped = (char *)malloc((b1 - b0) * 2 + 1);
            if (!mapped) { failed = 1; break; }
            size_t ml = 0;
            for (size_t k = b0; k < b1; k++)
                ml += (size_t)utf8_encode_cp(
                    mapped + ml,
                    byte_to_cp[(unsigned char)utf8[k]]);
            mapped[ml] = '\0';
            {
                /* With ignore_merges, a piece that is itself a
                 * vocabulary entry is emitted directly; any
                 * other piece goes through ordinary ranked BPE
                 * merging. That is the reference tokenizer's
                 * behavior, verified against it on a 70-case
                 * battery: a greedy longest-match segmentation
                 * (tried and reverted) diverges on repeated
                 * characters, where the reference emits the
                 * merge-derived pairs ("aaaaaaaaaa" -> five
                 * "aa" entries), not the longest entry. */
                int direct = tk->ignore_merges
                             ? smap_get(&tk->vocab, mapped) : -1;
                free(mapped);
                if (direct >= 0) {
                    work[n++] = direct;
                    ci += plen_cp;
                    continue;
                }
                int pn = 0;
                for (size_t k = b0; k < b1; k++) {
                    char sym[4];
                    int w = utf8_encode_cp(
                        sym, byte_to_cp[(unsigned char)utf8[k]]);
                    sym[w] = '\0';
                    int id = smap_get(&tk->vocab, sym);
                    if (id < 0) { failed = 1; break; }
                    work[n + pn++] = id;
                }
                if (failed) break;
                int merged = bpe_merge(tk, work + n, pn);
                if (merged < 0) { failed = 1; break; }
                n += merged;
            }
            ci += plen_cp;
        }
        free(cps); free(cls); free(coff);
        if (failed) { free(work); return -1; }
    }
    if (n > max_ids) { free(work); return -1; }
    memcpy(ids_out, work, (size_t)n * sizeof(int));
    free(work);
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
