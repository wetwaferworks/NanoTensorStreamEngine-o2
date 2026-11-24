/* json.c - see json.h. Recursive descent, depth-capped. */
#include "json.h"

#include <stdlib.h>
#include <string.h>

#define MAX_DEPTH 64

typedef struct {
    const char *p, *end;
    int depth;
} parser;

static void skip_ws(parser *ps) {
    while (ps->p < ps->end &&
           (*ps->p == ' ' || *ps->p == '\t' || *ps->p == '\n' ||
            *ps->p == '\r'))
        ps->p++;
}

static nemo_json *node_new(nemo_json_type t) {
    nemo_json *j = (nemo_json *)calloc(1, sizeof(nemo_json));
    if (j) j->type = t;
    return j;
}

void nemo_json_free(nemo_json *j) {
    if (!j) return;
    free(j->str);
    for (int i = 0; i < j->count; i++) {
        if (j->keys) free(j->keys[i]);
        nemo_json_free(j->items[i]);
    }
    free(j->keys);
    free(j->items);
    free(j);
}

const nemo_json *nemo_json_get(const nemo_json *obj, const char *key) {
    if (!obj || obj->type != NEMO_JSON_OBJ) return NULL;
    for (int i = 0; i < obj->count; i++)
        if (strcmp(obj->keys[i], key) == 0) return obj->items[i];
    return NULL;
}

static int hex4(parser *ps, unsigned *out) {
    if (ps->end - ps->p < 4) return -1;
    unsigned v = 0;
    for (int i = 0; i < 4; i++) {
        char c = ps->p[i];
        v <<= 4;
        if (c >= '0' && c <= '9') v |= (unsigned)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (unsigned)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (unsigned)(c - 'A' + 10);
        else return -1;
    }
    ps->p += 4;
    *out = v;
    return 0;
}

/* Append one code point as UTF-8. Returns bytes written or -1. */
static int utf8_put(char *dst, unsigned cp) {
    if (cp < 0x80) {
        dst[0] = (char)cp;
        return 1;
    }
    if (cp < 0x800) {
        dst[0] = (char)(0xc0 | (cp >> 6));
        dst[1] = (char)(0x80 | (cp & 0x3f));
        return 2;
    }
    if (cp < 0x10000) {
        dst[0] = (char)(0xe0 | (cp >> 12));
        dst[1] = (char)(0x80 | ((cp >> 6) & 0x3f));
        dst[2] = (char)(0x80 | (cp & 0x3f));
        return 3;
    }
    if (cp < 0x110000) {
        dst[0] = (char)(0xf0 | (cp >> 18));
        dst[1] = (char)(0x80 | ((cp >> 12) & 0x3f));
        dst[2] = (char)(0x80 | ((cp >> 6) & 0x3f));
        dst[3] = (char)(0x80 | (cp & 0x3f));
        return 4;
    }
    return -1;
}

/* Parse a string starting at the opening quote. The decoded output is
 * never longer than the raw span (escapes only shrink), so one pass
 * with a span-sized buffer is exact. */
static char *parse_string_raw(parser *ps) {
    if (ps->p >= ps->end || *ps->p != '"') return NULL;
    ps->p++;
    const char *start = ps->p;
    /* First pass: find the closing quote, honoring backslash escapes. */
    const char *q = start;
    while (q < ps->end && *q != '"') {
        if (*q == '\\') q++;
        q++;
    }
    if (q >= ps->end) return NULL;
    size_t span = (size_t)(q - start);
    char *out = (char *)malloc(span + 1);
    if (!out) return NULL;
    size_t o = 0;
    const char *p = start;
    int bad = 0;
    while (p < q && !bad) {
        unsigned char c = (unsigned char)*p;
        if (c != '\\') {
            out[o++] = (char)c;
            p++;
            continue;
        }
        p++;
        if (p >= q) { bad = 1; break; }
        char e = *p++;
        switch (e) {
            case '"': out[o++] = '"'; break;
            case '\\': out[o++] = '\\'; break;
            case '/': out[o++] = '/'; break;
            case 'b': out[o++] = '\b'; break;
            case 'f': out[o++] = '\f'; break;
            case 'n': out[o++] = '\n'; break;
            case 'r': out[o++] = '\r'; break;
            case 't': out[o++] = '\t'; break;
            case 'u': {
                parser tmp = {p, ps->end, 0};
                unsigned cp;
                if (hex4(&tmp, &cp) != 0) { bad = 1; break; }
                p = tmp.p;
                if (cp >= 0xd800 && cp <= 0xdbff) {
                    /* High surrogate: a low one must follow. */
                    if (ps->end - p >= 6 && p[0] == '\\' && p[1] == 'u') {
                        parser tmp2 = {p + 2, ps->end, 0};
                        unsigned lo;
                        if (hex4(&tmp2, &lo) == 0 && lo >= 0xdc00 &&
                            lo <= 0xdfff) {
                            cp = 0x10000 + ((cp - 0xd800) << 10) +
                                 (lo - 0xdc00);
                            p = tmp2.p;
                        } else {
                            bad = 1;
                            break;
                        }
                    } else {
                        bad = 1;
                        break;
                    }
                } else if (cp >= 0xdc00 && cp <= 0xdfff) {
                    bad = 1;
                    break;
                }
                int w = utf8_put(out + o, cp);
                if (w < 0) { bad = 1; break; }
                o += (size_t)w;
                break;
            }
            default: bad = 1; break;
        }
    }
    if (bad) {
        free(out);
        return NULL;
    }
    out[o] = '\0';
    ps->p = q + 1;
    return out;
}

static nemo_json *parse_value(parser *ps);

static int push_item(nemo_json *parent, char *key, nemo_json *child) {
    nemo_json **ni = (nemo_json **)realloc(
        parent->items, (size_t)(parent->count + 1) * sizeof(nemo_json *));
    if (!ni) return -1;
    parent->items = ni;
    if (parent->type == NEMO_JSON_OBJ) {
        char **nk = (char **)realloc(
            parent->keys, (size_t)(parent->count + 1) * sizeof(char *));
        if (!nk) return -1;
        parent->keys = nk;
        parent->keys[parent->count] = key;
    }
    parent->items[parent->count] = child;
    parent->count++;
    return 0;
}

static nemo_json *parse_value(parser *ps) {
    if (++ps->depth > MAX_DEPTH) return NULL;
    skip_ws(ps);
    nemo_json *j = NULL;
    if (ps->p >= ps->end) goto done;
    char c = *ps->p;
    if (c == '{') {
        ps->p++;
        j = node_new(NEMO_JSON_OBJ);
        if (!j) goto done;
        skip_ws(ps);
        if (ps->p < ps->end && *ps->p == '}') {
            ps->p++;
            goto done;
        }
        for (;;) {
            skip_ws(ps);
            char *key = parse_string_raw(ps);
            if (!key) { nemo_json_free(j); j = NULL; goto done; }
            skip_ws(ps);
            if (ps->p >= ps->end || *ps->p != ':') {
                free(key);
                nemo_json_free(j);
                j = NULL;
                goto done;
            }
            ps->p++;
            nemo_json *child = parse_value(ps);
            if (!child || push_item(j, key, child) != 0) {
                free(key);
                nemo_json_free(child);
                nemo_json_free(j);
                j = NULL;
                goto done;
            }
            skip_ws(ps);
            if (ps->p < ps->end && *ps->p == ',') {
                ps->p++;
                continue;
            }
            if (ps->p < ps->end && *ps->p == '}') {
                ps->p++;
                goto done;
            }
            nemo_json_free(j);
            j = NULL;
            goto done;
        }
    } else if (c == '[') {
        ps->p++;
        j = node_new(NEMO_JSON_ARR);
        if (!j) goto done;
        skip_ws(ps);
        if (ps->p < ps->end && *ps->p == ']') {
            ps->p++;
            goto done;
        }
        for (;;) {
            nemo_json *child = parse_value(ps);
            if (!child || push_item(j, NULL, child) != 0) {
                nemo_json_free(child);
                nemo_json_free(j);
                j = NULL;
                goto done;
            }
            skip_ws(ps);
            if (ps->p < ps->end && *ps->p == ',') {
                ps->p++;
                continue;
            }
            if (ps->p < ps->end && *ps->p == ']') {
                ps->p++;
                goto done;
            }
            nemo_json_free(j);
            j = NULL;
            goto done;
        }
    } else if (c == '"') {
        char *s = parse_string_raw(ps);
        if (!s) goto done;
        j = node_new(NEMO_JSON_STR);
        if (!j) {
            free(s);
            goto done;
        }
        j->str = s;
    } else if (c == 't') {
        if (ps->end - ps->p >= 4 && memcmp(ps->p, "true", 4) == 0) {
            ps->p += 4;
            j = node_new(NEMO_JSON_BOOL);
            if (j) j->num = 1.0;
        }
    } else if (c == 'f') {
        if (ps->end - ps->p >= 5 && memcmp(ps->p, "false", 5) == 0) {
            ps->p += 5;
            j = node_new(NEMO_JSON_BOOL);
        }
    } else if (c == 'n') {
        if (ps->end - ps->p >= 4 && memcmp(ps->p, "null", 4) == 0) {
            ps->p += 4;
            j = node_new(NEMO_JSON_NULL);
        }
    } else if (c == '-' || (c >= '0' && c <= '9')) {
        char *endp = NULL;
        double v = strtod(ps->p, &endp);
        if (endp && endp != ps->p) {
            ps->p = endp;
            j = node_new(NEMO_JSON_NUM);
            if (j) j->num = v;
        }
    }
done:
    ps->depth--;
    return j;
}

nemo_json *nemo_json_parse(const char *text, size_t len) {
    if (!text || len == 0) return NULL;
    parser ps = {text, text + len, 0};
    nemo_json *j = parse_value(&ps);
    if (!j) return NULL;
    skip_ws(&ps);
    if (ps.p != ps.end) {
        nemo_json_free(j);
        return NULL;
    }
    return j;
}
