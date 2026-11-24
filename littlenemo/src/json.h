/* json.h - Little Nemo minimal JSON reader, pure C.
 * A small DOM parser for the checkpoint's JSON side files
 * (tokenizer.json, tokenizer_config.json, config.json). Objects,
 * arrays, strings (full escape handling including \uXXXX and
 * surrogate pairs, decoded to UTF-8), numbers, booleans, null.
 * Hardened like the rest of the house: length-guarded input, a depth
 * cap, and trailing garbage is an error. Not a general JSON library;
 * it reads configuration, it does not write it.
 */
#ifndef NEMO_JSON_H
#define NEMO_JSON_H

#include <stddef.h>

typedef enum {
    NEMO_JSON_NULL,
    NEMO_JSON_BOOL,
    NEMO_JSON_NUM,
    NEMO_JSON_STR,
    NEMO_JSON_ARR,
    NEMO_JSON_OBJ
} nemo_json_type;

typedef struct nemo_json nemo_json;
struct nemo_json {
    nemo_json_type type;
    double num;          /* NUM (and BOOL as 0/1) */
    char *str;           /* STR: malloc'd UTF-8, NUL-terminated */
    nemo_json **items;   /* ARR items / OBJ values */
    char **keys;         /* OBJ keys (NULL for ARR) */
    int count;
};

/* Parse a complete document. Returns NULL on any malformed input. */
nemo_json *nemo_json_parse(const char *text, size_t len);
void nemo_json_free(nemo_json *j);

/* Object lookup by key (NULL when absent or not an object). */
const nemo_json *nemo_json_get(const nemo_json *obj, const char *key);

#endif
