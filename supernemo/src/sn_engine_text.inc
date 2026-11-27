/* ---- protocol text escaping ---- */

static void emit_escaped(const char *verb, const char *text, size_t n) {
    fputs(verb, stdout);
    fputc(' ', stdout);
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)text[i];
        if (c == '\\') fputs("\\\\", stdout);
        else if (c == '\n') fputs("\\n", stdout);
        else if (c == '\t') fputs("\\t", stdout);
        else fputc(c, stdout);
    }
    fputc('\n', stdout);
    fflush(stdout);
}

static void unescape(char *s) {
    char *r = s, *w = s;
    while (*r) {
        if (*r == '\\' && r[1]) {
            r++;
            *w++ = (*r == 'n') ? '\n' : (*r == 't') ? '\t' : *r;
            r++;
        } else {
            *w++ = *r++;
        }
    }
    *w = 0;
}

/* ---- think/answer stream splitting ---- */

#define THINK_OPEN "<think>"
#define THINK_CLOSE "</think>"

typedef struct {
    char buf[8192]; /* decoded bytes not yet emitted */
    size_t len;
    int thinking;   /* current mode */
} splitter;

static size_t utf8_prefix(const char *s, size_t n) {
    /* Longest prefix of s[0..n) ending on a UTF-8 boundary. */
    if (n == 0) return 0;
    size_t i = n - 1;
    while (i > 0 && ((unsigned char)s[i] & 0xC0) == 0x80) i--;
    unsigned char lead = (unsigned char)s[i];
    size_t need = lead < 0x80 ? 1 : lead < 0xE0 ? 2 : lead < 0xF0 ? 3 : 4;
    return (i + need <= n) ? n : i;
}

static void split_flush(splitter *sp, int final) {
    for (;;) {
        const char *delim = sp->thinking ? THINK_CLOSE : THINK_OPEN;
        size_t dlen = strlen(delim);
        /* find the delimiter in the pending bytes */
        size_t at = (size_t)-1;
        if (sp->len >= dlen) {
            for (size_t i = 0; i + dlen <= sp->len; i++) {
                if (memcmp(sp->buf + i, delim, dlen) == 0) { at = i; break; }
            }
        }
        if (at != (size_t)-1) {
            if (at > 0)
                emit_escaped(sp->thinking ? "THINK" : "TOK", sp->buf, at);
            memmove(sp->buf, sp->buf + at + dlen, sp->len - at - dlen);
            sp->len -= at + dlen;
            sp->thinking = !sp->thinking;
            continue;
        }
        /* No delimiter: hold back a tail that could be a delimiter
         * prefix split across tokens, plus any partial UTF-8 tail. */
        size_t hold = 0;
        if (!final) {
            size_t maxk = sp->len < dlen - 1 ? sp->len : dlen - 1;
            for (size_t k = maxk; k > 0; k--) {
                if (memcmp(sp->buf + sp->len - k, delim, k) == 0) {
                    hold = k;
                    break;
                }
            }
        }
        size_t emit = utf8_prefix(sp->buf, sp->len - hold);
        if (emit == 0) return;
        emit_escaped(sp->thinking ? "THINK" : "TOK", sp->buf, emit);
        memmove(sp->buf, sp->buf + emit, sp->len - emit);
        sp->len -= emit;
        return;
    }
}

static void split_feed(splitter *sp, const char *bytes) {
    size_t n = strlen(bytes);
    /* Copy in chunks that always fit (Archie P1-3): a final flush
     * can leave up to 3 bytes of a split UTF-8 codepoint in the
     * buffer, so testing the fit once, before flushing, still
     * lets len + n overrun the buffer when n is within a few
     * bytes of its size. The fit is therefore re-tested after
     * every flush, and oversized input is fed through in pieces;
     * the splitter's delimiter hold-back keeps a delimiter split
     * across pieces intact. */
    while (n > 0) {
        if (sp->len + n >= sizeof sp->buf) split_flush(sp, 1);
        size_t room = sizeof sp->buf - sp->len;
        if (room == 0) {
            /* The buffer holds only bytes a final flush cannot
             * emit (a partial codepoint): emit them raw rather
             * than spin. */
            emit_escaped(sp->thinking ? "THINK" : "TOK", sp->buf,
                         sp->len);
            sp->len = 0;
            room = sizeof sp->buf;
        }
        size_t take = n < room ? n : room;
        memcpy(sp->buf + sp->len, bytes, take);
        sp->len += take;
        bytes += take;
        n -= take;
        split_flush(sp, 0);
    }
}

