/* test_sn_cmdloop.c - pending-command buffer in the SuperNemo engine loop.
 *
 * A long PROMPT grows the getline buffer. A short command is then adopted
 * from a fresh 128-byte allocation (as poll_command does during generation).
 * The next getline must not trust a stale capacity, or it will overflow the
 * heap. This test walks that sequence with the same free/reset as
 * supernemo/src/sn_engine.c (and littlenemo/src/engine.c).
 *
 * Found in Archie's round 7 review.
 * Pure C99, no locks.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  include <malloc.h>
#  define ALLOC_SIZE(p) _msize(p)
#else
#  include <malloc.h>
#  define ALLOC_SIZE(p) malloc_usable_size(p)
#endif

static int passes, fails;

static long eng_getline(char **lineptr, size_t *n, FILE *stream)
{
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

static long checked_getline(char **lineptr, size_t *n, FILE *stream)
{
    if (*lineptr) {
        size_t have = ALLOC_SIZE(*lineptr);
        if (*n > have) {
            printf("FAIL getline cap %zu is larger than the allocation %zu\n", *n, have);
            fails++;
            return -1;
        }
    }
    return eng_getline(lineptr, n, stream);
}

static FILE *write_script(void)
{
    char path[512];
#if defined(_WIN32)
    snprintf(path, sizeof path, "%s\\ntse-r7-cmdloop.txt", getenv("TEMP") ? getenv("TEMP") : ".");
#else
    snprintf(path, sizeof path, "/tmp/ntse-r7-cmdloop.txt");
#endif
    FILE *f = fopen(path, "wb");
    if (!f) return NULL;
    /* 1. A long PROMPT, long enough to grow cap past the 128-byte first alloc. */
    fputs("PROMPT ", f);
    for (int i = 0; i < 4000; i++) fputc('A', f);
    fputc('\n', f);
    /* 2. A short command that poll_command would queue during generation. */
    fputs("RESET\n", f);
    /* 3. Another long line: this is the getline that overflowed the heap. */
    fputs("PROMPT ", f);
    for (int i = 0; i < 2000; i++) fputc('B', f);
    fputc('\n', f);
    fputs("QUIT\n", f);
    fclose(f);
    return fopen(path, "rb");
}

int main(void)
{
    FILE *in = write_script();
    if (!in) { printf("FAIL could not write the command script\n"); return 1; }

    char *line = NULL; size_t cap = 0;
    char *pending_cmd = NULL;
    int adopted = 0, n_prompt = 0, n_reset = 0, saw_quit = 0;

    for (;;) {
        if (pending_cmd) {
            line = pending_cmd;
            pending_cmd = NULL;
            adopted++;
        } else {
            if (checked_getline(&line, &cap, in) < 0) break;
        }
        size_t L = strlen(line);
        while (L && (line[L - 1] == '\n' || line[L - 1] == '\r')) line[--L] = 0;
        if (!strncmp(line, "QUIT", 4)) { saw_quit = 1; break; }
        else if (!strncmp(line, "RESET", 5)) n_reset++;
        else if (!strncmp(line, "PROMPT ", 7)) {
            n_prompt++;
            /* Queue one short command during the first generation, matching
             * poll_command() in sn_engine.c (fresh buffer, cap = 0). */
            if (!pending_cmd && n_prompt == 1) {
                char *pl = NULL; size_t pc = 0;
                if (checked_getline(&pl, &pc, in) >= 0)
                    pending_cmd = pl;
            }
        }
        free(line);
        line = NULL;
        cap = 0;
    }
    free(line);
    free(pending_cmd);
    fclose(in);

    if (n_prompt == 2) { passes++; printf("PASS two PROMPT lines were processed\n"); }
    else { fails++; printf("FAIL expected 2 PROMPT lines, got %d\n", n_prompt); }
    if (n_reset == 1) { passes++; printf("PASS the short RESET was adopted between them\n"); }
    else { fails++; printf("FAIL expected 1 RESET, got %d\n", n_reset); }
    if (adopted == 1) { passes++; printf("PASS pending_cmd was adopted once with a fresh buffer\n"); }
    else { fails++; printf("FAIL pending_cmd adopted %d times\n", adopted); }
    if (saw_quit) { passes++; printf("PASS QUIT ended the loop after the second getline\n"); }
    else { fails++; printf("FAIL QUIT was not reached\n"); }

    printf("%d PASS / %d FAIL\n", passes, fails);
    return fails ? 1 : 0;
}
