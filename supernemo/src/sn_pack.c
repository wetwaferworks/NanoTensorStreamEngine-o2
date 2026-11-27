/* sn_pack.c - see sn_pack.h. Pure C99, no locks. */
#include "sn_pack.h"

#include <stdlib.h>
#include <string.h>

static int seek64(FILE *f, uint64_t off)
{
#ifdef _WIN32
    return _fseeki64(f, (long long)off, SEEK_SET);
#else
    return fseeko(f, (off_t)off, SEEK_SET);
#endif
}

float sn_f16_to_f32(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu, mant = h & 0x3ffu, bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {
            exp = 127 - 15 + 1;
            while (!(mant & 0x400u)) { mant <<= 1; exp--; }
            mant &= 0x3ffu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) bits = sign | 0x7f800000u | (mant << 13);
    else bits = sign | ((exp + 112) << 23) | (mant << 13);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static uint64_t name_hash(const char *s)
{
    uint64_t h = 14695981039346656037ull;
    for (; *s; s++) { h ^= (uint8_t)*s; h *= 1099511628211ull; }
    return h;
}

static int rd32(FILE *f, uint32_t *v) { return fread(v, 4, 1, f) == 1 ? 0 : -1; }
static int rd64(FILE *f, uint64_t *v) { return fread(v, 8, 1, f) == 1 ? 0 : -1; }

void sn_pack_close(sn_pack *p)
{
    if (!p) return;
    for (int i = 0; i < p->n; i++) free(p->t[i].name);
    free(p->t); free(p->ht); free(p->prefix);
    memset(p, 0, sizeof(*p));
}

const sn_tensor *sn_pack_find(const sn_pack *p, const char *name)
{
    if (!p->ht_cap) return NULL;
    uint64_t h = name_hash(name) % (uint64_t)p->ht_cap;
    for (int probe = 0; probe < p->ht_cap; probe++) {
        int i = p->ht[h];
        if (i < 0) return NULL;
        if (strcmp(p->t[i].name, name) == 0) return &p->t[i];
        h = (h + 1) % (uint64_t)p->ht_cap;
    }
    return NULL;
}

/* Sidecar: one line per tensor, "name<TAB>dtype<TAB>shape<TAB>count".
 * Header line starts with "name". Returns the count for a name via a
 * simple linear merge: the sidecar is sorted like the manifest is not,
 * so build a temporary table. */
typedef struct { char *name; uint64_t count; } shape_ent;

static int load_shapes(const char *path, shape_ent **out, int *nout)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "sn_pack: cannot open %s\n", path); return -1; }
    int cap = 1024, n = 0;
    shape_ent *e = (shape_ent *)malloc(sizeof(shape_ent) * cap);
    char line[2048];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "name\t", 5) == 0) continue;
        char *t1 = strchr(line, '\t'); if (!t1) continue;
        char *t2 = strchr(t1 + 1, '\t'); if (!t2) continue;
        char *t3 = strchr(t2 + 1, '\t'); if (!t3) continue;
        *t1 = 0;
        if (n == cap) { cap *= 2; e = (shape_ent *)realloc(e, sizeof(shape_ent) * cap); }
        e[n].name = (char *)malloc(strlen(line) + 1);
        strcpy(e[n].name, line);
        e[n].count = strtoull(t3 + 1, NULL, 10);
        n++;
    }
    fclose(f);
    *out = e; *nout = n;
    return 0;
}

int sn_pack_open(sn_pack *p, const char *prefix, const char *shapes_tsv)
{
    memset(p, 0, sizeof(*p));
    char path[1100];
    p->prefix = (char *)malloc(strlen(prefix) + 1);
    strcpy(p->prefix, prefix);

    snprintf(path, sizeof(path), "%s.nq2", prefix);
    FILE *fs = fopen(path, "rb");
    if (!fs) { fprintf(stderr, "sn_pack: cannot open %s\n", path); return -1; }
    char magic[4];
    uint32_t version, nseg, flags;
    uint64_t tot_weights, gen;
    if (fread(magic, 1, 4, fs) != 4 || memcmp(magic, "NQ2\0", 4) != 0
        || rd32(fs, &version) || version != 3
        || rd32(fs, &p->block) || rd32(fs, &p->entries)
        || rd64(fs, &tot_weights) || rd64(fs, &p->tot_blocks)
        || rd32(fs, &nseg)) {
        fprintf(stderr, "sn_pack: %s is not an NQ2 v3 stream\n", path);
        fclose(fs); return -1;
    }
    uint64_t *nb = (uint64_t *)malloc(sizeof(uint64_t) * (nseg ? nseg : 1));
    if (fread(nb, 8, nseg, fs) != nseg || rd64(fs, &gen) || rd32(fs, &flags)) {
        fprintf(stderr, "sn_pack: truncated stream header\n");
        free(nb); fclose(fs); return -1;
    }
    p->gen = gen;
    p->pq_d = 0;
    if (flags & 1u) { if (rd32(fs, &p->pq_d)) { free(nb); fclose(fs); return -1; } }
    if (flags & 2u) { fprintf(stderr, "sn_pack: per-tensor block sizes (--budgets) are not supported by SuperNemo\n"); free(nb); fclose(fs); return -1; }
    uint32_t n_raw = 0;
    uint32_t *raw_ix = NULL;
    if (flags & 4u) {
        if (rd32(fs, &n_raw)) { free(nb); fclose(fs); return -1; }
        raw_ix = (uint32_t *)malloc(sizeof(uint32_t) * (n_raw ? n_raw : 1));
        if (fread(raw_ix, 4, n_raw, fs) != n_raw) { free(nb); free(raw_ix); fclose(fs); return -1; }
    }
    p->body_off = (uint64_t)
#ifdef _WIN32
        _ftelli64(fs);
#else
        ftello(fs);
#endif
    fclose(fs);
    p->nsub = p->pq_d ? p->block / p->pq_d : 1;
    p->book_floats = (uint64_t)p->entries * p->block;

    /* book header: NQ2B version block entries nbooks gen [d] */
    snprintf(path, sizeof(path), "%s.nq2book", prefix);
    FILE *fb = fopen(path, "rb");
    if (!fb) { fprintf(stderr, "sn_pack: cannot open %s\n", path); free(nb); free(raw_ix); return -1; }
    uint32_t bver, bblk, bent, nbooks, bd = 0; uint64_t bgen;
    if (fread(magic, 1, 4, fb) != 4 || memcmp(magic, "NQ2B", 4) != 0
        || rd32(fb, &bver) || rd32(fb, &bblk) || rd32(fb, &bent)
        || rd32(fb, &nbooks) || rd64(fb, &bgen)
        || ((flags & 1u) && rd32(fb, &bd))) {
        fprintf(stderr, "sn_pack: bad book header in %s\n", path);
        fclose(fb); free(nb); free(raw_ix); return -1;
    }
    fclose(fb);
    if (bgen != gen || bblk != p->block || bent != p->entries || bd != p->pq_d) {
        fprintf(stderr, "sn_pack: stream (generation %016llx) and book (%016llx) do not belong together\n",
                (unsigned long long)gen, (unsigned long long)bgen);
        free(nb); free(raw_ix); return -1;
    }
    uint64_t book_hdr = 28 + (p->pq_d ? 4 : 0);

    /* manifest */
    snprintf(path, sizeof(path), "%s.nq2manifest", prefix);
    FILE *mf = fopen(path, "r");
    if (!mf) { fprintf(stderr, "sn_pack: cannot open %s\n", path); free(nb); free(raw_ix); return -1; }
    char line[2048];
    if (!fgets(line, sizeof(line), mf) || strncmp(line, "NQ2MANIFEST", 11) != 0) { fclose(mf); free(nb); free(raw_ix); return -1; }
    unsigned long long mgen = 0; int mn = 0;
    if (!fgets(line, sizeof(line), mf) || sscanf(line, "generation %llx", &mgen) != 1 || mgen != gen) {
        fprintf(stderr, "sn_pack: manifest generation does not match the stream\n"); fclose(mf); free(nb); free(raw_ix); return -1; }
    if (!fgets(line, sizeof(line), mf) || sscanf(line, "tensors %d", &mn) != 1 || (uint32_t)mn != nseg) {
        fprintf(stderr, "sn_pack: manifest tensor count differs from the stream\n"); fclose(mf); free(nb); free(raw_ix); return -1; }
    p->t = (sn_tensor *)calloc((size_t)mn, sizeof(sn_tensor));
    p->n = mn;
    shape_ent *se = NULL; int nse = 0;
    if (load_shapes(shapes_tsv, &se, &nse) != 0) { fclose(mf); free(nb); free(raw_ix); return -1; }
    int se_cap = 1; while (se_cap < nse * 2) se_cap <<= 1;
    int *seh = (int *)malloc(sizeof(int) * (size_t)se_cap);
    for (int i = 0; i < se_cap; i++) seh[i] = -1;
    for (int i = 0; i < nse; i++) {
        uint64_t h = name_hash(se[i].name) & (uint64_t)(se_cap - 1);
        while (seh[h] >= 0) h = (h + 1) & (uint64_t)(se_cap - 1);
        seh[h] = i;
    }
    uint64_t book_i = 0, raw_acc = 0, blk_acc = 0;
    int bad = 0;
    for (int i = 0; i < mn && !bad; i++) {
        char nm[1024], kind[16]; unsigned long long b0, nbl, d1, d2; unsigned bk;
        if (!fgets(line, sizeof(line), mf)
            || sscanf(line, "tensor %1023s %llu %llu %u %llx %llx %15s", nm, &b0, &nbl, &bk, &d1, &d2, kind) != 7) {
            fprintf(stderr, "sn_pack: bad manifest line %d\n", i); bad = 1; break; }
        sn_tensor *t = &p->t[i];
        t->name = (char *)malloc(strlen(nm) + 1); strcpy(t->name, nm);
        t->raw = strcmp(kind, "raw") == 0;
        t->block0 = b0; t->nblocks = nbl;
        if (b0 != blk_acc || nbl != nb[i]) { fprintf(stderr, "sn_pack: manifest block range of %s disagrees with the stream\n", nm); bad = 1; break; }
        blk_acc += nbl;
        /* count from the sidecar */
        uint64_t h = name_hash(nm) & (uint64_t)(se_cap - 1);
        int found = -1;
        while (seh[h] >= 0) { if (strcmp(se[seh[h]].name, nm) == 0) { found = seh[h]; break; } h = (h + 1) & (uint64_t)(se_cap - 1); }
        if (found < 0 && t->raw) { fprintf(stderr, "sn_pack: %s is raw but missing from the shapes sidecar\n", nm); bad = 1; break; }
        t->count = found >= 0 ? se[found].count : nbl * p->block;
        if (!t->raw) { t->book_off = book_hdr + book_i * p->book_floats * 4; book_i++; }
        else { t->raw_off = 0; /* set below */ }
    }
    fclose(mf);
    for (int i = 0; i < nse; i++) free(se[i].name);
    free(se); free(seh);
    if (bad) { free(nb); free(raw_ix); sn_pack_close(p); return -1; }
    if (book_i != nbooks) { fprintf(stderr, "sn_pack: book count %llu differs from the book header %u\n", (unsigned long long)book_i, nbooks); free(nb); free(raw_ix); sn_pack_close(p); return -1; }
    /* raw offsets follow the raw index list order */
    uint64_t raw_base = p->body_off + p->tot_blocks * ((uint64_t)p->nsub + 4);
    for (uint32_t k = 0; k < n_raw; k++) {
        sn_tensor *t = &p->t[raw_ix[k]];
        t->raw_off = raw_base + raw_acc * 2;
        raw_acc += t->count;
    }
    free(nb); free(raw_ix);

    p->ht_cap = 1; while (p->ht_cap < p->n * 2) p->ht_cap <<= 1;
    p->ht = (int *)malloc(sizeof(int) * (size_t)p->ht_cap);
    for (int i = 0; i < p->ht_cap; i++) p->ht[i] = -1;
    for (int i = 0; i < p->n; i++) {
        uint64_t h = name_hash(p->t[i].name) % (uint64_t)p->ht_cap;
        while (p->ht[h] >= 0) h = (h + 1) % (uint64_t)p->ht_cap;
        p->ht[h] = i;
    }
    return 0;
}

int sn_rd_open(const sn_pack *p, sn_rd *r)
{
    char path[1100];
    snprintf(path, sizeof(path), "%s.nq2", p->prefix);
    r->fs = fopen(path, "rb");
    snprintf(path, sizeof(path), "%s.nq2book", p->prefix);
    r->fb = fopen(path, "rb");
    if (!r->fs || !r->fb) { sn_rd_close(r); return -1; }
    return 0;
}

void sn_rd_close(sn_rd *r)
{
    if (r->fs) fclose(r->fs);
    if (r->fb) fclose(r->fb);
    r->fs = r->fb = NULL;
}

int sn_expand(const sn_pack *p, sn_rd *r, const sn_tensor *t,
              uint64_t first, uint64_t n, float *out)
{
    if (first + n > t->count) return -1;
    if (n == 0) return 0;
    if (t->raw) {
        uint16_t *tmp = (uint16_t *)malloc(n * 2);
        if (!tmp) return -1;
        if (seek64(r->fs, t->raw_off + first * 2) != 0 || fread(tmp, 2, n, r->fs) != n) { free(tmp); return -1; }
        for (uint64_t i = 0; i < n; i++) out[i] = sn_f16_to_f32(tmp[i]);
        free(tmp);
        return 0;
    }
    uint32_t B = p->block, nsub = p->nsub, d = p->pq_d ? p->pq_d : B, E = p->entries;
    uint64_t b_lo = first / B, b_hi = (first + n + B - 1) / B;   /* covering blocks */
    uint64_t nbk = b_hi - b_lo;
    if (b_hi > t->nblocks) return -1;
    uint64_t g0 = t->block0 + b_lo;
    uint8_t *idx = (uint8_t *)malloc(nbk * nsub);
    uint16_t *mn = (uint16_t *)malloc(nbk * 2), *sc = (uint16_t *)malloc(nbk * 2);
    float *book = (float *)malloc(p->book_floats * 4);
    float *blk = (float *)malloc((size_t)B * 4);
    if (!idx || !mn || !sc || !book || !blk) { free(idx); free(mn); free(sc); free(book); free(blk); return -1; }
    int rc = 0;
    if (seek64(r->fs, p->body_off + g0 * nsub) != 0 || fread(idx, nsub, nbk, r->fs) != nbk) rc = -1;
    if (!rc && (seek64(r->fs, p->body_off + p->tot_blocks * nsub + g0 * 2) != 0 || fread(mn, 2, nbk, r->fs) != nbk)) rc = -1;
    if (!rc && (seek64(r->fs, p->body_off + p->tot_blocks * (nsub + 2ull) + g0 * 2) != 0 || fread(sc, 2, nbk, r->fs) != nbk)) rc = -1;
    if (!rc && (seek64(r->fb, t->book_off) != 0 || fread(book, 4, p->book_floats, r->fb) != p->book_floats)) rc = -1;
    if (!rc) {
        for (uint64_t b = 0; b < nbk; b++) {
            float m = sn_f16_to_f32(mn[b]), s = sn_f16_to_f32(sc[b]);
            if (p->pq_d) {
                for (uint32_t sub = 0; sub < nsub; sub++) {
                    const float *c = book + ((uint64_t)sub * E + idx[b * nsub + sub]) * d;
                    for (uint32_t i = 0; i < d; i++) blk[sub * d + i] = m + s * c[i];
                }
            } else {
                const float *c = book + (uint64_t)idx[b * nsub] * B;
                for (uint32_t i = 0; i < B; i++) blk[i] = m + s * c[i];
            }
            uint64_t w0 = (b_lo + b) * B;           /* weight index of blk[0] */
            uint64_t lo = w0 < first ? first : w0;
            uint64_t hi = w0 + B > first + n ? first + n : w0 + B;
            for (uint64_t w = lo; w < hi; w++) out[w - first] = blk[w - w0];
        }
    }
    free(idx); free(mn); free(sc); free(book); free(blk);
    return rc;
}
