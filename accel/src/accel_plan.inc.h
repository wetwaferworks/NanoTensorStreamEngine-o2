/* accel_plan.inc.h - prepared stream plan (Lane 2, Gizmo's Tensor
 * Handshake GPU concurrency lane). Included at the end of
 * accel_core.c: it shares the core's validation, budget, fault
 * policy and handshake primitives. Co-implemented by Cortana; NTSE
 * and the nq2 format are Elfie's; lane proposal Gizmo; direction JD.
 *
 * Why it exists: the generic stream (ntse_accel_stream) re-lays out
 * every tensor on the CPU each pass and re-sends the shared book
 * with every tensor; on the full Sigma pack that was 169 MB/token
 * and 8.5 tok/s against 82 resident. A plan does the layout once,
 * keeps each distinct book resident, and moves only what changes
 * hands per token.
 *
 * Hazards and who guards them (one driver thread issues every API
 * call; there is no lock anywhere):
 *   pinned image arena  read-only after create; DMA source only
 *   slot s device buf   written by COPY(i), read by COMPUTE(i);
 *                       COPY(i+S) is enqueued only after the host
 *                       observed done[s] of item i (WAR safe)
 *   slot s pinned buf   x written by the host before COPY(i), y
 *                       written by COMPUTE(i); host reads y and
 *                       rewrites x only after observing done[s]
 *   book buffers        uploaded at create, read-only afterwards
 *   COMPUTE(i) after COPY(i): device-side ev_wait on h2d[s]
 */

#define PLAN_MAX_BOOKS 64u

typedef struct {
    const ntse_dtc_tensor *t;   /* caller's (CPU fallback)           */
    uint64_t rows, cols, nseg, nblk, nact;
    uint32_t block, entries, book_id;
    uint64_t img_off, img_bytes; /* in the host image                */
    uint64_t o_mnT, o_scT, o_act; /* relative to the image start     */
} plan_tensor;

struct ntse_plan {
    ntse_plan_config cfg;
    uint32_t n;
    plan_tensor *pt;
    int on_device, pinned;
    uint8_t *img;  void *img_tok;      /* host image (pinned or malloc) */
    uint64_t img_bytes;
    uint32_t nbooks;
    ntse_dmem book[PLAN_MAX_BOOKS];    /* [bookT | book] per distinct   */
    uint64_t book_o_book[PLAN_MAX_BOOKS];
    const float *book_src[PLAN_MAX_BOOKS];
    uint32_t book_B[PLAN_MAX_BOOKS], book_K[PLAN_MAX_BOOKS];
    uint64_t book_bytes;
    uint32_t S;
    ntse_dmem dev[NTSE_MAX_SLOTS];
    ntse_dmem flg[NTSE_MAX_SLOTS];     /* lazy flags, own buffer each */
    uint64_t max_Fb;
    uint8_t *pin[NTSE_MAX_SLOTS]; void *tok[NTSE_MAX_SLOTS];
    uint64_t max_img, max_xb, max_yb, max_Ab, max_Tb;
    uint64_t d_x, d_y, d_A, d_T, dev_slot;   /* device slot offsets   */
    uint64_t p_img, p_x, p_y, pin_slot;      /* pinned slot offsets   */
    uint64_t pinned_total;
    double create_ms;
};

void ntse_plan_config_default(ntse_plan_config *c)
{
    c->slots = 2;
    c->op = NTSE_OP_DTC_EAGER;
    c->max_batch = 1;
    c->pin_image = 1;
    c->pin_budget = 0;
}

static uint64_t plan_pin_budget(const ntse_plan_config *c)
{
    uint64_t b = c->pin_budget;
    if (!b) b = env_mb("NTSE_ACCEL_PIN_MB");
    if (!b) b = 1ull << 30;
    if (b > (4ull << 30)) {
        fprintf(stderr, "ntse-accel: pinned budget %llu MiB clamped to 4096 MiB\n",
                (unsigned long long)(b >> 20));
        b = 4ull << 30;
    }
    return b;
}

/* transpose into [nseg][rows] and list the active codes, once */
static uint64_t plan_fill_image(const ntse_dtc_tensor *t, const plan_tensor *q,
                                uint8_t *dst)
{
    uint8_t *idxT = dst;
    uint16_t *mnT = (uint16_t *)(dst + q->o_mnT);
    uint16_t *scT = (uint16_t *)(dst + q->o_scT);
    for (uint64_t r = 0; r < t->rows; r++)
        for (uint64_t j = 0; j < q->nseg; j++) {
            uint64_t s = r * q->nseg + j, d = j * t->rows + r;
            idxT[d] = t->idx[s];
            mnT[d] = t->mn_h[s];
            scT[d] = t->sc_h[s];
        }
    uint64_t nact = 0;
    if (act_encodable(t)) {
        uint32_t *act = (uint32_t *)(dst + q->o_act);
        for (uint64_t j = 0; j < q->nseg; j++) {
            uint8_t used[NTSE_MAX_ENTRIES];
            memset(used, 0, sizeof used);
            const uint8_t *col = idxT + j * t->rows;
            for (uint64_t r = 0; r < t->rows; r++) used[col[r]] = 1;
            for (uint32_t k = 0; k < t->entries; k++)
                if (used[k]) act[nact++] = (uint32_t)(j * t->entries + k);
        }
    }
    return nact;
}

static void plan_release_device(ntse_accel *a, ntse_plan *p)
{
    if (!a->be) return;
    for (uint32_t s = 0; s < NTSE_MAX_SLOTS; s++) {
        if (p->pin[s]) a->ops->hfree(a->be, p->pin[s], p->tok[s]);
        p->pin[s] = NULL; p->tok[s] = NULL;
        charge_free(a, &p->dev[s]);
        charge_free(a, &p->flg[s]);
    }
    for (uint32_t b = 0; b < p->nbooks; b++) charge_free(a, &p->book[b]);
    if (p->img && p->pinned) {
        a->ops->hfree(a->be, p->img, p->img_tok);
        p->img = NULL; p->img_tok = NULL; p->pinned = 0;
    }
    p->on_device = 0;
}

/* everything device-side; on any refusal the plan stays host-only */
static int plan_to_device(ntse_accel *a, ntse_plan *p)
{
    const ntse_be_ops *o = a->ops;
    uint64_t pin_budget = plan_pin_budget(&p->cfg);
    int rc = NTSE_OK;
    /* books: deduplicated by geometry and content, resident once */
    for (uint32_t i = 0; i < p->n && rc == NTSE_OK; i++) {
        const ntse_dtc_tensor *t = p->pt[i].t;
        uint32_t b = 0;
        for (; b < p->nbooks; b++)
            if (p->book_B[b] == t->block && p->book_K[b] == t->entries
                && (p->book_src[b] == t->book
                    || !memcmp(p->book_src[b], t->book, (size_t)t->block * t->entries * 4)))
                break;
        if (b == p->nbooks) {
            if (p->nbooks == PLAN_MAX_BOOKS) { rc = NTSE_E_UNSUPPORTED; break; }
            uint64_t bb = (uint64_t)t->entries * t->block * 4;
            uint64_t ob = al(bb), tot = al(ob + bb);
            uint8_t *h = (uint8_t *)malloc((size_t)tot);
            if (!h) { rc = NTSE_E_NOMEM; break; }
            float *bookT = (float *)h;
            for (uint32_t k = 0; k < t->entries; k++)
                for (uint32_t e = 0; e < t->block; e++)
                    bookT[(size_t)e * t->entries + k] = t->book[(size_t)k * t->block + e];
            memcpy(h + ob, t->book, (size_t)bb);
            rc = charge_alloc(a, tot, &p->book[b]);
            if (rc == NTSE_OK) rc = o->put(a->be, NTSE_Q_COPY, &p->book[b], 0, h, tot, NULL);
            free(h);
            if (rc != NTSE_OK) break;
            p->book_o_book[b] = ob;
            p->book_src[b] = t->book;
            p->book_B[b] = t->block; p->book_K[b] = t->entries;
            p->book_bytes += tot;
            p->nbooks++;
        }
        p->pt[i].book_id = b;
    }
    if (rc != NTSE_OK) return rc;
    /* the image: pinned arena when it fits, else stays pageable */
    if (p->cfg.pin_image && p->img_bytes <= pin_budget) {
        uint8_t *h = NULL; void *tok = NULL;
        if (o->halloc(a->be, p->img_bytes, (void **)&h, &tok) == NTSE_OK && h) {
            memcpy(h, p->img, (size_t)p->img_bytes);
            free(p->img);
            p->img = h; p->img_tok = tok; p->pinned = 1;
            p->pinned_total += p->img_bytes;
        } else if (a->cfg.verbose) {
            fprintf(stderr, "ntse-accel: plan: pinned image refused by the driver; "
                    "staging through pinned slots\n");
        }
    }
    /* slots */
    p->d_x = al(p->max_img);
    p->d_y = al(p->d_x + p->max_xb);
    p->d_A = al(p->d_y + p->max_yb);
    p->d_T = al(p->d_A + p->max_Ab);
    p->dev_slot = al(p->d_T + p->max_Tb);
    p->p_img = 0;
    p->p_x = p->pinned ? 0 : al(p->max_img);
    p->p_y = al(p->p_x + p->max_xb);
    p->pin_slot = al(p->p_y + p->max_yb);
    /* pinned memory is bounded as a whole: image arena plus slots */
    if (p->pinned_total + (uint64_t)p->S * p->pin_slot > pin_budget) {
        if (a->cfg.verbose)
            fprintf(stderr, "ntse-accel: plan: pinned host budget %llu B too small for "
                    "%u slots of %llu B\n", (unsigned long long)pin_budget, p->S,
                    (unsigned long long)p->pin_slot);
        return NTSE_E_BUDGET;
    }
    for (uint32_t s = 0; s < p->S && rc == NTSE_OK; s++) {
        rc = charge_alloc(a, p->dev_slot, &p->dev[s]);
        if (rc == NTSE_OK && p->cfg.op == NTSE_OP_DTC_LAZY)
            rc = charge_alloc(a, p->max_Fb, &p->flg[s]);
        if (rc == NTSE_OK)
            rc = o->halloc(a->be, p->pin_slot, (void **)&p->pin[s], &p->tok[s]);
        if (rc == NTSE_OK) p->pinned_total += p->pin_slot;
    }
    if (rc == NTSE_OK) {
        int r1 = o->qsync(a->be, NTSE_Q_COPY);
        if (r1 != NTSE_OK) rc = r1;
    }
    return rc;
}

int ntse_accel_plan_create(ntse_accel *a, const ntse_dtc_tensor *const *ts,
                           uint32_t n, const ntse_plan_config *cfg, ntse_plan **out)
{
    if (out) *out = NULL;
    if (!a || !ts || !cfg || !out || n == 0 || n > (1u << 20)
        || cfg->slots == 0 || cfg->slots > NTSE_MAX_SLOTS
        || cfg->max_batch == 0 || cfg->max_batch > NTSE_MAX_BATCH
        || (cfg->op != NTSE_OP_DTC_EAGER && cfg->op != NTSE_OP_DTC_ACTIVE
            && cfg->op != NTSE_OP_DTC_DIRECT && cfg->op != NTSE_OP_DTC_LAZY))
        return NTSE_E_INVAL;
    double t0 = now_ms();
    char why[200];
    for (uint32_t i = 0; i < n; i++) {
        int rc = ntse_dtc_validate(ts[i], why, sizeof why);
        if (rc != NTSE_OK) {
            if (a->cfg.verbose)
                fprintf(stderr, "ntse-accel: plan tensor %u refused: %s\n", i, why);
            return rc;
        }
        if (cfg->op == NTSE_OP_DTC_ACTIVE && !act_encodable(ts[i]))
            return NTSE_E_UNSUPPORTED;
    }
    ntse_plan *p = (ntse_plan *)calloc(1, sizeof *p);
    if (!p) return NTSE_E_NOMEM;
    p->cfg = *cfg; p->n = n; p->S = cfg->slots;
    p->pt = (plan_tensor *)calloc(n, sizeof *p->pt);
    if (!p->pt) { free(p); return NTSE_E_NOMEM; }
    uint64_t B = cfg->max_batch, off = 0;
    for (uint32_t i = 0; i < n; i++) {
        const ntse_dtc_tensor *t = ts[i];
        plan_tensor *q = &p->pt[i];
        q->t = t; q->rows = t->rows; q->cols = t->cols;
        q->block = t->block; q->entries = t->entries;
        q->nseg = t->cols / t->block; q->nblk = q->rows * q->nseg;
        uint64_t per = t->entries < t->rows ? t->entries : t->rows;
        uint64_t act_cap = act_encodable(t) ? q->nseg * per : 0;
        q->o_mnT = al(q->nblk);
        q->o_scT = al(q->o_mnT + q->nblk * 2);
        q->o_act = al(q->o_scT + q->nblk * 2);
        q->img_off = off;
        q->img_bytes = al(q->o_act + act_cap * 4 + 4);  /* trimmed after fill */
        off = al(off + q->img_bytes);
    }
    if (!on_gpu(a)) {           /* host-only from the start: no image */
        p->create_ms = now_ms() - t0;
        *out = p;
        return NTSE_OK;
    }
    p->img = (uint8_t *)malloc((size_t)(off ? off : 1));
    if (!p->img) { free(p->pt); free(p); return NTSE_E_NOMEM; }
    /* fill and trim: only o_act + nact codes are sent per token */
    uint64_t packed = 0;
    for (uint32_t i = 0; i < n; i++) {
        plan_tensor *q = &p->pt[i];
        uint8_t *dst = p->img + packed;
        /* filled in place at the packed offset (never past the slack) */
        q->nact = plan_fill_image(ts[i], q, dst);
        q->img_off = packed;
        q->img_bytes = al(q->o_act + q->nact * 4 + 4);
        packed = al(packed + q->img_bytes);
        uint64_t xb = B * q->cols * 4, yb = B * q->rows * 4, Ab = B * q->nseg * 4;
        uint64_t Tb = (cfg->op == NTSE_OP_DTC_DIRECT) ? 0 : B * q->nseg * q->entries * 4;
        if (q->img_bytes > p->max_img) p->max_img = q->img_bytes;
        if (xb > p->max_xb) p->max_xb = xb;
        if (yb > p->max_yb) p->max_yb = yb;
        if (Ab > p->max_Ab) p->max_Ab = Ab;
        if (Tb > p->max_Tb) p->max_Tb = Tb;
        uint64_t Fb = al(q->nseg * q->entries * 4);
        if (Fb > p->max_Fb) p->max_Fb = Fb;
    }
    p->img_bytes = packed ? packed : 256;
    int rc = plan_to_device(a, p);
    if (rc == NTSE_OK) {
        p->on_device = 1;
    } else {
        {
            fault(a, rc, "plan create");
            plan_release_device(a, p);
            if (a->cfg.verbose)
                fprintf(stderr, "ntse-accel: plan is host-only (%s; device or pinned "
                        "budget, see above): runs on the CPU reference\n",
                        ntse_accel_strerror(rc));
        }
        p->nbooks = 0; p->book_bytes = 0; p->pinned_total = 0;
        free(p->img); p->img = NULL;   /* the CPU path reads the caller's arrays */
        if (rc == NTSE_E_DEVICE && !a->cfg.allow_fallback) {
            free(p->pt); free(p);
            return rc;
        }
    }
    p->create_ms = now_ms() - t0;
    *out = p;
    return NTSE_OK;
}

void ntse_accel_plan_info_get(const ntse_plan *p, ntse_plan_info *info)
{
    memset(info, 0, sizeof *info);
    if (!p) return;
    info->tensors = p->n; info->books = p->nbooks;
    info->on_device = p->on_device; info->pinned_image = p->pinned;
    info->image_bytes = p->on_device ? p->img_bytes : 0;
    info->book_bytes = p->book_bytes;
    info->slot_dev_bytes = p->on_device ? p->dev_slot : 0;
    info->slot_pin_bytes = p->on_device ? p->pin_slot : 0;
    info->pinned_bytes = p->pinned_total;
    info->create_ms = p->create_ms;
}

void ntse_accel_plan_free(ntse_accel *a, ntse_plan *p)
{
    if (!p) return;
    if (a && p->on_device) {
        /* nothing is released while a queue may still own it */
        a->ops->qsync(a->be, NTSE_Q_COPY);
        a->ops->qsync(a->be, NTSE_Q_COMPUTE);
        plan_release_device(a, p);
    }
    if (p->img && !p->pinned) free(p->img);
    free(p->pt);
    free(p);
}

static int plan_cpu(const ntse_plan *p, const float *const *x, float *const *y,
                    uint32_t batch)
{
    for (uint32_t i = 0; i < p->n; i++) {
        int rc = ntse_cpu_matvec(p->pt[i].t, p->cfg.op, x[i], batch, y[i]);
        if (rc != NTSE_OK) return rc;
    }
    return NTSE_OK;
}

typedef struct {
    int state[NTSE_MAX_SLOTS];
    uint32_t item[NTSE_MAX_SLOTS], observed[NTSE_MAX_SLOTS];
    uint64_t violations;
} plan_own;

static int plan_release_slot(plan_own *w, uint32_t s, uint32_t i)
{
    if (HS_LD(&w->observed[s]) != i + 1 || HS_LD(&w->state[s]) != SLOT_INFLIGHT) {
        w->violations++;
        return NTSE_E_STATE;
    }
    HS_ST(&w->state[s], SLOT_FREE);
    return NTSE_OK;
}

static int plan_enqueue(ntse_accel *a, ntse_plan *p, plan_own *w, uint32_t s,
                        uint32_t i, const float *x, uint32_t batch, uint64_t *h2d)
{
    const ntse_be_ops *o = a->ops;
    const plan_tensor *q = &p->pt[i];
    /* stage: FREE -> STAGING (CAS), write x (and the image when it is
     * pageable) into this slot's pinned buffer, publish STAGED */
    if (!hs_cas_int(&w->state[s], SLOT_FREE, SLOT_STAGING)) { w->violations++; return NTSE_E_STATE; }
    uint64_t xb = (uint64_t)batch * q->cols * 4;
    const uint8_t *img_src = p->img + q->img_off;
    if (!p->pinned) {
        memcpy(p->pin[s] + p->p_img, img_src, (size_t)q->img_bytes);
        img_src = p->pin[s] + p->p_img;
    }
    memcpy(p->pin[s] + p->p_x, x, (size_t)xb);
    w->item[s] = i;
    HS_STX(&w->observed[s], 0u);
    HS_ST(&w->state[s], SLOT_STAGED);
    /* hand to the device: STAGED -> INFLIGHT (CAS) */
    if (!hs_cas_int(&w->state[s], SLOT_STAGED, SLOT_INFLIGHT) || w->item[s] != i) {
        w->violations++;
        return NTSE_E_STATE;
    }
    void *itok = p->pinned ? p->img_tok : p->tok[s];
    int rc = o->put(a->be, NTSE_Q_COPY, &p->dev[s], 0, img_src, q->img_bytes, itok);
    if (rc == NTSE_OK)
        rc = o->put(a->be, NTSE_Q_COPY, &p->dev[s], p->d_x, p->pin[s] + p->p_x, xb, p->tok[s]);
    if (rc == NTSE_OK) rc = o->ev_record(a->be, NTSE_Q_COPY, (int)s);
    if (rc == NTSE_OK) rc = o->ev_wait(a->be, NTSE_Q_COMPUTE, (int)s);
    if (rc != NTSE_OK) return rc;
    *h2d += q->img_bytes + xb;
    ntse_kargs k;
    memset(&k, 0, sizeof k);
    const ntse_dmem *bm = &p->book[q->book_id];
    k.bookT.m = bm; k.bookT.off = 0;
    k.book.m = bm;  k.book.off = p->book_o_book[q->book_id];
    k.idxT.m = &p->dev[s]; k.idxT.off = 0;
    k.mnT.m = &p->dev[s];  k.mnT.off = q->o_mnT;
    k.scT.m = &p->dev[s];  k.scT.off = q->o_scT;
    k.act.m = &p->dev[s];  k.act.off = q->o_act;
    k.x.m = &p->dev[s]; k.x.off = p->d_x;
    k.y.m = &p->dev[s]; k.y.off = p->d_y;
    k.A.m = &p->dev[s]; k.A.off = p->d_A;
    k.T.m = &p->dev[s]; k.T.off = p->d_T;
    if (p->cfg.op == NTSE_OP_DTC_LAZY) { k.F.m = &p->flg[s]; k.F.off = 0; }
    k.rows = q->rows; k.cols = q->cols; k.nseg = q->nseg; k.nact = q->nact;
    k.block = q->block; k.entries = q->entries; k.batch = batch;
    rc = enqueue_op(a, NTSE_Q_COMPUTE, p->cfg.op, &k);
    if (rc != NTSE_OK) return rc;
    uint64_t yb = (uint64_t)batch * q->rows * 4;
    rc = o->get(a->be, NTSE_Q_COMPUTE, p->pin[s] + p->p_y, &p->dev[s], p->d_y, yb, p->tok[s]);
    if (rc == NTSE_OK) rc = o->ev_record(a->be, NTSE_Q_COMPUTE, NTSE_MAX_SLOTS + (int)s);
    return rc;
}

static int plan_retire(ntse_accel *a, ntse_plan *p, plan_own *w, uint32_t i,
                       float *y, uint32_t batch, uint64_t *d2h)
{
    uint32_t s = i % p->S;
    if (a->force_early_recycle && i == 0) {
        /* test hook: recycle before observing completion; must be refused */
        if (plan_release_slot(w, s, i) == NTSE_OK) return NTSE_E_STATE;
        a->force_early_recycle = 0;
    }
    int rc = a->ops->ev_sync(a->be, NTSE_MAX_SLOTS + (int)s);
    if (rc != NTSE_OK) return rc;
    if (a->ops->ev_query(a->be, NTSE_MAX_SLOTS + (int)s) != 1
        || a->ops->ev_query(a->be, (int)s) != 1 || w->item[s] != i) {
        w->violations++;
        return NTSE_E_STATE;
    }
    HS_ST(&w->observed[s], i + 1);
    uint64_t yb = (uint64_t)batch * p->pt[i].rows * 4;
    memcpy(y, p->pin[s] + p->p_y, (size_t)yb);   /* commit, in order */
    *d2h += yb;
    return plan_release_slot(w, s, i);
}

int ntse_accel_plan_run(ntse_accel *a, ntse_plan *p, const float *const *x,
                        float *const *y, uint32_t batch, ntse_stream_stats *st)
{
    ntse_stream_stats local;
    if (!st) st = &local;
    memset(st, 0, sizeof *st);
    st->lane = NTSE_LANE_CPU;
    if (!a || !p || !x || !y || batch == 0 || batch > p->cfg.max_batch)
        return NTSE_E_INVAL;
    for (uint32_t i = 0; i < p->n; i++) if (!x[i] || !y[i]) return NTSE_E_INVAL;
    st->items = p->n;
    st->slot_bytes = p->pin_slot;
    double t0 = now_ms();
    if (!p->on_device || !on_gpu(a)) {
        int rc = plan_cpu(p, x, y, batch);
        st->wall_ms = now_ms() - t0;
        return rc;
    }
    plan_own w;
    memset(&w, 0, sizeof w);
    uint64_t h2d = 0, d2h = 0;
    int rc = NTSE_OK;
    uint32_t retired = 0;
    for (uint32_t i = 0; i < p->n && rc == NTSE_OK; i++) {
        if (i >= p->S) {   /* the slot this item needs is item i-S's */
            rc = plan_retire(a, p, &w, retired, y[retired], batch, &d2h);
            if (rc != NTSE_OK) break;
            retired++;
        }
        rc = plan_enqueue(a, p, &w, i % p->S, i, x[i], batch, &h2d);
    }
    while (rc == NTSE_OK && retired < p->n) {
        rc = plan_retire(a, p, &w, retired, y[retired], batch, &d2h);
        if (rc == NTSE_OK) retired++;
    }
    int r1 = a->ops->qsync(a->be, NTSE_Q_COPY);
    int r2 = a->ops->qsync(a->be, NTSE_Q_COMPUTE);
    if (rc == NTSE_OK && (r1 != NTSE_OK || r2 != NTSE_OK)) rc = NTSE_E_DEVICE;
    st->ownership_violations = w.violations;
    a->h2d += h2d;
    if (rc == NTSE_OK) {
        st->h2d_bytes = h2d; st->d2h_bytes = d2h;
        st->lane = a->lane;
        st->wall_ms = now_ms() - t0;
        return NTSE_OK;
    }
    fault(a, rc, "plan run");
    if (!a->cfg.allow_fallback || rc == NTSE_E_INVAL) return rc;
    if (a->cfg.verbose)
        fprintf(stderr, "ntse-accel: plan run on %s failed (%s); all outputs "
                "recomputed on the CPU reference\n", ntse_lane_name(a->lane),
                ntse_accel_strerror(rc));
    rc = plan_cpu(p, x, y, batch);
    st->wall_ms = now_ms() - t0;
    return rc;
}

/* PCIe probe: median of 5 timed host->device copies of `bytes` from
 * pinned and from pageable memory, and device->host into pinned.
 * Gives the hardware bound the streamed numbers are judged against. */
int ntse_accel_probe_bandwidth(ntse_accel *a, uint64_t bytes, double *h2d_pinned_gbs,
                               double *h2d_pageable_gbs, double *d2h_pinned_gbs)
{
    *h2d_pinned_gbs = *h2d_pageable_gbs = *d2h_pinned_gbs = 0;
    if (!a || bytes == 0 || bytes > (1ull << 30)) return NTSE_E_INVAL;
    if (!on_gpu(a)) return NTSE_E_NODEV;
    ntse_dmem d;
    int rc = charge_alloc(a, bytes, &d);
    if (rc != NTSE_OK) return rc;
    uint8_t *pin = NULL; void *tok = NULL;
    uint8_t *pg = (uint8_t *)malloc((size_t)bytes);
    rc = pg ? a->ops->halloc(a->be, bytes, (void **)&pin, &tok) : NTSE_E_NOMEM;
    if (rc == NTSE_OK) {
        memset(pin, 1, (size_t)bytes); memset(pg, 2, (size_t)bytes);
        double t[3][5];
        for (int r = 0; r < 5 && rc == NTSE_OK; r++) {
            double t0 = now_ms();
            rc = a->ops->put(a->be, NTSE_Q_COPY, &d, 0, pin, bytes, tok);
            if (rc == NTSE_OK) rc = a->ops->qsync(a->be, NTSE_Q_COPY);
            t[0][r] = now_ms() - t0;
            t0 = now_ms();
            if (rc == NTSE_OK) rc = a->ops->put(a->be, NTSE_Q_COPY, &d, 0, pg, bytes, NULL);
            t[1][r] = now_ms() - t0;
            t0 = now_ms();
            if (rc == NTSE_OK) rc = a->ops->get(a->be, NTSE_Q_COPY, pin, &d, 0, bytes, tok);
            if (rc == NTSE_OK) rc = a->ops->qsync(a->be, NTSE_Q_COPY);
            t[2][r] = now_ms() - t0;
        }
        if (rc == NTSE_OK) {
            double *o[3] = { h2d_pinned_gbs, h2d_pageable_gbs, d2h_pinned_gbs };
            for (int k = 0; k < 3; k++) {
                for (int i = 1; i < 5; i++)          /* insertion sort, 5 values */
                    for (int j = i; j > 0 && t[k][j] < t[k][j - 1]; j--) {
                        double v = t[k][j]; t[k][j] = t[k][j - 1]; t[k][j - 1] = v;
                    }
                *o[k] = t[k][2] > 0 ? (double)bytes / (t[k][2] * 1e-3) / 1e9 : 0;
            }
        }
        a->ops->hfree(a->be, pin, tok);
    }
    free(pg);
    charge_free(a, &d);
    if (rc != NTSE_OK) fault(a, rc, "bandwidth probe");
    return rc;
}
