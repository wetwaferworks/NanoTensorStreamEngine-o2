/* fid-linalg.h ... fidelity lab: the inverse-Hessian factor.
 *
 * Co-implemented by Cortana at JD's direction, 2026-11-27. Pure C99.
 *
 * fid_inv_upper_factor(H, n, damp, U) computes the upper-triangular
 * U with (H + damp * mean(diag H) * I)^-1 = U^T U, which is the
 * factor the error-feedback (GPTQ-style) quantizer walks: row j of
 * U, scaled, is row j of the inverse Hessian of the columns still
 * unquantized once columns 0..j-1 are fixed.
 *
 * Route: reverse the index order (A = J Hd J), factor A = L L^T by
 * a blocked right-looking Cholesky in double, invert L by
 * independent column solves, and map back, U = J L^-1 J. Every
 * parallel step writes disjoint rows or columns in a fixed order,
 * so U is bit-identical for any thread count. Returns 0, or -1 if
 * the damped matrix is not positive definite.
 */
#ifndef FID_LINALG_H
#define FID_LINALG_H
#include "fid-common.h"

#define FID_CB 64

typedef struct { double *A; int64_t n, kb, nb; } FidChol;

static double fid_dot_d(const double *a, const double *b, int64_t n)
{
    double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += a[i] * b[i]; s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2]; s3 += a[i + 3] * b[i + 3];
    }
    for (; i < n; i++) s0 += a[i] * b[i];
    return (s0 + s1) + (s2 + s3);
}

/* panel: rows below the diagonal block, columns kb..kb+nb */
static void fid_chol_panel(void *vc, int64_t it)
{
    FidChol *c = (FidChol *)vc;
    int64_t i = c->kb + c->nb + it, n = c->n;
    double *Ai = c->A + i * n;
    for (int64_t j = c->kb; j < c->kb + c->nb; j++) {
        const double *Aj = c->A + j * n;
        double s = Ai[j] - fid_dot_d(Ai + c->kb, Aj + c->kb, j - c->kb);
        Ai[j] = s / Aj[j];
    }
}

/* trailing update of row i: A[i][j] -= L[i][panel] . L[j][panel] */
static void fid_chol_trail(void *vc, int64_t it)
{
    FidChol *c = (FidChol *)vc;
    int64_t j0 = c->kb + c->nb, i = j0 + it, n = c->n;
    double *Ai = c->A + i * n;
    for (int64_t j = j0; j <= i; j++)
        Ai[j] -= fid_dot_d(Ai + c->kb, c->A + j * n + c->kb, c->nb);
}

static int fid_cholesky(double *A, int64_t n)
{
    FidChol c;
    c.A = A; c.n = n;
    for (int64_t kb = 0; kb < n; kb += FID_CB) {
        int64_t nb = n - kb < FID_CB ? n - kb : FID_CB;
        for (int64_t j = kb; j < kb + nb; j++) {
            double *Aj = A + j * n;
            double d = Aj[j] - fid_dot_d(Aj + kb, Aj + kb, j - kb);
            if (!(d > 0.0)) return -1;
            Aj[j] = sqrt(d);
            for (int64_t i = j + 1; i < kb + nb; i++) {
                double *Ai = A + i * n;
                Ai[j] = (Ai[j] - fid_dot_d(Ai + kb, Aj + kb, j - kb)) / Aj[j];
            }
        }
        c.kb = kb; c.nb = nb;
        int64_t rest = n - kb - nb;
        if (rest > 0) {
            fid_par_for(rest, fid_chol_panel, &c);
            fid_par_for(rest, fid_chol_trail, &c);
        }
    }
    return 0;
}

typedef struct { const double *L; double *X; int64_t n; } FidTinv;

/* column j of L^-1, stored contiguous in X[j*n ..] */
static void fid_tinv_col(void *vc, int64_t j)
{
    FidTinv *t = (FidTinv *)vc;
    int64_t n = t->n;
    double *x = t->X + j * n;
    for (int64_t i = 0; i < j; i++) x[i] = 0.0;
    x[j] = 1.0 / t->L[j * n + j];
    for (int64_t i = j + 1; i < n; i++) {
        const double *Li = t->L + i * n;
        x[i] = -fid_dot_d(Li + j, x + j, i - j) / Li[i];
    }
}

static int fid_inv_upper_factor(const float *H, int64_t n, double damp,
                                float *U)
{
    double md = 0.0;
    for (int64_t i = 0; i < n; i++) md += H[i * n + i];
    md /= (double)n;
    if (!(md > 0.0)) md = 1.0;
    double lam = damp * md;
    double *A = (double *)fid_alloc((size_t)n * n * sizeof(double));
    for (int64_t i = 0; i < n; i++)
        for (int64_t j = 0; j < n; j++)
            A[i * n + j] = (double)H[(n - 1 - i) * n + (n - 1 - j)]
                           + (i == j ? lam : 0.0);
    if (fid_cholesky(A, n) != 0) { free(A); return -1; }
    double *X = (double *)fid_alloc((size_t)n * n * sizeof(double));
    FidTinv t;
    t.L = A; t.X = X; t.n = n;
    fid_par_for(n, fid_tinv_col, &t);
    free(A);
    /* U[a][b] = Linv[n-1-a][n-1-b] = X[(n-1-b)*n + (n-1-a)] */
    for (int64_t a = 0; a < n; a++)
        for (int64_t b = 0; b < n; b++)
            U[a * n + b] = b < a ? 0.0f
                : (float)X[(n - 1 - b) * n + (n - 1 - a)];
    free(X);
    return 0;
}

#endif
