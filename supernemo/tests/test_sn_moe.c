/* test_sn_moe.c - fixture checks for the SuperNemo MoE block (tiny shapes). */
#include "../src/sn_moe.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { fails++; printf("FAIL: %s\n", msg); } } while (0)

static float lcg_state = 1.0f;
static unsigned lcg = 12345u;
static float rnd(void) { lcg = lcg * 1664525u + 1013904223u; return ((float)((lcg >> 8) & 0xffff) / 32768.0f - 1.0f) * 0.5f; }

enum { H = 6, L = 4, I = 5, S = 7, E = 8, K = 3 };
static float gate_w[E * H], bias_[E], fc1[L * H], fc2[H * L], sup[S * H], sdn[H * S];
static float eup[E][I * L], edn[E][L * I];

static int fetch(void *u, int e, const float **up, const float **down)
{ (void)u; *up = eup[e]; *down = edn[e]; return 0; }
static int fetch_fail(void *u, int e, const float **up, const float **down)
{ (void)u; (void)e; (void)up; (void)down; return 1; }

int main(void)
{
    (void)lcg_state;
    for (int i = 0; i < E * H; i++) gate_w[i] = rnd();
    for (int i = 0; i < E; i++) bias_[i] = rnd() * 0.2f;
    for (int i = 0; i < L * H; i++) fc1[i] = rnd();
    for (int i = 0; i < H * L; i++) fc2[i] = rnd();
    for (int i = 0; i < S * H; i++) sup[i] = rnd();
    for (int i = 0; i < H * S; i++) sdn[i] = rnd();
    for (int e = 0; e < E; e++) {
        for (int i = 0; i < I * L; i++) eup[e][i] = rnd();
        for (int i = 0; i < L * I; i++) edn[e][i] = rnd();
    }
    float x[H];
    for (int i = 0; i < H; i++) x[i] = rnd() * 2.0f;
    sn_moe_cfg c = { H, L, I, S, E, K, 5.0f, 1 };

    /* 1. relu2 and sigmoid basics */
    CHECK(sn_relu2(-3.0f) == 0.0f && sn_relu2(3.0f) == 9.0f && sn_relu2(0.0f) == 0.0f, "relu2 values");
    CHECK(fabsf(sn_sigmoid(0.0f) - 0.5f) < 1e-7f, "sigmoid(0)=0.5");

    /* 2. routing: weights sum to the routed scale; ids distinct */
    int top[K]; float tw[K];
    CHECK(sn_moe_route(&c, gate_w, bias_, x, top, tw) == 0, "route ok");
    float sum = 0; for (int k = 0; k < K; k++) sum += tw[k];
    CHECK(fabsf(sum - 5.0f) < 1e-4f, "normalised weights sum to 5.0");
    int distinct = 1;
    for (int a = 0; a < K; a++) for (int b = a + 1; b < K; b++) if (top[a] == top[b]) distinct = 0;
    CHECK(distinct, "top ids distinct");

    /* 3. the bias picks experts but the weights use the UNBIASED score:
     * an enormous bias on expert 5 must put expert 5 first, and its weight
     * must still equal sigmoid(logit)/sum*scale, not saturate. */
    float big[E]; for (int e = 0; e < E; e++) big[e] = 0.0f; big[5] = 100.0f;
    int t2[K]; float w2[K];
    sn_moe_route(&c, gate_w, big, x, t2, w2);
    CHECK(t2[0] == 5, "huge bias selects expert 5 first");
    float logit5 = 0; for (int i = 0; i < H; i++) logit5 += gate_w[5 * H + i] * x[i];
    float s5 = sn_sigmoid(logit5), ssum = 0;
    for (int k = 0; k < K; k++) { float l = 0; for (int i = 0; i < H; i++) l += gate_w[t2[k] * H + i] * x[i]; ssum += sn_sigmoid(l); }
    CHECK(fabsf(w2[0] - 5.0f * s5 / ssum) < 1e-4f, "weight uses the unbiased score");

    /* 4. ties go to the lower index */
    float zero_w[E * H] = {0}; float zb[E] = {0};
    int t3[K]; float w3[K];
    sn_moe_route(&c, zero_w, zb, x, t3, w3);
    CHECK(t3[0] == 0 && t3[1] == 1 && t3[2] == 2, "ties pick the lowest indices");
    CHECK(fabsf(w3[0] - 5.0f / 3.0f) < 1e-5f, "equal scores share the scale equally");

    /* 5. K == E and norm off: no normalisation, scale only */
    sn_moe_cfg c2 = c; c2.topk = E; c2.norm_topk = 0; c2.scale = 2.0f;
    int t4[E]; float w4[E];
    sn_moe_route(&c2, zero_w, zb, x, t4, w4);
    CHECK(fabsf(w4[0] - 2.0f * 0.5f) < 1e-6f, "norm off: weight = scale * sigmoid(0)");
    sn_moe_cfg bad = c; bad.topk = E + 1;
    CHECK(sn_moe_route(&bad, gate_w, bias_, x, t4, w4) == -1, "k > experts refused");

    /* 6. full block against an independent double-precision reference */
    float y[H]; int got_top[K];
    CHECK(sn_moe_forward(&c, gate_w, bias_, fc1, fc2, sup, sdn, fetch, NULL, x, y, got_top) == 0, "forward ok");
    double zl[L], acc[L] = {0}, yd[H], sm[S];
    for (int r = 0; r < L; r++) { double a = 0; for (int i = 0; i < H; i++) a += (double)fc1[r * H + i] * x[i]; zl[r] = a; }
    for (int k = 0; k < K; k++) {
        int e = got_top[k];
        double mid[I];
        for (int r = 0; r < I; r++) { double a = 0; for (int i = 0; i < L; i++) a += (double)eup[e][r * L + i] * zl[i]; mid[r] = a > 0 ? a * a : 0; }
        for (int r = 0; r < L; r++) { double a = 0; for (int i = 0; i < I; i++) a += (double)edn[e][r * I + i] * mid[i]; acc[r] += (double)tw[k] * a; }
    }
    /* tw from the earlier route call uses the same x and tables, so it matches */
    for (int r = 0; r < H; r++) { double a = 0; for (int i = 0; i < L; i++) a += (double)fc2[r * L + i] * acc[i]; yd[r] = a; }
    for (int r = 0; r < S; r++) { double a = 0; for (int i = 0; i < H; i++) a += (double)sup[r * H + i] * x[i]; sm[r] = a > 0 ? a * a : 0; }
    for (int r = 0; r < H; r++) { double a = 0; for (int i = 0; i < S; i++) a += (double)sdn[r * S + i] * sm[i]; yd[r] += a; }
    double maxd = 0; for (int r = 0; r < H; r++) { double d = fabs(yd[r] - y[r]); if (d > maxd) maxd = d; }
    CHECK(maxd < 1e-4, "forward matches the double reference");
    CHECK(got_top[0] == top[0] && got_top[1] == top[1] && got_top[2] == top[2], "forward routes like sn_moe_route");

    /* 7. a failing expert fetch fails the step */
    CHECK(sn_moe_forward(&c, gate_w, bias_, fc1, fc2, sup, sdn, fetch_fail, NULL, x, y, NULL) == -3, "failed fetch fails the step");

    printf("%d checks, %d failed\n", checks, fails);
    return fails ? 1 : 0;
}
