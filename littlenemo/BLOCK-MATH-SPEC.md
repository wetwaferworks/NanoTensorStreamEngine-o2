# Little Nemo: block math spec (P2 reference)

The exact computation the C engine must reproduce, extracted 2026-11-24 from
NVIDIA's own sources for NVIDIA-Nemotron-Nano-9B-v2: modeling_nemotron_h.py
(the mixer's fused forward, its torch_forward reference path, the block, MLP,
and cache code) and configuration_nemotron_h.py, cross-checked against the
mamba_ssm layernorm_gated reference implementation. Checkpoint constants are
from the model's config.json. Tensor shapes were verified against the real
checkpoint by the P0 inspector (shards 1 and 2).

Constants: hidden 4480, layers 56 (27 Mamba-2, 25 MLP, 4 attention at
14, 21, 30, 39 per hybrid_override_pattern), vocab 131072 untied,
eps 1e-5 everywhere, BOS 1, EOS 12, PAD 0.

## Block (every layer)

residual = x
h = RMSNorm(x cast to the norm weight's dtype), full width 4480:
    rstd = 1 / sqrt(mean(h^2) + 1e-5), out = h * rstd * norm.weight
out = residual + mixer(h)
residual_in_fp32 is False for this checkpoint: the residual stream stays in
the stream dtype in the reference. (Our C accumulators are fp32 internally;
the cast points above are the ones that define the reference arithmetic.)

## Mamba-2 block

in_proj: 4480 -> 22656, no bias. Split of the projection, in order:
[0, 0, gate z (10240), conv input u (12288), dt raw (128)].
The two zero-width segments are the gated-MLP branch (d_mlp computes to 0
for this model). u splits after the conv as [x (10240), B (1024), C (1024)];
B and C are per-group: 8 groups of 128, broadcast to the 128 heads
(16 heads per group).

Conv (decode): y_u = silu( sum_{k=0..3} W[k] * u_{t-3+k} + bias ),
u_j = 0 for j < 0. The cache window holds the last 4 conv inputs.
(Prefill applies the same causal conv over the whole sequence.)

SSM decode step, per head h (128), head dim d (80), state n (128):
    dt = softplus(dt_raw[h] + dt_bias[h])
    A[h] = -exp(A_log[h])                       (fp32)
    state[h][d][n] *= exp(dt * A[h])
    state[h][d][n] += dt * B[h][n] * x[h][d]
    y[h][d] = sum_n C[h][n] * state[h][d][n]    (state AFTER the update)
    y[h][d] += D[h] * x[h][d]
CORRECTION to the architecture study: there is NO [0.001, 0.1] clamp on dt
for this checkpoint. time_step_limit defaults to (0.0, +inf) in
NemotronHConfig and config.json does not override it, so the clamp in the
reference path is a no-op. 0.001 and 0.1 are mamba_dt_min / mamba_dt_max,
the dt_bias INITIALIZATION range at training time, not an inference clamp.
The 1e-4 dt_init_floor is likewise an init constant.

Prefill: the chunked scan (chunk 128) is the same recurrence unrolled;
the P2 gate is that our prefill reproduces the recurrent step's state and
outputs on a test sequence.

Gated norm (the study's open detail, now settled): MambaRMSNormGated runs
with norm_before_gate = False, group_size 1280 (8 groups over 10240).
Per the mamba_ssm reference (rms_norm_ref and its Triton kernel):
    g = y * silu(z)                              (gate applied FIRST)
    per group: rstd = 1 / sqrt(mean(g^2) + 1e-5) (fp32 statistics)
    out = g * rstd * norm.weight
Then out_proj: 10240 -> 4480, no bias.

## MLP block

out = down( relu2( up(x) ) ), up 4480 -> 15680, down 15680 -> 4480,
no biases. relu2(v) = relu(v)^2.

## Attention block (layers 14, 21, 30, 39)

q 4480 -> 5120 (40 heads x 128), k and v 4480 -> 1024 (8 KV heads x 128),
o 5120 -> 4480, no biases anywhere. KV repeated 5x to the Q heads.
No position embeddings of any kind (NoPE). Causal attention, scale
1 / sqrt(128). KV cache stores bf16 in the reference design.

## Embedding and head

backbone.embeddings.weight [131072 x 4480]; final norm backbone.norm_f;
lm_head separate [131072 x 4480] (tie_word_embeddings False).
