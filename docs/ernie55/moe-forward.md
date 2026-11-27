# Routed latent-MoE forward block (`arch/ernie55`, story 2-8)

A fourth opt-in on top of the feed-forward probe of story 2-7 (see `ffn-forward.md`):

```
rad-convert <ERNIE 5.5 directory> --set ernie55_weights_only=1 --set ernie55_moe_weights=1 \
    --set ernie55_ffn_forward=1 --set ernie55_moe_forward=1 --recipe data/recipes/ernie55-weights.recipe \
    -o ernie55-moe.rad
radiance --config serve.yaml        # POST /v1/embeddings with token ids
```

`ernie55_moe_forward=1` is refused without the three earlier keys (and with a value other than 0 or 1).
Without it the plugin behaves exactly as in stories 2-4 to 2-7: a routed layer runs only its shared
expert. It is still the feed-forward probe, not the language model: no attention, MTP or head.

## Equations (pinned Baidu model, vLLM-ERNIE b8414da, `nvidia/moe.py` `Ernie5_5MoE.forward`)

For a routed layer (`l >= first_k_dense_replace`), `x` the normed collapse of the layer's mHC ffn
branch (the same `x` the shared expert reads):

```
logits  = [x Wr0^T | x Wr1^T]                 [T, 2E] f32 (gate.weight, gate.weight_1; never narrowed)
score_e = sigmoid(logits[e]) + sigmoid(logits[E+e])
ids     = top-k_e (score_e + e_score_correction_bias[e])           ties to the lower id; distinct
w_j     = score[ids_j]                         the UNCORRECTED score
w_j    /= max(sum_j w_j, 1e-20)                 only with norm_topk_prob
w_j    *= routed_scaling_factor_param[ids_j]    always, per logical expert id, after the division
xl      = bf16(x fc1^T)                         [T, Lt]
per expert: g = bf16(xl W1^T), u = bf16(xl W3^T), a = bf16(situ(g, u)), ye = bf16(a W2^T)
comb    = bf16(sum_j w_j ye_j)                  experts COMBINED first, slots in ascending order
nl      = bf16(rmsnorm(comb) * latent_norm)     the latent norm runs ONCE, after the combine
ro      = bf16(nl fc2^T)                        routed_output_transform
y       = bf16(ro + shared(x))                  shared expert reads the original hidden-width x
```

Normalising each expert's output before the weighted sum is a different function (the experts have
different magnitudes); the fixture of the sealed test distinguishes the two by more than 1e-2.

The rounding points are the engine's ops' (`moe_gemm` and `gemm_nt` round to BF16, `moe_gather` sums
the weighted slots in f32 and rounds once, `rmsnorm` and `add` round once). vLLM's fused-MoE kernel
may place the weight multiply inside the down projection instead; the difference is last-bit BF16
rounding, not an equation, and is not pinned by the release.

## Program

Per routed layer, after `ernie_mhc_pre_rms`: the shared expert as in 2-7 (output into its own plane),
`gemm_nt_f32` twice (router halves into the two halves of an f32 `[T, 2E]` logits plane),
`ernie_router_topk` (the correction bias and the learned scale are its weights), `moe_scatter`,
`gemm_nt` (fc1), `moe_gemm` for gate and for up (token-ordered activation, into the halves of one
`[T*k, 2F]` plane), `ernie_situ_mul`, `moe_gemm` for down (sorted order), `moe_gather`, `rmsnorm`
(`rms_norm_eps`), `gemm_nt` (fc2), `add` with the shared output, then `mhc_post` as before. Every op
already existed; no libref op was added. `moe_gemm` reads a weight table of one run of consecutive handles
per projection, so `declare_routed` now declares each layer's experts by projection (all gates, all ups,
all downs); the movement unit is still the group `(layer, expert, slot)`, expert-major, as in 2-6.

## Dtypes

The routers are declared f32 (the loader widens the stored BF16 exactly): the Baidu gate holds its
weights and computes its matmul and logits in f32. The container still stores the BF16 bytes, so a
weights-only (2-6) container is unchanged. The correction bias and the learned scale are F32 and read
at F32. A BF16 read of either, or a BF16 logit, moves the output by more than 1e-3 in the sealed test.

## Config keys (no defaults)

`norm_topk_prob` (0/1), `moe_split_feature_routing` and `routed_scaling_factor_learnable` (both true),
and, when present, `n_group` and `topk_group` (1) and `scoring_func` (`sigmoid`); plus the 2-7 keys.
`num_experts_per_tok` is the top-k and `rms_norm_eps` the latent norm's epsilon. Each missing or
invalid one is refused at conversion naming the key.

## Limits

* BF16 latent projections and experts only: a block-fp8 one (the release's) is refused by name
  (`layer N expert E is block-fp8; the 2-8 program reads BF16 experts only`, `layer N latent
  projections are block-fp8; ...`). The 2-5 dense/shared limit is unchanged.
* One rank, target model only, host (libref) kernels. `ernie_mhc_pre_rms` has no GPU kernel.
* The routing histogram `moe.ecnt` is a persistent plane; nothing reads it yet.

Tests: the story's sealed `2-8-latent-expert-forward.acceptance.cjs`.
