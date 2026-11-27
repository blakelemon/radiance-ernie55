# Target output: head, final norm, vocabulary projection and MultiMax (`arch/ernie55`, story 2-9)

A fifth opt-in on top of the feed-forward program of stories 2-7 and 2-8 (see `ffn-forward.md`,
`moe-forward.md`):

```
rad-convert <ERNIE 5.5 directory> --set ernie55_weights_only=1 --set ernie55_moe_weights=1 \
    --set ernie55_ffn_forward=1 --set ernie55_moe_forward=1 --set ernie55_target_output=1 \
    --recipe data/recipes/ernie55-weights.recipe -o ernie55-target.rad
radiance --config serve.yaml        # POST /v1/completions
```

`ernie55_target_output=1` is refused without `ernie55_weights_only=1` and `ernie55_ffn_forward=1`, with a
value other than 0 or 1, and, when the model has routed layers (`first_k_dense_replace` below
`num_hidden_layers`), without `ernie55_moe_forward=1` (and `ernie55_moe_weights=1`): their experts would be
skipped and the scores would not be the model's. A model with no routed layer needs no routed keys.
Without the key the plugin behaves exactly as in stories 2-4 to 2-8 (a pooled embedding program).

## What it is, and what it is not

The language-model head on top of the feed-forward stack, served through the engine's normal output path:
the program declares a logits buffer (`rad_declare_logits`) instead of the pooled one, so the scheduler,
the sampler chain, `logprobs` and the rest of generation see an ordinary text model. There is still no
attention, so a position's scores depend on that token alone (the last hidden state of a sequence is that
token's feed-forward stack output); the MTP draft head does not exist (`draft_depth` stays 0) and
the first prefill chunks of a long prompt compute their stacks and sample nothing, as for every model.
The attention layers of stories 3-x change what a position's hidden state depends on, not the head.

## Equations (pinned Baidu model, vLLM-ERNIE b8414da `nvidia/model.py`; vLLM v0.32 `hc_head_fuse_tilelang`)

On each row the sampler asks for (`RadBatch::out_ids`; the head is per token, so it runs on those rows only), `h`
the final `hc * H` residual streams:

```
rs      = rsqrt(sum(h^2) / (hc*H) + rms_norm_eps)            epsilon INSIDE the root
gate_j  = sigmoid(dot(fn_j, h) * rs * hc_head_scale + hc_head_base_j) + hc_eps
x       = bf16(sum_j gate_j * h_j)                            ernie_mhc_head, one rounding
y       = bf16(rmsnorm(x) * model.norm.weight)                final RMSNorm, rms_norm_eps
logit   = y . lm_head                                          float32 [V]
score   = logit + t0*relu(r0-logit) + t1*relu(logit-r1) + t2*relu(r2-logit)^2 + t3*relu(logit-r3)^2
```

`r` is `model.lm_head.multimax_ranges`, `t` is `model.lm_head.multimax_ts`; `score` (ernie_multimax, story 1-2 and
2-13) is what sampling receives. Inside the learned ranges the transform is the identity; outside it is not
monotone for a negative coefficient, which is why the arg-max of the scores can differ from the arg-max of
the raw projection.

## Program

After the last layer, per step with `n_out > 0`: `gather_rows` (the sampled rows of the final streams),
`ernie_mhc_head`, `rmsnorm`, `logits_gemm` (`output.weight`, f32 logits) and `ernie_multimax` in place on the
logits buffer. The logits buffer is `[R, vocab_size]` with `R` sized as the stock plugins size it
(`max_seqs * (max_spec + 1)`, at most `max_tok`, at least `max_out_rows`). A scored step (`score_only`) is not
offered: the plugin does not declare `rad_declare_scores`, so the core always takes the full path.

## Dtypes

The head collapse reads `head_hc.fn`, `head_hc.base` and `head_hc.scale` as F32 (2-5); `multimax.ranges` and
`multimax.coefficients` are stored BF16 and declared F32 (an exact widening at load), the only dtype
`ernie_multimax` accepts: declaring them BF16 makes the engine refuse the program when it serves.
Logits are accumulated in f32 and not rounded to BF16 before the transform (story 2-13's form). Settled by the
review of 2-9: Baidu's `head_dtype` is f32 by default and vLLM accumulates the logits in f32, so the
f32 form matches the reference.

## Limits

* One rank, target model only, host (libref) kernels; `ernie_mhc_pre_rms` has no GPU kernel.
* BF16 projections only, as in 2-7 and 2-8; the head tensors are BF16/F32 in the release.
* No attention, no MTP draft head, no KV cache: see the first section.

Tests: the story's sealed `2-9-target-output.acceptance.cjs` (vocabulary 19, so the sampler's top-20
log-probabilities are the full distribution).
