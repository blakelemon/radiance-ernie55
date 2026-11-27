# Dense and shared feed-forward forward blocks (`arch/ernie55`, story 2-7)

A third opt-in on top of the weights-only conversion of story 2-5 (see `dense-weight-coverage.md`):

```
rad-convert <ERNIE 5.5 directory> --set ernie55_weights_only=1 --set ernie55_ffn_forward=1 \
    --recipe data/recipes/ernie55-weights.recipe -o ernie55-ffn.rad
radiance --config serve.yaml        # POST /v1/embeddings with token ids
```

`ernie55_ffn_forward=1` is refused without `ernie55_weights_only=1`. It may be combined with
`ernie55_moe_weights=1`; the routed tensors are then in the container and are not read. Without
the key the plugin behaves exactly as in stories 2-4 to 2-6 (stock: config summary then
`RAD_E_UNSUPPORTED`; weights-only: a container the engine refuses at startup for lack of logits).

## What it is, and what it is not

An FFN-block probe, not the language model. Attention, the MTP draft and the
head do not exist here (the head is the opt-in of story 2-9, `target-output.md`; attention and the draft are epic 3), and the routed experts are the opt-in of story 2-8, so
the program declares only what this story owns and serves it as a pooled (embedding) program on the host backend. For each
sequence, per target layer `l`:

```
h0           = hc_enter(embed_lookup(token_embd, tokens))            first layer: embedding broadcast into hc streams
x, post, comb = ernie_mhc_pre_rms(h, blk.l.ffn_hc.{fn,scale,base}, blk.l.ffn_norm.weight)
gate = x @ Wg^T ; up = x @ Wu^T            (two gemm_nt into the halves of one [rows, 2I] plane)
a    = ernie_situ_mul([gate | up]) ; y = a @ Wd^T
h    = mhc_post(y, h, post, comb)
pooled = seg_mean over the sequence's rows of the final hc*H streams, f32
```

The server returns the pooled vector L2-normalised (the embeddings endpoint's choice, not the
model's). Every layer is the feed-forward half alone: the attention branch of the real layer is
absent, so `h` enters the feed-forward branch unchanged. A routed layer (`l >= first_k_dense_replace`)
runs only its shared expert, whose output the real model adds to the routed experts' (`ernie55_moe_forward=1`, story 2-8, `moe-forward.md`).

## Equations (pinned Baidu model, vLLM-ERNIE b8414da, `nvidia/model.py`, `nvidia/moe.py`, `mhc.py`)

* Dense (`l < first_k_dense_replace`): `Ernie5_5MLP(intermediate_size)`: `gate_up_proj` (gate rows
  then up rows), `SituAndMul(beta=activation_situ_beta, linear_beta=activation_situ_linear_beta)`,
  `down_proj`. Weights `blk.l.ffn_{gate,up,down}.weight` (checkpoint `mlp.{gate,up,down}_proj`).
* Shared: the same `Ernie5_5MLP` of width `moe_intermediate_size * n_shared_experts`, with no gate and
  no scale of its own, fed the layer's normed `x` (the original hidden state at full width, not the
  `moe_latent_size` input of the routed experts). Weights `blk.l.ffn_{gate,up,down}_shexp.weight`
  (checkpoint `shared_experts.{w1,w3,w2}`). The shared output is combined with the routed output in 2-8.
* SiTU is `ernie_situ_mul` of story 1-1: `beta*tanh(g/beta)*sigmoid(g) * linear_beta*tanh(u/linear_beta)`
  in f32, one rounding to BF16, gate first.
* The mHC `ffn` branch: `_mhc_args("ffn")` is `(fn, scale, base, rms_norm_eps, hc_eps, hc_eps,
  hc_post_alpha = 2.0, hc_sinkhorn_iters)`: the projection `eps` and the fused norm's `rms_eps` are
  both `rms_norm_eps`, `pre_eps` and `sinkhorn_eps` both `hc_eps`, `post_mult` 2.0. The first layer's
  broadcast kernel (`fn_broadcast` = `fn` summed over the hc axis) equals the ordinary branch over
  `hc` identical streams; the graph materialises the broadcast with `hc_enter` instead.

## Config keys the program requires (no defaults)

`hidden_act` = `situ`, `rms_norm_eps` > 0, `hc_eps` >= 0, `hc_sinkhorn_iters` 1..256,
`activation_situ_beta` > 0, `activation_situ_linear_beta` > 0, `hc_mult` 2..8, plus the 2-5 keys, and
`num_attention_heads` (the engine's container check). Each missing or invalid one is refused at
conversion naming the key. The release's own values are 1e-6, 1e-6, 20, 4 and 25.

## Limits

* BF16 projections only: block-fp8 dense or shared projections (the release's) are refused by name
  (`layer N feed-forward projections are block-fp8; the 2-7 program reads BF16 projections only`).
* One rank, target model only, host (libref) kernels; no GPU kernel exists for `ernie_mhc_pre_rms`.
* The probe's `probe` reports `draft_depth` 0 when the key is set (an embedding program has no
  drafter; the checkpoint's `num_nextn_predict_layers` is otherwise unchanged).

Tests: the story's sealed `2-7-dense-shared-forward.acceptance.cjs`.
