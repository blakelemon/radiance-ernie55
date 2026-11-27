# Dense, shared and head weight coverage (`arch/ernie55`, story 2-5)

The ERNIE plugin can declare the dense, shared and head weights of ERNIE 5.5 Flash so that
`rad-convert` writes them and the container reader returns them. It is opt in, because no forward
block exists yet and a container without ops and logits cannot serve:

```
rad-convert <ERNIE 5.5 directory> --set ernie55_weights_only=1 \
    --recipe data/recipes/ernie55-weights.recipe -o ernie55-weights.rad
```

Without `ernie55_weights_only=1` the plugin behaves exactly as in story 2-4 (config summary, then
`RAD_E_UNSUPPORTED`, no weight read). The key is recorded in the container's metadata.

Names are the pinned release's (`model.safetensors.index.json`, revision 028544b2) and shapes and
dtypes are its shard headers' (shards 1, 4 and 128) and the Baidu model's
(vLLM-ERNIE b8414da, `nvidia/model.py`, `nvidia/moe.py`). `H` hidden, `V` vocab, `hc` hc_mult,
`mix = (2+hc)*hc`, `W = hc*H`, `I` intermediate_size, `S = moe_intermediate_size*n_shared_experts`.

| Declared weight | Checkpoint tensor(s) | Stored shape and dtype | Declared |
|---|---|---|---|
| `token_embd.weight` | `model.embed_tokens.weight` | [V, H] BF16 | BF16 |
| `output.weight` | `lm_head.weight` | [V, H] BF16 | BF16 |
| `output_norm.weight` | `model.norm.weight` | [H] BF16 | BF16 |
| `head_hc.fn` | `model.mhc_contract.hc_head_fn` | [W, hc] F32 (in, out), transposed | [hc, W] F32 |
| `head_hc.base`, `head_hc.scale` | `model.mhc_contract.hc_head_base`, `_scale` | [hc], [1] F32 | F32 |
| `multimax.ranges`, `multimax.coefficients` | `model.lm_head.multimax_ranges`, `multimax_ts` | [4] BF16 | F32 (exact widening at load) |
| `blk.L.attn_norm.weight` | `model.layers.L.input_layernorm.weight` | [H] BF16 | BF16 |
| `blk.L.ffn_norm.weight` | `model.layers.L.post_attention_layernorm.weight` | [H] BF16 | BF16 |
| `blk.L.{attn,ffn}_hc.fn` | `...{self_attention,mlp}_hyper_connection.mapping_proj.weight` | [W, mix] F32 (in, out), transposed | [mix, W] F32 |
| `blk.L.{attn,ffn}_hc.base` | `...hyper_connection.bias` | [mix] F32 | F32 |
| `blk.L.{attn,ffn}_hc.scale` | `...hyper_connection.alpha_pre`, `alpha_post`, `alpha_res` | [1] F32 each | [3] F32, order pre, post, res |
| `blk.L.ffn_{gate,up,down}.weight`, L < first_k_dense_replace | `model.layers.L.mlp.{gate,up,down}_proj.weight` | [I, H], [I, H], [H, I] | same |
| `blk.L.ffn_{gate,up,down}_shexp.weight`, L >= first_k_dense_replace | `...block_sparse_moe.shared_experts.{w1,w3,w2}.weight` | [S, H], [S, H], [H, S] | same |

`L` runs over the `num_hidden_layers` target layers. Gate and up stay separate weights (no
gate_up fusion yet). The release stores the dense and shared linears as block-FP8 (`F8_E4M3` codes
plus an F32 `weight_scale_inv` plane); a linear is declared either as one BF16 weight or as a codes
view and a scale view (`blk.L.ffn_gate.weight` and `blk.L.ffn_gate.scale`) of one stored weight,
and both forms are kept byte for byte. The mixers are stored (in, out) and read (out, in): the
converter transposes them in F32 and the recipe's `cast dtype=f32` rules store the result.

Not declared here: attention, routers, latent projections, routed experts and the draft (MTP)
layer; they belong to later stories.

## Diagnostics

* A required tensor that the checkpoint lacks: the converter names it
  (`'X' is declared over 'X': it needs checkpoint tensor 'T', which is not in ...`).
* A tensor whose shape or dtype is not the one above (transposed, wrong width, F32 where BF16 is
  required, fp8 off a 128 block): `radiance: ernie55: checkpoint tensor 'T' (weight 'X') is bf16
  [12, 8], but this model needs bf16 [8, 12]`.
* `vocab_size`, `intermediate_size` (only when a dense layer exists) and `moe_intermediate_size`
  (only when a shared-expert layer exists) must be present integers.

The source dtype of a transposed mixer is not checked: the transpose widens any source to F32.

Tests: the story's sealed `2-5-dense-weight-coverage.acceptance.cjs`.
