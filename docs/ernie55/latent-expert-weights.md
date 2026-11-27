# Routed latent-MoE weight coverage (`arch/ernie55`, story 2-6)

A second opt-in on top of the weights-only conversion of story 2-5 (see `dense-weight-coverage.md`).
It is a separate key because the sealed 2-5 contract converts checkpoints that carry no routed
tensor:

```
rad-convert <ERNIE 5.5 directory> --set ernie55_weights_only=1 --set ernie55_moe_weights=1 \
    --recipe data/recipes/ernie55-weights.recipe -o ernie55-weights.rad
```

`ernie55_moe_weights=1` without `ernie55_weights_only=1` is refused, naming both. With both, every
routed layer (`first_k_dense_replace` <= L < `num_hidden_layers`; the draft layer is not declared)
also holds the weights below. `E` n_routed_experts (at most 65536), `H` hidden, `Lt`
moe_latent_size, `F` moe_intermediate_size. Names are the release's (`model.safetensors.index.json`,
revision 028544b2) and shapes and dtypes are its shard headers' (layer 1 in shards 1 and 4, layer 43
in shard 128) and `vllm_ernie/.../nvidia/moe.py` (b8414da). `m` = `model.layers.L.block_sparse_moe`.

| Declared weight | Checkpoint tensor | Stored shape, dtype | Declared |
|---|---|---|---|
| `blk.L.ffn_gate_inp.weight` | `m.gate.weight` | [E, H] BF16 | same |
| `blk.L.ffn_gate_inp_1.weight` | `m.gate.weight_1` | [E, H] BF16 | same |
| `blk.L.exp_probs_b.bias` | `m.e_score_correction_bias` | [E] F32 | F32 |
| `blk.L.ffn_gate_inp.scale` | `m.gate.routed_scaling_factor_param` | [E] F32 | F32 |
| `blk.L.ffn_latent_in.weight` | `m.fc1_latent_proj.weight` | [Lt, H] block-FP8 (or BF16) | same |
| `blk.L.ffn_latent_out.weight` | `m.fc2_latent_proj.weight` | [H, Lt] block-FP8 (or BF16) | same |
| `blk.L.ffn_latent_norm.weight` | `m.latent_norm.weight` | [Lt] BF16 | BF16 |
| `blk.L.ffn_gate_exps.X.weight` | `m.experts.X.w1.weight` | [F, Lt] block-FP8 (or BF16) | same |
| `blk.L.ffn_up_exps.X.weight` | `m.experts.X.w3.weight` | [F, Lt] block-FP8 (or BF16) | same |
| `blk.L.ffn_down_exps.X.weight` | `m.experts.X.w2.weight` | [Lt, F] block-FP8 (or BF16) | same |

The two routers are two separate matrices: `gate.weight` scores logits `[0, E)` and `gate.weight_1`
logits `[E, 2E)` (the vLLM loader stacks them in that order; `ernie_router_topk` reads
`[M, 2E]`). The bias selects experts and the scale multiplies the kept weights, both indexed by
logical expert id. Experts run in latent space (`fc1` maps H to Lt, the experts work at Lt with
intermediate F, `latent_norm` and `fc2` run after the combine; the shared expert reads the
original hidden states). FP8 weights are declared as a codes view and a `.scale` view of one stored
weight, as for the dense linears (`blk.L.ffn_gate_exps.X.scale`).

## Expert order

Each expert is declared by its logical id X (the number in the checkpoint name) as three weights in
movement slots 0 (gate), 1 (up), 2 (down), expert-major, so the container writes a layer's experts
0..E-1 as fixed-stride units. The declaration order is by projection (all gates, then all ups, then all
downs), which makes each projection one run of consecutive handles for `moe_gemm`'s weight table
(story 2-8); the units and slots are as above. Which shard holds a tensor never matters: the loader resolves each
declared name through the checkpoint's index (or its one file). When a tensor name is in two shards
the index entry picks the copy; with no index entry the loader refuses the checkpoint (`'T' is also
in ...`).

## Diagnostics

* A required tensor that is absent: the converter names it (same text whatever the shard order).
* Wrong shape or dtype, including a transposed projection, a latent width other than
  `moe_latent_size`, an F32 auxiliary stored narrower, an F32 router: `radiance: ernie55: checkpoint
  tensor 'T' (weight 'X') is bf16 [5, 8], but this model needs bf16 [4, 8] (moe_latent_size 4)`.
  Experts add `expert X of layer L (moe_latent_size .., moe_intermediate_size ..)`.
* A checkpoint tensor for expert id E (one past the configured count), whichever of its `w1`, `w3` and
  `w2` it is: `checkpoint tensor '..' is expert E of layer L, but n_routed_experts is E`. Only the first id
  past the end is probed (the plugin can ask for names, not list them), so a surplus id beyond a gap is
  not seen.
* `n_routed_experts` above 65536 is refused.

Not declared here: attention, the draft layer's weights and its routed tensors, and any forward
block. The scale plane of an FP8 weight is not shape-checked by the plugin (the converter derives
the encoding from it).

Tests: the story's sealed `2-6-latent-expert-weights.acceptance.cjs`.
