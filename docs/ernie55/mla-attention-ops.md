# MLA-DSA attention operations (story 3-2a)

Operators (`libref/ref_ernie_attn.cpp`, host only, ABI 40 untouched): `ernie_rope_tail`,
`ernie_mla_project`, `ernie_vha_mix`, `ernie_mla_sparse_attn`. Operand and parameter spelling is in
`docs/OPS.md`; the facts they implement are `m08`, `m11` to `m13`, `m15` to `m17`, `m20`, `m21` and
`m24` of `docs/ernie55/attention-reference.md`. They are the pieces the 3-7 reuse table found missing
for one MLA-DSA layer; story 3-2b wires them, with the existing `act_mul` (sigmoid gate), GEMMs and
`ernie_dsa_select` (3-1), into an opt-in plugin program.

## The audited order of one layer

```
q_a = q_a_layernorm(q_a_proj(x))                      existing ops
q   = q_b_proj(q_a)                  [T, 64 * 256]    per head [nope 192 | rope 64]
kv  = kv_a_proj_with_mqa(x)          [T, 576]         [latent 512 | k_pe 64]
kv  = [kv_a_layernorm(kv[:512]) | kv[512:]]
ernie_rope_tail(q, heads 64, width 256, rope 64)      GPT-J interleaved, trailing slice
ernie_rope_tail(kv, heads 1,  width 576, rope 64)     only k_pe turns; the latent is untouched
cache write of the 576-wide row per token
ids/slots = ernie_dsa_select(...)                     3-1
q_abs = ernie_mla_project(absorb, q_nope part)        [T, 64 * 512], q_width 256 reads only the nope columns
o     = ernie_mla_sparse_attn(q_abs, q_rope, cache, slots, sink)       [T, 64 * 512]
o     = ernie_mla_project(expand, o)                  [T, 64 * 256]
o     = ernie_vha_mix(o, U, V)                        head axis
o     = act_mul(sigmoid, gate_logits, o)              element-wise: sigmoid(o_gate(q_a)) * o
y     = o_proj(o)                                     wo_b, [4096]
```

The gate multiplies after the mix and before the output projection (`m16`). The element gate is the
existing `act_mul` with `act` sigmoid: it rounds the sigmoid to the tensor dtype, multiplies and rounds
again, which is `m21` (the gate is taken in the model dtype). `ref_head_gate` (a per-head factor 2) is
not it.

## Why new operations (3-7 reuse verdicts)

* `ref_mla_project` needs `heads * 2 * d` matrix rows with one width `d` for keys and values; the ERNIE
  `kv_b_proj` has `64 * (192 + 256) = 28672` rows. `ernie_mla_project` takes `k_dim` and `v_dim`
  separately, and in `absorb` mode reads only the leading `k_dim` columns of a `q_width`-wide head so
  the rope columns of the `q_b` row never enter the absorbed query.
* `ref_mla_attn_paged` attends every key of a dense causal range, has no sink and a per-head scalar
  gate. `ernie_mla_sparse_attn` attends the physical slots a row lists (the output of
  `ernie_dsa_select`), has the sink, and leaves the gate to `act_mul`.
* `ref_rope` rotates the leading slice. `ernie_rope_tail` rotates the trailing slice (`m08`).
* `vha_mix` does not exist (`m17`).

## Semantics

* `ernie_rope_tail`: pair `i` of the trailing `rope_dim` values is `(2i, 2i + 1)`, angle
  `position * theta^(-2i / rope_dim)` with the position the row's absolute request-local position
  (`m24`): a chunk of a prefill must pass `chunk_start + i`, and rows never depend on their batch.
  The angle is formed in double, cos/sin rounded to f32, the rotation is f32, one bf16 rounding. NOTE (Review): the pinned DSA layers use the default RotaryEmbedding whose cos/sin table is cast to the model dtype (bf16 cos/sin from f32 angles); only the HCA class keeps an f32 cache. This op keeps an f32 table (the sealed 3-2a test pins it bit-exactly): a device kernel and 3-2b must account for the bf16 table, which needs a recorded contract amendment.
  Story 3-2b added `rope_table` (`f32` default, `bf16` rounds cos and sin to bf16: the DSA layers' table); the program passes `bf16`.
* `ernie_mla_project`: `absorb` `y[t,h,:] = x_nope[t,h,:k_dim] . W_UK[h]`, `expand`
  `y[t,h,c] = sum_l x[t,h,l] W_UV[h][c,l]`. Absorbing then expanding is the same function as projecting
  every cached latent row to per-head keys and values first (the unabsorbed scalar reference the
  acceptance test compares against), up to bf16 intermediates.
* `ernie_mla_sparse_attn`: `score = scale * (q . latent + q_rope . rope)` (scale a parameter, 0.0625 for
  ERNIE, `m11`), value = latent, one fp32 sink logit per head in the running max and the denominator
  only (`m15`). `-1` slots are pads; a slot listed twice counts twice (the HCA union does not
  de-duplicate, `h08`); a slot outside `[-1, pages * block_size)`, a NaN or `+inf` sink refuse before
  any store; `-inf` is no sink; no key and a finite sink gives zeros.
* `ernie_vha_mix`: `out[t,h,c] = sum_g m[h,g] o[t,g,c]`, `m = I + V U^T` folded once. Operand overlap rule (enforced since story 3-2b, `RAD_E_INVAL`): the bytes of `out` may not overlap those of `o` (extents; no in-place call).

## Choices on the unresolved facts

* **m18, dtype of the folded VHA matrix: SETTLED (review, 2026-11-21).** At tp=1 vLLM builds `m` under the model dtype (base_loader.py:73) and matmuls it against the bf16 `o`, so the fold is bf16: ERNIE callers (3-2b) pass `fold=bf16`. The op still defaults to `fold=f32` (changing the default needs a contract amendment, mutant V04); both are exposed and pinned exactly by the sealed 3-2a test. Measured on real layers 7 and 42: the bf16 fold moves entries by up to 7.2e-3, about one output rounding.
* **m22, kernel numerics.** The reference accumulates scores, softmax weights and the value sum in f32
  and rounds the output to bf16 once. A bf16-MMA kernel that rounds the probabilities to bf16 before the
  value product differs by 1.9e-3 relative L2 (mean over three synthetic 2176-key trials,
  measured), one bf16 unit roundoff (1.95e-3). A device kernel is
  accepted against this reference within that band; bit-level agreement with FlashMLA was not read.
* **m25, DSA latent cache dtype.** The op reads a bf16 cache (the `auto` backend dtype, 1152-byte rows).
  An FP8 e4m3 latent with one scale per 128 values (the `fp8_ds_mla` layout's body) changes the attention
  output by 3.0% mean and 3.1% max relative L2 on the same synthetic data, 15 times the bf16 band, so the
  CPU reference does not adopt it; the real-weight effect is for stories 5-x. The synthetic data are
  Gaussian, not real activations: the number bounds a design choice, it is not a model-quality claim.

## Not done here

The plugin program (3-2b), the indexer score math (`m09`, `m10`), KV cache lifecycle, HCA layers,
chunked-prefill cache writes, `wo_b` dequantisation from block-FP8, any device kernel, and any real
activation run.
