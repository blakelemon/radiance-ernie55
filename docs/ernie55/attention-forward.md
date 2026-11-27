# MLA-DSA attention program (story 3-2b)

Opt-in: `rad-convert --set ernie55_attention_forward=1` on top of `ernie55_weights_only=1` and `ernie55_ffn_forward=1` (and, as
before, `ernie55_moe_weights=1` + `ernie55_moe_forward=1`, `ernie55_target_output=1`). Optional `--set ernie55_attention_block=N`
(KV page size, default 64 as vLLM's DSA_BLOCK_SIZE; the tests use 4 and 8). Without the key nothing changes. Host (libref) backend,
single rank, MLA-DSA layers only: **a model with any HCA main layer is refused at declare by name** (`layer 1 is an HCA layer
(compress_ratios[1] = 128)`), until 3-4 builds HCA. The facts are `m08`..`m17`, `m21`, `m24`, `m26` of `attention-reference.md`.

## One layer, in the order the program runs it

```
x, post_a, comb_a = ernie_mhc_pre_rms(h; attn_hc, input_layernorm gain)         attn-branch mHC, fused norm
qa = rmsnorm(q_a_proj x)  -> qn                                                  q latent, normed (the gate and the indexer read it)
q  = q_b_proj qn                      [T, heads * (nope + rope)]
kv = kv_a_proj_with_mqa x             [T, latent + rope];  kv[:latent] = rmsnorm(kv[:latent])
ernie_rope_tail(q, heads, width nope+rope, rope_table bf16), ernie_rope_tail(kv, 1, latent+rope)   absolute positions
kv_store -> latent cache (slot_mapping of the latent group)
indexer:  ernie_idx_key(wk x) -> index cache;  ernie_idx_query(wq_b qn);  w = weights_proj x
          ernie_row_req(cu_seqlens) -> req;  ernie_idx_logits(..., index block table) -> scores [T, max_ctx]
          ernie_dsa_select(scores, positions, req, latent block table) -> ids, slots   (window 128, top-k 2048)
q_abs = ernie_mla_project(absorb, q, q_width nope+rope);  q_rope = ernie_head_slice(q)
o = ernie_mla_sparse_attn(q_abs, q_rope, latent cache, slots, sink f32);  ov = ernie_mla_project(expand, o)
om = ernie_vha_mix(ov, U, V, fold = bf16);  og = act_mul(sigmoid, gate_proj qn, om);  y = o_proj og
h' = mhc_post(y, h, post_a, comb_a)       then the FFN half of 2-7/2-8 on h' (its own mHC branch)
```

The two halves alternate two stream buffers (attention h0 -> h1, FFN h1 -> h0), so the final streams are in h0.

## Choices

* **KV cache layout.** Two `RAD_KV_FULL` groups, each bound to every MLA layer, both paged at one block size (64, or
  `ernie55_attention_block`): `kv_latent` (bf16, 1 head, `head_dim` 288: the engine's `[pages, 1, block, 576]` row is
  `[normed latent 512 | rotated k_pe 64]`, written by `kv_store`, as `bailing_hybrid`'s MLA keeps it) and `kv_dsa_index`
  (`u8`, 1 head, `head_dim` 66: `[pages, 1, block, 132]`, 128 FP8 codes then the fp32 scale, one row a token, `rad_kv_group_dense_read`
  as glm5next's index group because the logits read the whole history every step). Each group has its own `slot_mapping` and block
  table from the batch; selection runs on logical ids and is translated through the latent table last. The block size of each group
  is claimed by the first op after its declaration that carries one (`kv_store`, `ernie_idx_key`) and read back to check.
  The latent cache is bf16 by default (m25: an FP8 latent costs ~3% on synthetic Gaussian data). `--kv-cache-dtype fp8` (opt-in, LOSSY, not
  qualified on the real model) stores that one group as `fp8_ds_mla` rows
  (`u8`, 1 head, `head_dim` 328: 512 E4M3 codes, 4 f32 power-of-two scales, 64 bf16 rope = 656 bytes against 1152) through
  `ernie_mla_latent_store` and reads them with `ernie_mla_sparse_attn kv_layout=fp8_ds_mla`; the indexer's cache, the HCA groups and any
  drafter cache keep their dtype. `--kv-cache-dtype i8` stays refused.
* **Block-FP8 weights: refused by name, converted exactly outside.** The release stores `q_a/q_b/kv_a/kv_b/gate/o_proj` and the indexer
  `wq_b` as block-FP8. This program reads BF16 projections. A block-fp8 container is converted with the converter's recipe rule
  `--quant 'blk.*.attn_q_a.weight=cast:dtype=bf16'` (one per projection: `attn_q_a`, `attn_q_b`, `attn_kv_a_mqa`, `attn_kv_b`,
  `attn_gate`, `attn_output`, `attn_idx_q_b`), which dequantises exactly; a projection still block-fp8 is refused at declare with that command
  in the message. (Same recipe route as the f32 `hc.fn` casts of 2-7..2-9; the real checkpoint was not converted here: headers only.)
* **Rope table dtype: bf16.** `ernie_rope_tail` gained `rope_table` (default f32, the sealed 3-2a form); the program passes `bf16`
  for the main q/k_pe rotary and the indexer ops use their bf16 default, one table dtype for the DSA layer (m08: default
  RotaryEmbedding, `cache.to(dtype)`).
* **VHA fold: bf16** (m18, settled at tp = 1); the op default stays f32.
* **wk and weights_proj stay two GEMMs** (the pinned model merges them as `wk_weights_proj`; each output column is an independent
  dot product, so the split is numerically the merged GEMM).
* **Positions** are the batch's `positions` (absolute, request-local) for both rotaries, the indexer and the selection, so a chunked prefill
  equals a one-shot one (tested bit for bit).
* **Score plane.** `ernie_idx_logits` writes `[max-num-batched-tokens, max-model-len]` f32; the program refuses more than 2^30 scores.
  Fine for the CPU reference checkpoint and tests; a device program would score in tiles (not done).

## Amendments this story made to earlier ops (all additive; no sealed contract was edited)

`ernie_vha_mix` refuses an overlapping `out`; `ernie_rope_tail` `rope_table`; optional `num_pages` on `ernie_mla_sparse_attn` and
`ernie_dsa_select` (the pool size is unknown at declare); `ernie_idx_key`/`ernie_idx_logits` accept the `[pages, 1, block, row]` cache
view; new ops `ernie_row_req` (request id per row from `cu_seqlens`) and `ernie_head_slice`. The 3-7 facts, report and doc now state the
bf16 DSA rope table (it had been read as f32).

## Not done

HCA layers (3-4), draft/MTP layer, tensor parallelism, block-FP8 weights read in place (`gemm_nt_q`), an FP8 latent cache, MXFP4 indexer
cache, a device (HIP) kernel for any of the new ops, tiled score planes for production context lengths, the real 43-layer checkpoint, and
any real-activation comparison with the reference runtime.
