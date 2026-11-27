# MLA-DSA indexer scores (story 3-2i)

Operators (`libref/ref_ernie_attn.cpp`, host only, ABI 40 untouched): `ernie_idx_query`, `ernie_idx_key`,
`ernie_idx_logits`. They produce the per-token score row `ernie_dsa_select` (3-1) reads, so an MLA-DSA layer
has real candidates beyond the 128-token window. Operand and parameter spelling is in `docs/OPS.md`; the facts
are `m09`, `m10`, `m13`, `m24`, `m26` of `docs/ernie55/attention-reference.md`. Nothing here is wired into a
plugin program yet (that is 3-2b).

## Geometry (frozen from the audit and the checkpoint headers)

| Quantity | Value | Source |
| --- | --- | --- |
| index heads, head dim | 64, 128 | `config.json` `index_n_heads`, `index_head_dim` |
| rope slice | 64, **rope first**: `[rope 64 | nope 64]`, NeoX split-half pairs `(i, i + 32)` | `m09` (`attention.py` 665, 620) |
| rope | plain, `rope_theta` 8,000,000, absolute request-local positions, no YaRN | `m08`, `m24` |
| q projection | `wq_b` `[8192, 1024]` block-FP8 (+ `weight_scale_inv [64, 8]`), from the 1024-wide q latent | headers, all 6 DSA layers |
| k projection, head weights | `wk` `[128, 4096]`, `weights_proj` `[64, 4096]` (the pinned model merges them as `wk_weights_proj`) | headers |
| k norm | `LayerNorm(128, eps 1e-6)`, weight and bias, bf16 | `m09`, headers |
| logit | `sum_h relu(q_h . k_j * k_scale_j) * w_h`, `w_h = weight_h * q_scale_h * 128^-1/2 * 64^-1/2` | `m10` |
| top-k, window | 2048, 128 (candidates `[0, max(seq - 128, 0))`) | `m04`, `m05` |
| key cache row | one row per token, **132 bytes**: 128 E4M3 codes then one fp32 scale (little endian); 64-token pages | `m13` |

The checkpoint header check: the six indexer tensors of every DSA layer (7, 14, 21, 28,
35, 42), 36 headers, match these shapes and dtypes (0 mismatches).

## The ops

```
qf, qs   = ernie_idx_query(q [M, 64*128] bf16, positions)       rope-first NeoX rope -> Hadamard -> FP8 + ue8m0 scale
cache   <- ernie_idx_key(k [M, 128] bf16, gamma, beta, positions, slots)   LayerNorm -> same transform -> 132-byte rows
scores   = ernie_idx_logits(qf, qs, w [M, 64] bf16, cache, block_table, req, positions)   [M, L] f32, logical token ids
ids, slots = ernie_dsa_select(scores, positions, req, block_table)        3-1, unchanged
```

* Per head (query) or per key row after the norm: split rope first, rotate the rope part NeoX at the
  **absolute** position (`angle = position * theta^(-2i/64)`), rejoin as `[rotated | nope]`, then the
  orthonormal Hadamard over all 128, then one E4M3 group of 128 with the power-of-two (ue8m0) scale
  `2^ceil(log2(max(absmax, 1e-10) / 448))`, codes `E4M3(clamp(v / scale, +-448))`, round to nearest even.
* bf16 boundaries are the pinned model's: after the rotary, after the Hadamard (hadacore returns bf16),
  after the LayerNorm. The arithmetic between boundaries is double, so the reference is the exact value
  rounded where the model rounds (an f32 reference flipped one bf16 rounding in 0.4% of key rows, measured
  while building: that moved the token's score by 1%).
* `ernie_idx_logits` walks the block table per token (`block_table[req][j / block]`), so it masks per
  token, which `ref_dsa_score` does not. Scores are indexed by **logical** token id (what `ernie_dsa_select`
  reads); a shuffled page table moves only physical slots (`slots`, and where the cache rows live), never
  `scores` or `ids`. Columns from the window start (`seq - 128`) are `-inf`, as are keys on an
  uninitialised page (entry `< 0`, `>= pages`, or a block past the table).
* Chunked prefill: each chunk writes its keys at their absolute positions (`ernie_idx_key`) and scores its
  own query rows (`ernie_idx_query` + `ernie_idx_logits`) against every key written so far. Positions are
  absolute, so a chunk's scores and selections equal a one-shot prefill bit for bit (sealed test); the
  drift mutants (chunk-local positions) change the cache and the scores.
* Decode is the same with one row per request; the plugin story supplies `slots` from the scheduler.

## Reuse verdicts (3-7 table, with measurements)

* `ref_dsa_score`: **not used for the production path.** It masks per live block (a block is one table
  entry; here one entry per token would be needed) and has no per-row window start. The sealed test feeds it
  a per-token slot table with one row and shows it agrees with `ernie_idx_logits` below the window start to
  1e-5 (the product orders differ), so the weighted-head reduction is the same arithmetic.
* `ref_dsa_query`: same Hadamard and ue8m0 quantiser arithmetic in f32, but a scale floor of 1e-4 (here
  1e-10, the default of the pinned `per_token_group_quant_fp8`, from memory of that file: not in the audit
  pin set) and no rope, so it was not reused; the new ops contain the quantiser.
* `ref_layernorm`: not called; the key op fuses the LayerNorm with the rope and Hadamard so the bf16 boundary
  and double arithmetic apply; the arithmetic is the layernorm op's (biased variance, `eps` inside the
  root, weight then bias), not cross-tested.
* `ref_rope`: not used. Its `neox` mode with `rotary_dim` 64 rotates the leading slice with the right pairing, so it
  could give the f32-table form to within f32 angle noise, but it forms the angle and frequency ladder in f32,
  has no bf16 table, and does not apply the bf16/double boundary sequence, so the op rotates itself.

## Choices

* **Indexer key cache dtype: FP8 with a per-row ue8m0 scale (132 bytes a token).** It is the pinned default
  and the only layout valid with context parallelism (MXFP4 needs PCP = DCP = 1, `m26`). Measured on
  synthetic Gaussian activations with the real `k_norm` weight and bias of layers 7 and 42, 4097 keys, three
  query rows, production 64 heads x 128: a bf16 key cache
  (q still FP8) has 2.0-2.2% logit relative L2 against a double-precision ideal versus 2.9-3.1% for FP8 keys, and the
  two caches select the same 2048 tokens to 99.2-99.3% (the boundary tokens differ). FP8 is kept for
  parity; a bf16 key cache would cost about twice the memory per key row (256 against 132 bytes) for about one point less logit error.
  Synthetic data: no claim about the real model.
* **Rope table dtype: bf16 (the model dtype), `rope_table=f32` available.** The indexer builds its rotary with
  the same `get_rope` as the main path (`attention.py` 617, default dtype = model dtype), and the cos/sin
  table of vLLM's default `RotaryEmbedding` is cast to that dtype (review finding; the cast itself lives
  in `rotary_embedding/base.py`, which the 3-7 pin set did not fetch, `get_rope`'s default dtype did).
  Measured: an f32 table moves the logits by 1.0% relative L2 and the top-2048 set by 0.4%
  (same measurement), both worse and better than the ideal in equal measure (3.08% against
  3.10% of the double ideal), so the choice is documented, not data-driven. The main-path 3-2a
  `ernie_rope_tail` keeps its f32 table (sealed bit-exactly); that mismatch was closed by story 3-2b: `ernie_rope_tail` takes `rope_table` and the program passes `bf16` for the main path too. The
  angle is formed in double and its cosine rounded to f32 and then to bf16 (vLLM forms the angle in f32).
* **m06 (top-k order and ties)**: unchanged from 3-1: a candidate set that fits is listed ascending; a
  truncated one by descending score with ties to the **lower id**, an assumption (one legal outcome of the
  kernels). Pinned end to end by a tied fixture through the real logits (identical key rows, five of eight
  tied tokens survive: the lowest ids).
* **m22 (kernel numerics), measured, not read**: the CUDA DeepGEMM logits and hadacore were not read. Measured
  on the same data, relative to this reference: rounding the dot to bf16 (the ROCm fallback einsum) moves
  logits by 0.13% relative L2 (max 0.23% of the largest logit) and the top-2048 set by 0.03%; a global
  Hadamard scale (unnormalised, 1/sqrt(128) = 2^-3.5 not a power of two) leaves the ranking within the FP8
  noise (overlap 98.8-99.1%), so hadacore's exact constant only matters through the FP8 grid. FP8 logits
  against the double ideal: 3.0% relative L2, top-2048 overlap 98.8-99.2%. A device kernel is accepted
  against this reference within about 3% of the largest logit per score and a selection that differs only
  at the k-th place.
* **Scale floor 1e-10** in the quantiser (absmax floor) only matters for all-zero rows (scale 2^-42).

## Refusals

Every launch validates every operand and index before the first store: head dim not a power of two (4..256),
odd or oversized rope dim, non-positive theta or eps, negative position, slot below -1 or past the pool, a
request outside the block table, scores narrower than the candidate range, wrong dtypes, ranks or strides,
a null operand, an unknown `rope_table`. `ernie_idx_key` applies rows in order (a repeated slot keeps the
last); a negative slot is padding.

## Not done here

Plugin wiring (3-2b: `wq_b` block-FP8 dequantisation, merging `wk` and `weights_proj`, the real `slot_mapping`
from the scheduler, the cache group in the KV plan), MXFP4 indexer cache, any device kernel, real-activation
selection quality, the engine's byte order inside a 64-token page (the reference uses row-contiguous 132-byte
rows; DeepGEMM's page layout was not audited and is irrelevant to a host reference).
