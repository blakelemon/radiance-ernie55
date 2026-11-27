# ERNIE 5.5 HCA attention host ops (story 3-4a)

Host-only libref operations for the attention branch of the 37 HCA layers, beside the 3-2a (MLA) and 3-3b (compressor) ones. They compose, with the existing
`rmsnorm`, `gemm_nt_f32`, `ernie_hca_state_write`, `ernie_hca_compress`, `ernie_rope_tail`, `ernie_vha_mix`, `act_mul(sigmoid)` and `gemm` ops, into the
branch of `docs/ernie55/attention-reference.md` (facts h04 h08 h09 h10 h11 h16 h17); the plugin wiring is story 3-4b. `arch/ernie55` and ABI 40 are untouched.

## The three new ops

* `ernie_hca_q_prep(x inout bf16 [M, heads*head_dim], positions)`: per head `y = x * rsqrt(mean(x^2) + eps)` in f32 (no weight, h10), then the trailing `rope_dim`
  values rotated GPT-J interleaved at the absolute position with the f32 cos/sin table, rounded **once** to bf16 (the pinned fused q kernel rotates before it stores).
* `ernie_hca_union_attn(q, win_kv, cmp_kv, positions, req, win_table, cmp_table, sink, out)`: row at position `p` of request `q = req[r]` attends, in one softmax with the
  per-head fp32 sink (max and denominator, no value), the compressed rows `j < (p + 1) / ratio` (complete groups only, h03) and the window tokens
  `max(0, p - window + 1) .. p`. Key = value = the cache row (h09), so the rotated rope channels are part of the value; score `scale * q . row` with
  `scale = 512^-0.5`. The caches are paged `[pages, 1, block, head_dim]` bf16 with their own block tables (`[seqs, blocks]`, -1 = unallocated page).
  **The window and the newest complete group overlap and are not de-duplicated** (amended spec of 3-4, fact h08): a token inside the newest group is attended raw
  and again through its group row, one entry each. Only future positions, groups that are not complete yet, and -1 pages are excluded. A -1 entry skips its keys
  (a WINDOW-style group frees pages older than the window); an entry past the pool, a block index past the table width, a negative position or request, a NaN or
  +inf sink, a ratio other than 4 or 128 refuses before any store.
* `ernie_hca_wo_a(o, positions, wo_a, out)`: the trailing `rope_dim` values of every head of `o` (the attention output after VHA) are rotated by the **inverse**
  rotary (sin negated) at the query position (h10) and rounded to bf16, then the grouped `wo_a` `[groups * rank, heads / groups * head_dim]` (real:
  `linear_o_group_proj` `[8192, 4096]`, 8 groups of 8 heads x 512, rank 1024): group `g` reads its contiguous heads and weight rows `g * rank ..`. f32 accumulation,
  one bf16 store. Order in the layer (h16): VHA (`ernie_vha_mix`, head axis, commutes with the dim-axis inverse rope) -> this op -> `* sigmoid(gate)` on the wo_a output
  (gate from the normed q latent, width `groups * rank`) -> `wo_b`.

## YaRN (k22 / h12 resolved)

The HCA rotary is DeepseekV4ScalingRotaryEmbedding: theta `compress_rope_theta` 160000, YaRN factor 16, beta_fast 32, beta_slow 1, original length 65536, magnitude exactly 1
(mscale forced to 0 in `build_deepseek_v4_rope`), f32 cos/sin table. The correction range is `floor(dim(32)) .. ceil(dim(1))` = **15 .. 25** for 32 pairs at theta 160000
(pairs below 15 keep the plain frequency, above 25 are divided by 16, linear blend between); pinned by the sealed test at positions up to 65000.
Whether the branch is taken (h12): `Ernie5_5Config.rope_parameters = rope_scaling or ...` stores the checkpoint dict `{type: yarn, factor, beta_fast, beta_slow, original_max...}`
and `build_deepseek_v4_rope` reads `rope_parameters["rope_type"]`. Probe: with transformers 5.0.0 the dict is standardised to
`rope_type: yarn` (`rope_theta` is overridden by `compress_rope_theta` later); with transformers 4.57.1 it stays `type` only, so `["rope_type"]` would raise KeyError (a crash,
never a silent plain RoPE). The config also carries `hca_rope_type: yarn`. Either way a model that runs uses YaRN: **the program defaults to YaRN**; plain RoPE stays a
parameter (`ernie55_hca_rope=plain`, for the tests and for comparison). Not established: which transformers version vLLM v0.32.0 resolves (its requirements file was not read).
The pinned f32 table is computed in f32 (`arange(...) * inv_freq`); the ops form the angle in double and round cos and sin to f32, which differs from the model only at positions
where the f32 product loses bits (about 1e-4 rad at position 1e4, more beyond): recorded, not emulated.

Additive parameters: `ernie_rope_tail` gained `inverse` and the YaRN set, `ernie_hca_compress` the YaRN set (the compressed rows are rotated at the group start with the
same table); defaults are the previous plain behaviour, so the sealed 3-2a, 3-2b and 3-3b contracts are unaffected.

## Numerics and tolerances

q_prep: f32 RMS and rotation, one bf16 store. union: f32 scores, max, exp, weights, value sum, one bf16 store (the order of keys is compressed rows first, then the window, a fixed
summation order). wo_a: f32 rotation, bf16 rounding of the rotated value, f32 dot product. Compared with independent double-precision oracles in the sealed test: bf16 half-ulp
(2^-8 relative) plus a small absolute floor; bit-level agreement with FlashMLA is not claimed (m22 equivalent for HCA). The oracle shares no code with the ops; it shares the conventions
taken from the audit (stage roundings, f32 cos/sin from a double angle, GPT-J pairing).

## Not done here

The plugin program (3-4b): the compressor projections, the state ring, the cache groups and block tables, the gate, wo_b, mHC; block-fp8 `linear_o_group_proj` (read in place);
an fp8_ds_mla compressed row in the attention op (the op reads bf16 rows only; the layout was a parameter of the compressor); a window cache that frees old pages is supported
by -1 table entries but not exercised by an engine; a device kernel.
