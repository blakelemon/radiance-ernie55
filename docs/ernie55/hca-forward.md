# HCA attention program (story 3-4b)

Opt-in: `rad-convert --set ernie55_hca_forward=1` on top of `ernie55_attention_forward=1` (which needs `ernie55_weights_only=1` and `ernie55_ffn_forward=1`; `ernie55_target_output=1`,
`ernie55_moe_weights=1` + `ernie55_moe_forward=1` as before). Without `ernie55_hca_forward` an HCA main layer is still refused by name exactly as in 3-2b (the sealed 3-2b contract asserts
that message, so the HCA branch is a second key rather than a change of the first). Host (libref) backend, single rank. MLA-DSA and HCA layers interleave per the `compress_ratios` schedule: each
layer runs its own family's attention branch, then its feed-forward half. Optional keys: `ernie55_attention_block=N` (KV page size, default 64), `ernie55_hca_cmp_layout=auto|block|token` (the compressed group's layout; auto, the default, is block where `head_dim % (2 * period) == 0`, as in the release, and token otherwise; below), `ernie55_hca_period=4|128` (the compression
period the program uses; default 128 = the schedule, 4 is the other legal period k01 and exists so that small checkpoints and the overlap lanes can be tested; the config still says 128),
`ernie55_hca_rope=yarn|plain` (default yarn). The MLA keys (`hybrid_mla_*`, `index_*`) stay required whenever `ernie55_attention_forward=1`: one config serves both families.

## One HCA layer, in the order the program runs it

```
x, post, comb = ernie_mhc_pre_rms(h; attn_hc, input_layernorm gain)               attn-branch mHC, fused norm
qa = rmsnorm(linear_q_down_proj x) -> qn                                        q latent, normed (the gate reads it)
q  = linear_q_up_proj qn           [T, heads * D];  ernie_hca_q_prep(q)          weightless per-head RMS over D, YaRN rotary on the trailing rope_dim, one bf16 rounding
kv = rmsnorm(linear_kv_proj x, kv_layernorm);  ernie_rope_tail(kv, 1, D, YaRN)  K = V row, rotated at its own position
kv_store -> window cache
ckv = gemm_nt_f32(compressor.linear_wkv x), csc = gemm_nt_f32(compressor.linear_wgate x)   [T, coff * D] f32
ernie_hca_state_write(ckv, csc, ape, positions, state slots)                     state row [kv | score + APE[pos % period]]
ernie_hca_gather_slots(positions, req, state block table) -> gather               the coff * period state slots a completion reads
ernie_hca_compress(state, gather, positions, compressed slots, norm)             at (pos + 1) % period == 0: pool, RMS, rope at the group START, bf16 row
o  = ernie_hca_union_attn(q, window cache, compressed cache, ..., sink)          one softmax over [complete groups | window] and the sink, K = V, scale D^-0.5
om = ernie_vha_mix(o, U, V, fold = bf16)                                          head axis (commutes with the dim-axis inverse rope)
wa = ernie_hca_wo_a(om, positions, linear_o_group_proj)                           inverse rotary at the query position, grouped wo_a
og = act_mul(sigmoid, gate_proj qn, wa)                                           the gate multiplies the wo_a output (h16)
y  = o_proj og;  h' = mhc_post(y, h, post, comb);  then the FFN half
```

## Choices

* **Cache layout: three per-token `RAD_KV_FULL` groups bound to the HCA layers**, one block size (the `ernie55_attention_block`). `kv_hca_window`: bf16, row = the rotated K = V row (`D`),
  written by `kv_store`. `kv_hca_state`: f32, row = `[kv | score + APE]` (`2 * coff * D` f32), written by `ernie_hca_state_write` at the batch's slot mapping, so a state row is addressed by its
  position and **no ring exists**: the lifetime constraint of 3-3b (ring >= `coff * period` rows plus the current chunk) is met by keeping every row, a chunk split cannot change a row and a
  completion reads its window through the block table (`ernie_hca_gather_slots`). `kv_hca_compressed`: bf16, row = the compressed row. Two layouts (`ernie55_hca_cmp_layout`):
  **`block` (default where head_dim divides by 2 * period, kv-cmpblock): one row per `period`-token block.** The group's block size is `period` (it is claimed by the union attention, which carries `block_size = period` and is declared after the group, not by the gather, which names the state block size)
  and its page is exactly one row: the decl carries `head_dim / (2 * period)` so the engine's pool view `[pages, 1, period, D / period]` is `D` contiguous elements a page (production: block 128, 1 KiB a layer a page, 8 B a token and layer).
  Row `j` is completed by token `(j + 1) * period - 1`, the last of block `j`, and is stored at the slot of that token (`page * period + period - 1`, the engine's own slot mapping) and read back as page `cmp_table[req][j]`
  (`ernie_hca_union_attn cmp_layout=block`). A cached block of the group is a function of the tokens through its last one, as the prefix cache hashes it: a block holds the one row its own `period` tokens complete.
  Prefix entries span `lcm` of the groups' block sizes (128 tokens in production), a chunk quantum is the largest block (`--max-num-batched-tokens` must be at least 128). Only addresses moved against `token`: outputs are bit-identical.
  **`token`: the CPU-reference form this section first used**, kept for A/B: the row is **stored at the slot of the token that completed the group**
  (`(j + 1) * period - 1`) in a per-token group at the block size of the other HCA groups and read back through its block table (`cmp_layout=token`). The ABI has no kind with one row per `period` tokens (report 3-3), so the
  first choice was between op-owned translation through a block table with rows at dense logical ids `0, 1, 2, ...` and one row per token. Dense ids break the prefix cache: a cached block of the compressed group would hold
  rows computed from `period` times more tokens than the block's own, so a request sharing only the first block's tokens would reuse rows derived from tokens it does not have. The per-token anchor makes a block's
  content a function of the tokens up to its last one, which is exactly what the prefix cache hashes (tested: a prompt sharing a 16-token prefix and then differing, and a repeated prompt); the `block` layout keeps that property
  with a block of `period` tokens. The price of `token` is memory: per HCA layer and token a compressed slot `D` bf16 (production 1 KiB, `period - 1` of `period` slots never written), 37 KiB a token for 37 layers.
  The window group is `RAD_KV_WINDOW` and the state group a windowed f32 group since a2d322b5.
* **YaRN is the default** (k22 resolved at the pinned sources, see `hca-attention-ops.md`); `ernie55_hca_rope=plain` is the parameter. The rotary of q, kv, the compressed rows (at the group start) and
  the inverse rotary of the output all use the one table: theta `compress_rope_theta`, factor / beta_fast / beta_slow / original length from `rope_scaling.*`. The log records which.
* **State and compressed caches are f32 / bf16 and not quantised**: `fp8_ds_mla` rows stay a compressor parameter (k20 confirmed bf16 on synthetic rows; real rows need 3-6).
* **VHA fold bf16** as in 3-2b; **gate**: `act_mul(sigmoid)` on the wo_a output (width `o_groups * o_lora_rank`), computed from the normed q latent.
* **Block-fp8 projections are refused by name** with the exact dequantisation rule: `q_a`, `q_b`, `kv_a_mqa`, `gate`, `o_group`, `output` (the release stores the first five and `o_proj` block-fp8; the
  compressor linears and the sink, VHA, norms are BF16/F32). Compressor projections are read as f32 (exact widening of the stored BF16), as the routers are, because `gemm_nt_f32` serves f32 weights.
* **Positions** are the batch's absolute request-local positions for every rotary and for the window, so chunked prefill equals one-shot prefill (tested bit for bit).

## Tested (sealed contract for the HCA attention program)

Engine against an independent double-precision whole-layer oracle (|dlogp| <= 1e-4; every wrong equation of a 24-item variant list must be at least 2e-3 away and the engine must not follow it):
period 4 / window 8 (two HCA layers), period 4 with an MLA-DSA layer between two HCA layers, period 128 / window 128; production boundary prompts of 126, 127, 128, 129, 255, 256, 257 tokens
(the group is visible exactly at position 127) and decode through it; chunk splits 127/128/64/100 at period 128 and 8/12 at period 4 equal to one-shot bit for bit; two concurrent sequences, repeated
prompt, a partly shared prefix; declaration log and refusals by name; the block-fp8 projection refusal and exact cast rule.

## Not done

Production-size state/window/compressed caches (see above), block-fp8 read in place, the draft layer, tensor parallelism, `fp8_ds_mla` rows in the attention op, device kernels, the real 43-layer
checkpoint (headers of the HCA tensors were checked against the declared shapes, not converted), real-activation agreement with the reference runtime.
