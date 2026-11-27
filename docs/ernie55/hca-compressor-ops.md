# ERNIE 5.5 HCA compressor host ops (story 3-3b)

Two host operations in `libref/ref_ernie_hca.cpp`, the implementation of the semantics the 3-3 discovery report
(`docs/ernie55/hca-compression.md`) froze: `ernie_hca_state_write` and `ernie_hca_compress`. They are host-only libref
operations beside the 3-2a ones; `arch/ernie55` is not touched (plugin wiring is a later story), ABI 40 and the stock path are unchanged.

## What the ops do

* `ernie_hca_state_write(kv f32 [M, W], score f32 [M, W], ape f32 [ratio, W], positions i32 [M], slots i32 [M], state f32 INOUT [S, 2W])`, `W = coff * head_dim`
  (coff 2 at ratio 4, 1 at ratio 128): `state[slots[m]] = [kv[m] | score[m] + ape[positions[m] % ratio]]`. The APE row follows the **absolute**
  position, so the state does not depend on the chunking. Slot -1 skips (its position is not read); a repeated slot keeps the last row.
* `ernie_hca_compress(state f32 [S, 2W], gather_slots i32 [M, coff*ratio], positions i32 [M], out_slots i32 [M], norm_weight bf16 [D], out INOUT)`:
  a row computes only if `out_slots[m] >= 0` and `(positions[m] + 1) % ratio == 0` (k04). `gather_slots[m]` lists the state slots of the
  `coff*ratio` rows ending at the position, oldest first, `-1` for rows before position 0 (k06). Previous-group rows (window index `< ratio`, ratio 4 only)
  read channels `[0, D)` of kv and score, current-group rows `[D, 2D)` (k07). Per channel softmax in f32 (`exp`), RMSNorm in f32 with `eps` and the
  bf16 weight, rope = the trailing `rope_dim` channels rotated GPT-J interleaved at the group **start** `(position / ratio) * ratio`
  from the un-rounded f32 row (k11), cos and sin of the double angle rounded to f32, rounded once to bf16.
* Compressed slots are passed explicitly; the ops never derive a slot from a position. This leaves the gap of the report (no ABI KV group with one row
  per `ratio` tokens) to the plugin story: both options (RAD_KV_FULL with op-owned translation, or a per-token group) work with these ops.
* Validation covers every row with `out_slots >= 0` (rows with -1 are ignored): slot ranges, negative positions, a stored row with its whole window masked, a
  non-finite result, shapes, dtypes, `ratio` in {4, 128}, even `rope_dim <= head_dim`, `eps >= 0`, `theta > 0`. compress computes all rows before the first store, so a refusal stores nothing.

## Layouts and the k20 decision

`layout=bf16_full` (default): bf16 `[R, D]`, 1024 B per row at D 512. `layout=fp8_ds_mla`: u8 row of `nope + 2*rope + nope/quant_block + 1` bytes
(584 B at 448 + 128 + 7 + 1): E4M3 codes of the bf16-rounded nope channels, the rope channels as bf16 little endian, one UE8M0 byte (`exponent + 127`) per
64 nope channels, one zero pad byte. `scale = max(absmax / 448, 1e-4)`, `exponent = ceil(log2(scale))`, value times `2^-exponent`, clamp +-448 (the CPU/CuTeDSL rule, k09; the
Triton absmax floor, k10, is not implemented). This is the row-local form; paging the rows into blocks is a plugin concern.

Measured through the ops (acceptance test "layout error is recorded", ratio 4, head_dim 512, rope 64, 16 groups, relative L2 against the double-precision normalised row):

| part | bf16_full | fp8_ds_mla |
|---|---|---|
| nope (448 channels) | 1.69e-3 | 2.70e-2 |
| rope (64 channels) | 1.60e-3 | 1.60e-3 (the same bf16) |
| whole row | 1.68e-3 | 2.51e-2 |

FP8 costs 16.0 times the bf16 rounding on the nope channels, for 584 instead of 1024 bytes per row (0.57). These agree with the 3-3 fixture (2.66e-2, 1.67e-3, ratio 16).
**k20 is confirmed: the default stays full bf16**; fp8_ds_mla is a parameter. The numbers are synthetic (random state, norm weights 0.5 to 1.5); the real decision between
precision and the 1.75x row size needs rows from real hidden states (3-6) and a model-level error budget. Per-tensor FP8 rows (k13 layout (c)) were measured in story 3-3c on the same fixture and reference (below).

### Third layout: per-tensor FP8 (story 3-3c)

Layout (c) stores the whole 512-channel row (rope rotated in fp32) as bf16, times `1/fp8_scale`, clamped to +-448, converted to E4M3, with ONE per-tensor scale that belongs to the running
model (the fp8 KV cache scale), so it is not an op parameter here: it is measured in the sealed 3-3c test with an independent E4M3 oracle, at two scales that bracket it. Same 16 groups, ratio 4,
head_dim 512 as the table above (relative L2 against the double-precision normalised row):

| row part | bf16_full | fp8_ds_mla | per-tensor FP8, calibrated scale (absmax / 448 = 7.81e-3) | per-tensor FP8, scale 1.0 |
|---|---|---|---|---|
| nope (448) | 1.69e-3 | 2.70e-2 | 2.70e-2 | 2.70e-2 |
| rope (64) | 1.60e-3 | 1.60e-3 | 2.62e-2 | 2.63e-2 |
| whole row | 1.68e-3 | 2.51e-2 | 2.69e-2 | 2.69e-2 |
| bytes per row | 1024 | 584 | 512 (+ scale) | 512 (+ scale) |

The E4M3 mantissa (3 bits) sets the error whatever the scale on this data (the values lie between 2^-8 and 2^2 of the range, nothing clips, 6 of 8192 values flush to zero at scale 1.0); it also puts the rope channels
at FP8 precision, which fp8_ds_mla avoids. **k20 stays: full bf16 is the default**; per-tensor FP8 saves 12 percent over fp8_ds_mla for 7 percent more error and drops the rope precision. (Numbers synthetic; real rows
are still the final word, 3-6.)

## Operand rules (story 3-3c)

`ernie_hca_state_write` refuses (RAD_E_INVAL, before any store) a `state` that shares a byte with `kv`, `score`, `ape`, `positions` or `slots`; `ernie_hca_compress` refuses an `out` that shares a byte with any
input. The slots and positions were validated once and then read again while rows were being stored, so an overlapping output could change a validated index under the op (review of 3-3b).
`ernie_hca_compress` judges finiteness on the bf16-ROUNDED nope and rope values: a finite f32 value above about 3.396e38 rounds to bf16 infinity and refuses the launch (no store) in both layouts; a value that rounds to
the largest finite bf16 (3.3895e38) still stores.

## Rope: what is and is not established (k22)

The op rotates with **plain RoPE** at the `theta` parameter (production: `compress_rope_theta` 160000), angle `start * theta^(-2i/rope_dim)`. The report does not establish that the HCA layers
take the YaRN branch (config `rope_scaling.type = yarn`, factor 16; the vLLM reader looks for `rope_type`): that is k22 and it stays **unresolved**. If YaRN applies, the compressed-row rope angles of production
rows differ from these; the anchor (group start) is unaffected. Adding YaRN means a new optional parameter set (factor, original length, beta fast/slow, mscale), not a change of the existing ones.

## Tolerance and what is not claimed

Compared against an independent double-precision oracle (sealed test): bf16 rows within half a bf16 ulp of the exact row and, for at least 98 % of channels,
equal to the stage-rounded reference; FP8 scale bytes equal, codes within one E4M3 step (at least 97 % equal). Bit-level agreement with CUDA (CuTeDSL `exp2`, `rsqrt` fastmath, 8-warp combine) is not
claimed (k19); the scale-floor behaviour for degenerate rows (k21) is the CPU one.

## Not done here

Plugin wiring and cache declarations (3-4+), the projections that produce kv and score, the indexer at ratio 4, visibility of compressed rows to attention, YaRN (k22), the bit-level CUDA comparison (k19).
