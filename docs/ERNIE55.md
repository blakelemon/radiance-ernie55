# ERNIE 5.5 Flash on Radiance

An unofficial community port of ERNIE 5.5 Flash (HuggingFace `model_type: ernie5_5_moe`) to Radiance,
built on v2.2.0. The architecture is `arch/ernie55/`; the plugin claims the architecture id `ernie55moe`
(`core/format/checkpoint.cpp` drops the underscores of `model_type` to get it). Model weights are not
included in this tree and carry their own licence.

## Status

What works, on the rig the numbers below come from:

- chat completions with tools and reasoning;
- a 1,048,576-token context;
- 8 concurrent sequences;
- prefix caching across turns;
- the DSpark block-diffusion drafter, at speculative depth 7 with an adaptive window;
- an fp8 latent KV cache.

What has **not** been verified on the final commit:

- long-context retrieval and perplexity checks at 30K and 150K tokens;
- out-of-tree plugin loading;
- a KL-divergence/replay audit;
- the case where a cached prefix is evicted and then missed;
- an independent deep review of the last commits.

Hardware other than eight gfx1201 cards has not been tried.

One recipe in the tree is a plan rather than a conversion: `data/recipes/ernie55-iqr-v4.recipe` says so
in its own header (the plugin refuses the encodings it targets, and no ERNIE calibration Hessians
exist), and `docs/ernie55/iqr-v4-gap.md` records the gap.

## Layout

- `arch/ernie55/` -- the plugin: `ernie55.cpp` (config, `declare`, the forward graph), `ernie55_pins.h`
  (the row-band pins) and its `CMakeLists.txt`.
- `libr4d/r4d_ernie_*.hip` -- the device ops (common, ops, MLA, HCA, mHC, DSpark, small, pins, verdict);
  `libref/ref_ernie_*.cpp` holds the CPU reference each device row is judged against.
- `tools/ernie55/` -- the conversion audit and the probes: `audit_checkpoint.py`, `device_dispatch.py`,
  `attention_reference.py`, `hca_compression.py`, `fp8_stream.py`, the `iqr_*.py` planners and
  `iqr_probe.cpp`.
- `tests/ernie55/` -- the node tests and the synthetic checkpoints they draw on (`support/`,
  `support/synth/`); `tests/r4d_emu/test_ernie_*.cpp` the emulator tests; `tests/ernie_pins_test.cpp`,
  `tests/ernie_dspark_shape_parity_test.cpp` and `tests/preview_mxfp4_test.cpp` the host tests.
- `data/recipes/ernie55-*.recipe` -- `ernie55-preview`, `ernie55-preview-dspark`, `ernie55-weights`
  and `ernie55-iqr-v4`.
- `docs/ernie55/` -- the discovery and audit data the port was built from: the geometry, the attention
  and feed-forward notes, the checkpoint audit, the device dispatch and the memory plans.
- `core/`, `abi/` and the plugin loader carry the hooks the architecture needs (the declare-time
  builder, the checkpoint reader, the registry).

## Converting a checkpoint

`rad-convert` reads the HuggingFace directory and writes a `.rad` container. The ERNIE rules are
recipes, and the plugin's build-up switches are `--set` keys. The DSpark recipe's header gives the
whole invocation:

```
rad-convert /models/ernie55-flash-preview -o <out>.rad --tokenizer <ckpt>/tokenizer.json \
    --reuse <base>.rad --recipe data/recipes/ernie55-preview-dspark.recipe \
    --set ernie55_quant_forward=1 --set ernie55_dspark=1 --max-spec 7 --kv-cache-dtype bf16 \
    --max-tok 2048 --max-ctx 200000
```

`--max-tok` x `--max-ctx` must stay `<= 2^30` (the indexer score plane) and `--max-tok` must be at
least 1792 (DSpark: 256 sequences x block 7). Without the drafter, `data/recipes/ernie55-preview.recipe`
is the same trunk with no `dspark.*` rules.

The `--set` keys the plugin reads (`arch/ernie55/ernie55.cpp`), in the order the graph is built up:

| switch | what it does |
|---|---|
| `ernie55_weights_only=1` | declares the dense and shared tensors with their checkpoint names; the container cannot serve |
| `ernie55_moe_weights=1` | adds the routed experts, the routers and the latent projections; needs `ernie55_weights_only=1` |
| `ernie55_ffn_forward=1` | adds the dense and shared feed-forward half of every layer; needs `ernie55_weights_only=1` |
| `ernie55_moe_forward=1` | adds the routed latent-MoE block; needs `ernie55_weights_only=1`, `ernie55_moe_weights=1` and `ernie55_ffn_forward=1` |
| `ernie55_target_output=1` | ends the program in the language-model head instead of the pooled mean |
| `ernie55_attention_forward=1` | adds the MLA-DSA attention branch; needs `ernie55_weights_only=1` and `ernie55_ffn_forward=1` |
| `ernie55_hca_forward=1` | adds the HCA attention branch; needs `ernie55_attention_forward=1` |
| `ernie55_quant_forward=1` | the quantised serving forward, and the switch `--kv-cache-dtype fp8` is allowed under |
| `ernie55_dspark=1` | converts the drafter (`model.mtp.*`) into the container; it runs under `ernie55_quant_forward=1` |

## Serving

The configuration the measured numbers come from. The chat template it names is
`deploy/chat-templates/ernie55.jinja` in this tree (system, tools, reasoning effort and
`enable_thinking`), mounted into the container at `/etc/radiance/ernie55.jinja`:

```yaml
models:
  ernie55-flash-preview:
    model: /path/to/ernie55-flash-preview-dspark.rad
    override-chat-template: /etc/radiance/ernie55.jinja
    tp: 8
    kv-cache-dtype: fp8
    embedding-placement: vram
    max-model-len: 1048576
    max-num-batched-tokens: 2048
    max-num-seqs: 8
    num-speculative-tokens: 7
    prefill-interleave: auto:0.5
    draft-window: adaptive
environment:
  RADIANCE_ERNIE55_HCA_CMP_LAYOUT: "auto"
  RADIANCE_ERNIE55_IDX_TILE: "16384"
  RADIANCE_ERNIE55_DSPARK_KV_WINDOW: "1"
  R4D_AR8_FORM: "ring"
```

The environment switches, each read in the source named:

| switch | values | default | what it does |
|---|---|---|---|
| `RADIANCE_ERNIE55_HCA_CMP_LAYOUT` | `auto`, `block`, `token` | `auto` (container key `ernie55_hca_cmp_layout`) | the compressed HCA group's layout: one row per `period`-token block, or one slot a token. `auto` is `block` where `head_dim` is a multiple of `2 * period`, `token` otherwise (`arch/ernie55/ernie55.cpp`) |
| `RADIANCE_ERNIE55_IDX_TILE` | `0` or `1..2^24` columns | `0`, untiled (container key `ernie55_idx_tile`) | serve-time override of the indexer score plane's tiling, for A/B runs on one container |
| `RADIANCE_ERNIE55_DSPARK_KV_WINDOW` | `1` | off | stores the drafter's KV group as a sliding window of the drafter's attention window instead of a full-context group |
| `RADIANCE_ERNIE55_KV_LATENT_FMT` | `ds_mla`, `bf16id`, `i8gN`, `i10gN`, `i12gN` (N one of 128, 64, 32, 16) | `ds_mla` | the latent cache's row form under `--kv-cache-dtype fp8`: E4M3 codes, a lossless bf16 diagnostic row, or integer codes with one exact f32 scale per N latent values |
| `RADIANCE_ERNIE55_AR_OVERLAP` | `1` | off | overlaps the collectives of a quantised prefill step at tensor parallel 2 or more with the compute between them |
| `RADIANCE_ERNIE55_AR_OVERLAP_MIN` | an integer `>= 2` | `1280` | the fewest rows a prefill step must have before the overlap splits it |
| `R4D_AR8_FORM` | `lockstep`, `ring`, `split` | `lockstep` | the two-shot all-reduce's form when the launch names none (`libr4d/r4d_ar_entry.hip`) |

## Measured performance

One machine, one set of runs: eight gfx1201 cards, tensor-parallel 8, about 29.9 GiB of weights and
buffers held per card, with the configuration above.

- Prefill, single stream: about 2,770 tokens/s at a 1.6K prompt, 2,490 at 8K, 2,150 at 30K.
- Decode, single stream: about 70-80 tokens/s on prose, about 130 on code, about 220 on highly
  predictable text. The drafter's acceptance on prose is about 35-50%.
- A cold 140K-token prompt takes about 105 s to first token; the next turn with the prefix cached
  starts in 2-4 s.
- The KV pool with these settings holds 1,483,008 tokens per card.
- `RADIANCE_ERNIE55_DSPARK_KV_WINDOW=1` stores the drafter's KV as a window group: its start-up
  reservation fell from 3.09 GiB to 9 MiB per card, and the KV pool grew by about 61% at an otherwise
  equal configuration. Output was token-identical with the switch on and off in our runs (50 of 50
  prompts).

## Known behaviour and limits

- While one request prefills, decoding of the other sequences pauses for each prefill batch (about 1 s
  at batch 2,048), so concurrent streams look bursty.
- The drafter's drafted/accepted counters are not bit-stable from run to run.
- Only eight gfx1201 cards have been tried.

## Tests

The ERNIE tests registered in `tests/CMakeLists.txt`:

- unit and host: `ernie_hca_origin_test`; and, with the libr4d host sources,
  `ernie_dspark_shape_parity_test`
  and `ernie_pins_test`.
- emulator (`r4d_emu`): `r4d_emu_ernie_hca_origin`, `r4d_emu_ernie_hca_block`,
  `r4d_emu_ernie_hca_state`, `r4d_emu_ernie_hca_pipe`, `r4d_emu_ernie_dspark`,
  `r4d_emu_ernie_dsa_tiled`, `r4d_emu_ernie_dsa_select`, `r4d_emu_ernie_latent_fp8`,
  `r4d_emu_ernie_latent_i8`, `r4d_emu_ernie_latent_i10`, `r4d_emu_ernie_latent_i12`,
  `r4d_emu_ernie_latent_bf16id`, `r4d_emu_mhc_pin_default`, `r4d_emu_mhc_pin_pinned`,
  `r4d_emu_pair_pin`, and the `r4d_emu_row_identity_<family>` cases.
- lint: `ernie_no_device_globals`, `ernie_arg_decl`, `ernie_lint_fixture_globals_bad`,
  `ernie_lint_fixture_globals_good`, `ernie_lint_fixture_args_bad`, `ernie_lint_fixture_args_good`.

The node tests in `tests/ernie55/` (`*.test.cjs`, written against `node:test`) are not registered in
CMake; they draw on the synthetic checkpoints in `tests/ernie55/support/`.

## Licence

Apache-2.0, like upstream Radiance. See `LICENSE` and `NOTICE`.
