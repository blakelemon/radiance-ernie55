# ERNIE architecture discovery (`arch/ernie55`)

Radiance identifies an ERNIE 5.5 checkpoint from its converted `config.json` metadata. The plugin
`ernie55` (`arch/ernie55/ernie55.cpp`, ABI 40, unquantised) claims the architecture id
`ernie55moe`, which is `model_type: ernie5_5_moe` without underscores as `rad-convert` derives it.
The pinned block-FP8 release reaches it through the family's single-plugin rule, so one plugin
serves both the FP8 source and a BF16 synthetic fixture.

It validates, entry by entry and never from a modulo period:

- `num_hidden_layers` (target) and `num_nextn_predict_layers` (draft) as separate counts; the
  probe reports the draft count as `draft_depth` (not fixed);
- `compress_ratios` has `target + draft` entries, each `128` (HCA) or `-2` (MLA-DSA);
- `first_k_dense_replace` is at most the target layer count (the first dense layers lead);
- positive `hidden_size`, `head_dim`, `hc_mult`, `moe_latent_size`, `n_routed_experts`,
  `n_shared_experts`, `num_experts_per_tok` (at most the routed experts);
- `model_type` is `ernie5_5_moe` (relevant when the plugin is forced with `--arch`).

A valid config prints one `radiance: ernie55: config ok: ...` line on stderr (identity, layer
counts, family counts, geometry and the `H`/`M` schedule, `|` before the draft entries). It is
then refused with `RAD_E_UNSUPPORTED` because no forward block exists yet; later stories replace
that refusal. An invalid config is refused with `radiance: ernie55: invalid ERNIE 5.5 config: ...`
naming the offending key. The checks mirror `tools/ernie55/audit_checkpoint.py`.

Limits: the converter flattens JSON numbers to text, so `43.0` reads as `43`; the audit tool is
the strict-JSON check. Quantisation, tensor names and shapes are out of scope until 2-5.
