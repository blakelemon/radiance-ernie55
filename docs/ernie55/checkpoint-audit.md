# Checkpoint metadata audit (`tools/ernie55/audit_checkpoint.py`)

Offline account of an ERNIE 5.5 checkpoint **before** conversion, from metadata alone. Python
standard library only; no network, no weight loading.

```
python3 tools/ernie55/audit_checkpoint.py --metadata-dir /models/ERNIE-5.5-Flash-Base
```

`--metadata-dir` is a directory holding `config.json` and `model.safetensors.index.json` -- the
config and tensor-index entries of the six files `docs/ernie55/metadata-manifest.json` pins.
Nothing else is read: no shard is opened or even resolved to a path, so the command works against
a metadata-only directory, and shard files named in the index do not have to exist. Input files
are never written.

Valid metadata prints **one** JSON object on stdout and exits `0`. Anything else prints one
`audit_checkpoint.py: error: ...` line on stderr, prints no JSON, and exits `2`; the exit code is
authoritative (`--metadata-dir` misuse is also `2`).

## Output

| Field | Meaning |
| --- | --- |
| `main_layers`, `draft_layers` | `num_hidden_layers`, `num_nextn_predict_layers` |
| `attention.hca_layers` | Zero-based **main** layer indices whose `compress_ratios` entry is `128` (HCA) |
| `attention.mla_dsa_layers` | Zero-based **main** layer indices whose entry is `-2` (MLA-DSA) |
| `attention.draft_schedule` | The trailing `draft_layers` schedule entries, kept apart from the main layers |
| `geometry.hidden_size`, `geometry.head_dim`, `geometry.hc_mult` | `hidden_size`, `head_dim`, `hc_mult` as configured |
| `moe.dense_layers`, `moe.routed_layers` | Main layers split by the `first_k_dense_replace` prefix |
| `moe.routed_experts`, `moe.top_k`, `moe.latent_size` | `n_routed_experts`, `num_experts_per_tok`, `moe_latent_size` |
| `checkpoint.tensor_count`, `checkpoint.shard_count` | Entries in `weight_map`; unique shard filenames in it |
| `checkpoint.declared_bytes` | `metadata.total_size` as declared by the index, or `null` |
| `checkpoint.tensor_shapes_available` | Always `false` |
| `special_tokens.eos_token_id` | `eos_token_id`, or `null` when the config declares none |

A tensor index lists names and shard membership; it carries no shapes. The audit therefore never
infers a shape, a dtype or a per-tensor size, states `tensor_shapes_available: false`, and repeats
the warning on stderr. The layer split is read entry by entry from `compress_ratios` -- never
derived from a modulo period -- so a checkpoint with an irregular schedule is reported as it is
configured. `eos_token_id` is reported as a bare id: it says nothing about chat templates or any
prefix insertion, which conversion has to establish from the tokenizer files.

## What is refused

`model_type` other than `ernie5_5_moe`; a missing or non-object metadata file; anything malformed
in the fields above -- a boolean or float where an integer is required, a non-positive main-layer
count, hidden width, head dimension, `hc_mult`, latent width, routed or shared expert count, top-k,
a negative draft count or dense prefix, top-k above the routed expert count, a dense prefix longer
than the target, a negative or boolean `total_size`, a non-integer `eos_token_id`, an empty
`weight_map`, a `weight_map` value that is not a string, or a shard name that is not a plain
relative `*.safetensors` filename (no slash or backslash, no URL scheme or drive letter, no `.` /
`..` traversal, no control characters, no leading `~`, no surrounding whitespace); and any
`compress_ratios` whose length is not `main_layers + draft_layers` or whose entries are not
`128` / `-2` -- the only modes this first supported variant accepts.

Tests: `tests/ernie55/audit.test.cjs` (repository gate) and the story's sealed
`audit.acceptance.cjs`.
