#!/usr/bin/env python3
"""audit_checkpoint.py --metadata-dir DIR -- ERNIE 5.5 checkpoint metadata, offline.

    python3 tools/ernie55/audit_checkpoint.py --metadata-dir /models/ERNIE-5.5-Flash-Base

Reads only the two metadata files the tensor index and the model config are made of --
`config.json` and `model.safetensors.index.json` -- and prints one JSON report: the target
layers, the attention schedule split from the trailing draft entries, the configured geometry,
the MoE layer split, and the shard inventory. It never follows a shard path, never opens a
shard, never contacts the network and never writes an input file, so it runs against a metadata
directory holding nothing but those two files.

The tensor index lists tensor names and shard membership; it carries no tensor shapes, so the
report says so with `checkpoint.tensor_shapes_available: false` and a note on stderr rather
than inventing shapes. `checkpoint.declared_bytes` is whatever the index declares under
`metadata.total_size`, or null when the index declares none.

Valid metadata exits 0 with the JSON object on stdout. Invalid metadata exits 2 with a message
on stderr and no JSON. See docs/ernie55/checkpoint-audit.md.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys

CONFIG_NAME = "config.json"
INDEX_NAME = "model.safetensors.index.json"
MODEL_TYPE = "ernie5_5_moe"
HCA_MODE = 128
MLA_DSA_MODE = -2
SUPPORTED_MODES = (HCA_MODE, MLA_DSA_MODE)
SHARD_SUFFIX = ".safetensors"


class AuditError(Exception):
    """Metadata that cannot be audited; the message is the operator's whole error."""


def _load(directory: pathlib.Path, name: str) -> dict:
    path = directory / name
    if not path.is_file():
        raise AuditError(f"{name} is missing: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise AuditError(f"{name} cannot be read as UTF-8 text: {path}: "
                         f"{getattr(exc, 'strerror', None) or exc}")
    try:
        doc = json.loads(text)
    except ValueError as exc:
        raise AuditError(f"{name} is not valid JSON: {path}: {exc}")
    if not isinstance(doc, dict):
        raise AuditError(f"{name} is not a JSON object: {path}")
    return doc


def _integer(source: dict, field: str, what: str, lowest: int) -> int:
    if field not in source:
        raise AuditError(f"{what} is missing {field}")
    value = source[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise AuditError(f"{what} {field} must be an integer, got {json.dumps(value)}")
    if value < lowest:
        raise AuditError(f"{what} {field} must be an integer >= {lowest}, got {value}")
    return value


def _shard_problem(shard: str) -> str | None:
    if not shard.endswith(SHARD_SUFFIX) or len(shard) == len(SHARD_SUFFIX):
        return f"a shard filename must end in {SHARD_SUFFIX} and have a name before it"
    if shard != shard.strip():
        return "a shard filename must not have leading or trailing whitespace"
    if any(sep in shard for sep in {"/", "\\", os.sep, os.altsep} - {None, ""}):
        return "a shard filename must not contain a path separator"
    if ":" in shard:
        return "a shard filename must not contain a URL scheme or drive letter"
    if any(ord(c) < 0x20 or ord(c) == 0x7F for c in shard):
        return "a shard filename must not contain control characters"
    if shard.startswith("~"):
        return "a shard filename must not start with ~"
    return None


def _schedule(config: dict, main_layers: int, draft_layers: int
              ) -> tuple[list[int], list[int], list[int]]:
    if "compress_ratios" not in config:
        raise AuditError(f"{CONFIG_NAME} is missing compress_ratios")
    schedule = config["compress_ratios"]
    if not isinstance(schedule, list) or not schedule:
        raise AuditError(f"{CONFIG_NAME} compress_ratios must be a non-empty array of "
                         f"attention modes")
    expected = main_layers + draft_layers
    if len(schedule) != expected:
        raise AuditError(
            f"{CONFIG_NAME} compress_ratios has {len(schedule)} entries, but num_hidden_layers "
            f"({main_layers}) plus num_nextn_predict_layers ({draft_layers}) is {expected}"
        )
    modes: list[int] = []
    for index, mode in enumerate(schedule):
        if isinstance(mode, bool) or not isinstance(mode, int):
            raise AuditError(f"{CONFIG_NAME} compress_ratios[{index}] must be an integer "
                             f"attention mode, got {json.dumps(mode)}")
        if mode not in SUPPORTED_MODES:
            raise AuditError(f"{CONFIG_NAME} compress_ratios[{index}] is {mode}; this supported "
                             f"variant allows only {HCA_MODE} (HCA) and {MLA_DSA_MODE} (MLA-DSA)")
        modes.append(mode)
    hca = [i for i in range(main_layers) if modes[i] == HCA_MODE]
    mla_dsa = [i for i in range(main_layers) if modes[i] == MLA_DSA_MODE]
    return hca, mla_dsa, modes[main_layers:]


def audit_checkpoint(metadata_dir: pathlib.Path) -> dict:
    """Audit an ERNIE 5.5 metadata directory (`config.json` + `model.safetensors.index.json`).

    Returns the report dict; raises AuditError for anything that cannot be audited.
    """
    directory = pathlib.Path(metadata_dir)
    if not directory.is_dir():
        raise AuditError(f"metadata directory is not a directory: {directory}")
    config = _load(directory, CONFIG_NAME)
    index = _load(directory, INDEX_NAME)

    if config.get("model_type") != MODEL_TYPE:
        raise AuditError(f"{CONFIG_NAME} model_type must be \"{MODEL_TYPE}\", "
                         f"got {json.dumps(config.get('model_type'))}")

    main_layers = _integer(config, "num_hidden_layers", CONFIG_NAME, 1)
    draft_layers = _integer(config, "num_nextn_predict_layers", CONFIG_NAME, 0)
    hidden_size = _integer(config, "hidden_size", CONFIG_NAME, 1)
    head_dim = _integer(config, "head_dim", CONFIG_NAME, 1)
    hc_mult = _integer(config, "hc_mult", CONFIG_NAME, 1)
    latent_size = _integer(config, "moe_latent_size", CONFIG_NAME, 1)
    routed_experts = _integer(config, "n_routed_experts", CONFIG_NAME, 1)
    _integer(config, "n_shared_experts", CONFIG_NAME, 1)
    top_k = _integer(config, "num_experts_per_tok", CONFIG_NAME, 1)
    if top_k > routed_experts:
        raise AuditError(f"{CONFIG_NAME} num_experts_per_tok ({top_k}) is greater than "
                         f"n_routed_experts ({routed_experts})")
    dense_prefix = _integer(config, "first_k_dense_replace", CONFIG_NAME, 0)
    if dense_prefix > main_layers:
        raise AuditError(f"{CONFIG_NAME} first_k_dense_replace ({dense_prefix}) is greater than "
                         f"num_hidden_layers ({main_layers})")
    eos_token_id = None
    if "eos_token_id" in config:
        eos_token_id = _integer(config, "eos_token_id", CONFIG_NAME, 0)

    hca_layers, mla_dsa_layers, draft_schedule = _schedule(config, main_layers, draft_layers)

    if "weight_map" not in index:
        raise AuditError(f"{INDEX_NAME} is missing weight_map")
    weight_map = index["weight_map"]
    if not isinstance(weight_map, dict) or not weight_map:
        raise AuditError(f"{INDEX_NAME} weight_map must be a non-empty object mapping tensor name "
                         f"to shard filename")
    shards: set[str] = set()
    for tensor, shard in weight_map.items():
        if not isinstance(shard, str):
            raise AuditError(f"{INDEX_NAME} weight_map[{json.dumps(tensor)}] must name a shard "
                             f"filename, got {json.dumps(shard)}")
        problem = _shard_problem(shard)
        if problem:
            raise AuditError(f"{INDEX_NAME} weight_map[{json.dumps(tensor)}] has an unsafe shard "
                             f"filename {json.dumps(shard)}: {problem}")
        shards.add(shard)

    declared_bytes = None
    if "metadata" in index:
        metadata = index["metadata"]
        if not isinstance(metadata, dict):
            raise AuditError(f"{INDEX_NAME} metadata must be an object")
        if "total_size" in metadata:
            declared_bytes = _integer(metadata, "total_size", f"{INDEX_NAME} metadata", 0)

    return {
        "main_layers": main_layers,
        "draft_layers": draft_layers,
        "attention": {
            "hca_layers": hca_layers,
            "mla_dsa_layers": mla_dsa_layers,
            "draft_schedule": draft_schedule,
        },
        "geometry": {"hidden_size": hidden_size, "head_dim": head_dim, "hc_mult": hc_mult},
        "moe": {
            "dense_layers": list(range(dense_prefix)),
            "routed_layers": list(range(dense_prefix, main_layers)),
            "routed_experts": routed_experts,
            "top_k": top_k,
            "latent_size": latent_size,
        },
        "checkpoint": {
            "tensor_count": len(weight_map),
            "shard_count": len(shards),
            "declared_bytes": declared_bytes,
            "tensor_shapes_available": False,
        },
        "special_tokens": {"eos_token_id": eos_token_id},
    }


def _render(report: dict) -> str:
    """Pretty JSON, with a layer list kept on one line so a 43-layer schedule stays readable."""
    text = json.dumps(report, indent=2)
    return re.sub(r"\[\s*((?:-?\d+,?\s*)+)\]", lambda m: "[" + " ".join(m.group(1).split()) + "]",
                  text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="audit_checkpoint.py",
        description="Audit ERNIE 5.5 checkpoint metadata offline: layers, schedule, geometry, "
                    "shards. Reads config.json and model.safetensors.index.json only; never "
                    "opens a shard.",
    )
    parser.add_argument("--metadata-dir", required=True, type=pathlib.Path, metavar="DIR",
                        help=f"directory holding {CONFIG_NAME} and {INDEX_NAME}")
    args = parser.parse_args(argv)
    try:
        report = audit_checkpoint(args.metadata_dir)
    except AuditError as exc:
        print(f"audit_checkpoint.py: error: {exc}", file=sys.stderr)
        return 2
    print("warning: tensor shapes are unavailable from checkpoint metadata; "
          "tensor_shapes_available is false", file=sys.stderr)
    print(_render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
