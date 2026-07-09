#!/usr/bin/env python3
"""Merge trained mapping_enc2llm into a full XBridge checkpoint for inference."""

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_checkpoint",
        default=os.path.join(ROOT, "model/XBridge-base"),
        help="Full checkpoint used as skeleton (provides frozen weights).",
    )
    parser.add_argument(
        "--mapping_pt",
        required=True,
        help="Path to mapping_enc2llm.pt from train_encoder_only.py",
    )
    parser.add_argument(
        "--mapping_embed_pt",
        default="",
        help="Path to mapping_embed.pt (embed-fusion branch). Optional.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for merged inference checkpoint.",
    )
    args = parser.parse_args()

    mapping_state = torch.load(args.mapping_pt, map_location="cpu")
    embed_state = torch.load(args.mapping_embed_pt, map_location="cpu") if args.mapping_embed_pt else {}
    os.makedirs(args.output_dir, exist_ok=True)

    # Copy config + tokenizer sidecars; copy all model shards first.
    for name in os.listdir(args.base_checkpoint):
        src = os.path.join(args.base_checkpoint, name)
        dst = os.path.join(args.output_dir, name)
        if os.path.isdir(src):
            if not os.path.exists(dst):
                shutil.copytree(src, dst)
        elif name.endswith((".json", ".md", ".txt", ".model", ".safetensors")):
            shutil.copy2(src, dst)

    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config["llm_only"] = True
    config["freeze_mapping_enc2llm"] = False
    config["use_embed_fusion"] = bool(embed_state)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    index_path = os.path.join(args.base_checkpoint, "model.safetensors.index.json")
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    weight_map = dict(index["weight_map"])
    # Collect all trainable mappings to merge: {prefix: state_dict}
    mappings = {"mapping_enc2llm.": mapping_state}
    if embed_state:
        mappings["mapping_embed."] = embed_state
    for prefix, state in mappings.items():
        for key in state.keys():
            full_key = prefix + key
            weight_map[full_key] = weight_map.get(full_key, "model-00008-of-00008.safetensors")

    shard_to_keys = {}
    for key, shard in weight_map.items():
        shard_to_keys.setdefault(shard, []).append(key)

    for shard, keys in shard_to_keys.items():
        src_shard = os.path.join(args.base_checkpoint, shard)
        dst_shard = os.path.join(args.output_dir, shard)
        if os.path.isfile(dst_shard):
            tensors = load_file(dst_shard)
        else:
            tensors = load_file(src_shard)
        for key in keys:
            for prefix, state in mappings.items():
                if key.startswith(prefix):
                    local_key = key[len(prefix):]
                    if local_key in state:
                        tensors[key] = state[local_key]
                    break
        save_file(tensors, dst_shard, metadata={"format": "pt"})

    with open(os.path.join(args.output_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": index.get("metadata", {}), "weight_map": weight_map}, f, indent=2)

    print(f"Merged checkpoint written to {args.output_dir}")
    print("Use this path as --base_model in inference_xbridge_stage1.py / stage2_and_3.py")


if __name__ == "__main__":
    main()
