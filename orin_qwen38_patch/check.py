# SPDX-License-Identifier: Apache-2.0
"""Check the runtime and checkpoint without loading the model's weights."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def check(model=None, require_mtp=False):
    import torch
    import vllm

    if vllm.__version__.split("+")[0] != "0.20.0":
        raise RuntimeError(f"Expected vLLM 0.20.0, got {vllm.__version__}")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 7):
        raise RuntimeError(
            "The precompiled kernels require an available CUDA SM87 device"
        )
    bins = ROOT / "runtime/qwen38_orin/binaries"
    manifest = json.loads((bins / "manifest.json").read_text())
    for name, meta in manifest.items():
        digest = hashlib.sha256((bins / (name + ".cubin")).read_bytes()).hexdigest()
        if digest != meta["sha256"]:
            raise RuntimeError(f"Kernel checksum mismatch: {name}")
    if model:
        config = json.loads((Path(model) / "config.json").read_text())
        text = config.get("text_config", config)
        expected = {
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "num_hidden_layers": 64,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
        }
        for key, value in expected.items():
            if text.get(key) != value:
                raise RuntimeError(
                    f"Unsupported model: {key}={text.get(key)}; expected {value}"
                )
        quant = config.get("quantization_config") or {}
        groups = quant.get("config_groups") or {}
        if quant.get("quant_method") != "compressed-tensors" or not groups:
            raise RuntimeError(
                "Use the tested compressed-tensors AWQ checkpoint; see model-reference.json"
            )
        for group in groups.values():
            weights = group.get("weights") or {}
            if (
                weights.get("num_bits"),
                weights.get("group_size"),
                weights.get("symmetric"),
            ) != (4, 128, False):
                raise RuntimeError("Expected asymmetric W4 group-128 weights")
        if require_mtp:
            from safetensors import safe_open

            required = {
                "mtp." + name + ".weight"
                for name in (
                    "fc",
                    "pre_fc_norm_embedding",
                    "pre_fc_norm_hidden",
                    "norm",
                    "layers.0.input_layernorm",
                    "layers.0.post_attention_layernorm",
                    "layers.0.self_attn.q_proj",
                    "layers.0.self_attn.k_proj",
                    "layers.0.self_attn.v_proj",
                    "layers.0.self_attn.o_proj",
                    "layers.0.self_attn.q_norm",
                    "layers.0.self_attn.k_norm",
                    "layers.0.mlp.gate_proj",
                    "layers.0.mlp.up_proj",
                    "layers.0.mlp.down_proj",
                )
            }
            found = set()
            for shard in Path(model).glob("*.safetensors"):
                with safe_open(shard, framework="pt", device="cpu") as weights:
                    found.update(weights.keys())
            if missing := required - found:
                raise RuntimeError(
                    f"MTP tensors missing from checkpoint: {sorted(missing)}"
                )
    result = {
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "device_capability": [8, 7],
        "kernels_verified": len(manifest),
    }
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--require-mtp", action="store_true")
    args = parser.parse_args()
    check(args.model, args.require_mtp)
