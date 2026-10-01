# SPDX-License-Identifier: Apache-2.0
"""Check the runtime and checkpoint without loading the model's weights."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def check(model=None):
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
    check(parser.parse_args().model)
