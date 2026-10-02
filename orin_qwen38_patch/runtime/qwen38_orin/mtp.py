# SPDX-License-Identifier: Apache-2.0
"""Reject incomplete MTP checkpoints instead of drafting with unloaded weights."""


def install():
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP

    if getattr(Qwen3_5MTP, "_orin_weight_guard", False):
        return
    original = Qwen3_5MTP.load_weights

    def load(self, weights):
        loaded = original(self, weights)
        required = {
            name
            for name, _ in self.named_parameters()
            if name.endswith(".weight")
            and name != "model.embed_tokens.weight"
            and name != "lm_head.weight"
        }
        missing = required - loaded
        if missing:
            raise RuntimeError(f"Incomplete Qwen MTP checkpoint: {sorted(missing)}")
        print(f"ORIN_MTP_WEIGHTS verified={len(required)}", flush=True)
        return loaded

    Qwen3_5MTP.load_weights = load
    Qwen3_5MTP._orin_weight_guard = True
