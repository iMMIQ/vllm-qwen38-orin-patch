# SPDX-License-Identifier: Apache-2.0
"""Dynamic-row draft projections and batched candidate selection for Qwen MTP."""

import torch

from .config import request_buckets
from .draft_kernels import DraftLinear

PROJECTIONS = {
    (5120, 10240),  # MTP embedding/hidden-state projection
    (14336, 5120),  # QKV and query gate
    (5120, 6144),  # Attention output
    (34816, 5120),  # FFN gate/up
    (5120, 17408),  # FFN down
}


def install():
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

    if getattr(Qwen3_5MTP, "_orin_dynamic_draft", False):
        return
    original_load = UnquantizedLinearMethod.process_weights_after_loading
    original_apply = UnquantizedLinearMethod.apply
    original_proposer_load = SpecDecodeBaseProposer.load_model
    original_logits = Qwen3_5MTP.compute_logits

    def load(self, layer):
        original_load(self, layer)
        weight = getattr(layer, "weight", None)
        if (
            getattr(layer, "prefix", "").startswith("mtp.")
            and weight is not None
            and tuple(weight.shape) in PROJECTIONS
            and weight.dtype == torch.float16
            and weight.is_contiguous()
        ):
            layer._orin_draft_linear = DraftLinear(weight)
            layer._orin_draft_linear.warmup()

    def apply(self, layer, x, bias=None):
        fast = getattr(layer, "_orin_draft_linear", None)
        if fast is not None and x.dtype == torch.float16 and bias is None:
            return fast.scores(x, dtype=x.dtype)
        return original_apply(self, layer, x, bias)

    def proposer_load(self, target_model):
        result = original_proposer_load(self, target_model)
        model = self.model
        if isinstance(model, Qwen3_5MTP):
            weight = model.lm_head.weight
            if (
                tuple(weight.shape) == (248320, 5120)
                and weight.dtype == torch.float16
                and weight.is_contiguous()
            ):
                model._orin_draft_head = DraftLinear(weight)
                model._orin_draft_head.warmup(
                    head=True,
                    row_buckets=request_buckets(
                        self.vllm_config.scheduler_config.max_num_seqs
                    ),
                )
                print(
                    "ORIN_DRAFT rows=dynamic tiles=16,32,64,128 head=batched-fp16-rerank",
                    flush=True,
                )
        return result

    def get_top_tokens(self, hidden_states):
        fast = getattr(self, "_orin_draft_head", None)
        if (
            fast is not None
            and hidden_states.dtype == torch.float16
            and hidden_states.shape[-1] == fast.k
        ):
            return fast.top_tokens(hidden_states)
        return original_logits(self, hidden_states).argmax(dim=-1)

    UnquantizedLinearMethod.process_weights_after_loading = load
    UnquantizedLinearMethod.apply = apply
    SpecDecodeBaseProposer.load_model = proposer_load
    Qwen3_5MTP.get_top_tokens = get_top_tokens
    Qwen3_5MTP._orin_dynamic_draft = True
