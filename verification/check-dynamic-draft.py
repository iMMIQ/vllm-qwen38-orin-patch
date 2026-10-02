# SPDX-License-Identifier: Apache-2.0
"""Check runtime rows, integer GEMM, CUDA Graph replay and draft head edge cases.

Run in the validated Torch/Triton CUDA environment; no model weights needed.
"""

import json
import sys
from pathlib import Path
import torch
import triton

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "orin_qwen38_patch/runtime")
)
from qwen38_orin.draft_kernels import DraftLinear, _project, _quantize_rows
from qwen38_orin.config import decode_capture_sizes

torch.set_num_threads(1)
torch.manual_seed(20261002)
weight = torch.randn(137, 5120, device="cuda", dtype=torch.float16)
linear = DraftLinear(weight)


def count():
    return len(_project.device_caches[0][0])


rows = []
for m in (1, 17, 33, 65, 2, 3, 7, 8, 15, 16, 18, 31, 32, 47, 64, 127, 128, 129, 257):
    x = torch.randn(m, 5120, device="cuda", dtype=torch.float16)
    out = linear.scores(x)
    a = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty(m, device="cuda", dtype=torch.float32)
    _quantize_rows[(m,)](x, a, scales, 5120, 8192, num_warps=8)
    ref = (
        (a.cpu().long() @ linear.quantized.cpu().long().t()).float()
        * scales.cpu()[:, None]
        * linear.scales.cpu()[None, :]
    )
    actual = out.cpu()
    torch.testing.assert_close(actual, ref, atol=2e-4, rtol=2e-6)
    rows.append(
        dict(
            m=m, max_abs_error=float((actual - ref).abs().max()), gemm_variants=count()
        )
    )
    print(rows[-1], flush=True)
assert count() == 4, "M created extra kernel variants"
for m in (3, 17, 33, 65):
    x = torch.randn(m, 5120, device="cuda", dtype=torch.float16)
    ref = linear.scores(x)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = linear.scores(x)
    graph.replay()
    torch.testing.assert_close(out, ref, atol=0, rtol=0)
    del graph, out
for m in (1, 3, 17, 33):
    x = torch.randn(m, 5120, device="cuda", dtype=torch.float16)
    exact = torch.nn.functional.linear(x, weight).argmax(-1)
    top = linear.top_tokens(x)
    torch.testing.assert_close(top, exact, atol=0, rtol=0)
    zero = linear.top_tokens(torch.zeros_like(x))
    assert not zero.any().item()
    linear.top_tokens(x)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = linear.top_tokens(x)
    graph.replay()
    torch.testing.assert_close(captured, top, atol=0, rtol=0)
    del graph, captured
assert linear.scores(
    torch.empty(0, 5120, device="cuda", dtype=torch.float16)
).shape == (0, 137)
assert linear.top_tokens(
    torch.empty(0, 5120, device="cuda", dtype=torch.float16)
).shape == (0,)
assert decode_capture_sizes(13, 3, 8192) == [4, 8, 16, 32, 52]
assert decode_capture_sizes(32, 3, 8192) == [4, 8, 16, 32, 64, 128]
assert decode_capture_sizes(5, 0, 8192) == [1, 2, 4, 5]
result = dict(
    integer_reference=rows,
    fixed_row_tiles=[16, 32, 64, 128],
    final_gemm_variants=count(),
    graph_and_head_checks=True,
    odd_capacity_graph_buckets=True,
)
print("VERIFIED", json.dumps(result), flush=True)
