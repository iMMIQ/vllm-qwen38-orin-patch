# SPDX-License-Identifier: Apache-2.0
"""Compile SM87 cubins; the deployed runtime launches them via the CUDA driver."""

import tilelang

orin_jit = tilelang.jit(
    out_idx=[], execution_backend="nvrtc", target={"kind": "cuda", "arch": "sm_87"}
)
