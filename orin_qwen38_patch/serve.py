# SPDX-License-Identifier: Apache-2.0
"""Run the validated vLLM configuration with an optional native baseline."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "runtime"))
from qwen38_orin.config import decode_capture_sizes

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", required=True, help="Local compressed-tensors AWQ model directory"
)
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--served-model-name", default="qwen38-orin")
parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
parser.add_argument("--max-model-len", type=int, default=262144)
parser.add_argument("--max-num-seqs", type=int, default=8)
parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
parser.add_argument("--kv-cache-memory-gib", type=float, default=12)
parser.add_argument("--mtp-tokens", type=int, default=0)
parser.add_argument(
    "--no-fast-draft",
    action="store_true",
    help="Use original FP16 MTP projections and output head",
)
parser.add_argument("--block-size", type=int, default=2048)
parser.add_argument("--no-prefix-cache", action="store_true")
parser.add_argument(
    "--kv-cache-dtype",
    choices=["auto", "int8_per_token_head"],
    default="int8_per_token_head",
)
parser.add_argument(
    "--attention-backend", choices=["TRITON_ATTN", "FLASH_ATTN"], default="TRITON_ATTN"
)
parser.add_argument("--baseline", action="store_true", help="Disable all patch hooks")
args = parser.parse_args()
if not 0 <= args.mtp_tokens <= 8:
    parser.error("--mtp-tokens must be between 0 and 8")
if args.max_num_batched_tokens > 8192:
    parser.error("The prefill workspace supports at most 8192 tokens per batch")
try:
    capture_sizes = decode_capture_sizes(
        args.max_num_seqs, args.mtp_tokens, args.max_num_batched_tokens
    )
except ValueError as error:
    parser.error(str(error))
env = os.environ.copy()
# Run preflight with hooks off; activation takes place before the server import.
env["QWEN38_ORIN_PREFILL"] = "0"
subprocess.run(
    [sys.executable, str(ROOT / "check.py"), "--model", args.model]
    + (["--require-mtp"] if args.mtp_tokens else []),
    env=env,
    check=True,
)
env["QWEN38_ORIN_PREFILL"] = "0" if args.baseline else "1"
fast_draft = bool(args.mtp_tokens and not args.baseline and not args.no_fast_draft)
env["QWEN38_ORIN_DRAFT"] = "1" if fast_draft else "0"
env["PYTHONPATH"] = str(ROOT / "runtime") + (
    os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
)
env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
command = [
    sys.executable,
    "-m",
    "vllm.entrypoints.cli.main",
    "serve",
    args.model,
    "--served-model-name",
    args.served_model_name,
    "--host",
    args.host,
    "--port",
    str(args.port),
    "--dtype",
    "float16",
    "--language-model-only",
    "--enable-prompt-tokens-details",
    "--max-model-len",
    str(args.max_model_len),
    "--max-num-seqs",
    str(args.max_num_seqs),
    "--max-num-batched-tokens",
    str(args.max_num_batched_tokens),
    "--gpu-memory-utilization",
    str(args.gpu_memory_utilization),
    "--mamba-ssm-cache-dtype",
    "float32",
    "--kv-cache-dtype",
    args.kv_cache_dtype,
    "--attention-backend",
    args.attention_backend,
    "--kv-cache-memory-bytes",
    str(int(args.kv_cache_memory_gib * 2**30)),
    "--block-size",
    str(args.block_size),
    "--no-enable-prefix-caching" if args.no_prefix_cache else "--enable-prefix-caching",
    "--mamba-cache-mode",
    "none" if args.no_prefix_cache else "align",
    "--compilation-config",
    json.dumps(
        {
            "mode": 0,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": capture_sizes,
        }
    ),
]
if args.mtp_tokens:
    command.extend(
        [
            "--speculative-config",
            json.dumps(
                {
                    "method": "mtp",
                    "num_speculative_tokens": args.mtp_tokens,
                    "use_local_argmax_reduction": fast_draft,
                }
            ),
        ]
    )
print("Orin patch:", "baseline" if args.baseline else "W8A8 prefill", flush=True)
if args.mtp_tokens:
    print("MTP draft:", "dynamic W8A8" if fast_draft else "original FP16", flush=True)
os.execve(sys.executable, command, env)
