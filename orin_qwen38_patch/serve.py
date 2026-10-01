# SPDX-License-Identifier: Apache-2.0
"""Run the validated vLLM configuration with an optional native baseline."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", required=True, help="Local compressed-tensors AWQ model directory"
)
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--served-model-name", default="qwen38-orin")
parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
parser.add_argument("--baseline", action="store_true", help="Disable all patch hooks")
args = parser.parse_args()
env = os.environ.copy()
# Run preflight with hooks off; activation takes place before the server import.
env["QWEN38_ORIN_PREFILL"] = "0"
subprocess.run(
    [sys.executable, str(ROOT / "check.py"), "--model", args.model], env=env, check=True
)
env["QWEN38_ORIN_PREFILL"] = "0" if args.baseline else "1"
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
    "--max-model-len",
    "12288",
    "--max-num-seqs",
    "1",
    "--max-num-batched-tokens",
    "8192",
    "--gpu-memory-utilization",
    str(args.gpu_memory_utilization),
    "--mamba-ssm-cache-dtype",
    "float32",
    "--no-enable-prefix-caching",
    "--compilation-config",
    json.dumps(
        {
            "mode": 0,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1],
        }
    ),
]
print("Orin patch:", "baseline" if args.baseline else "W8A8 prefill", flush=True)
os.execve(sys.executable, command, env)
