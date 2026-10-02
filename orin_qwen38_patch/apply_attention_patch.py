# SPDX-License-Identifier: Apache-2.0
"""Apply the small vLLM 0.20 source patch required for Q8 MTP attention."""

import subprocess
from pathlib import Path

import vllm

if vllm.__version__.split("+")[0] != "0.20.0":
    raise SystemExit(f"Expected vLLM 0.20.0, found {vllm.__version__}")
root = Path(vllm.__file__).resolve().parent.parent
patch = Path(__file__).resolve().parent / "patches/vllm020-int8-attention.patch"
check = subprocess.run(
    ["git", "apply", "--check", str(patch)], cwd=root, capture_output=True, text=True
)
if check.returncode:
    reverse = subprocess.run(
        ["git", "apply", "--reverse", "--check", str(patch)],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if reverse.returncode == 0:
        print("Q8 MTP attention patch already applied.")
    else:
        raise SystemExit(
            "vLLM source does not match the supported patch:\n" + check.stderr
        )
else:
    subprocess.run(["git", "apply", str(patch)], cwd=root, check=True)
    print("Applied Q8 MTP attention patch to", root)
