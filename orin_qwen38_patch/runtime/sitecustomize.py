# SPDX-License-Identifier: Apache-2.0
"""Activate the patch in this process and vLLM's spawned workers."""

import os

if os.environ.get("QWEN38_ORIN_PREFILL") == "1":
    try:
        from qwen38_orin.bootstrap import install

        install()
    except Exception:
        import traceback

        traceback.print_exc()
        os._exit(75)
