# SPDX-License-Identifier: Apache-2.0
"""Check the vLLM interface version before installing runtime hooks."""


def install():
    import os
    import vllm

    if vllm.__version__.split("+")[0] != "0.20.0":
        raise RuntimeError(f"This patch targets vLLM 0.20.0; found {vllm.__version__}")
    from .backend import install as install_backend

    install_backend()
    from .mtp import install as install_mtp_guard

    install_mtp_guard()
    if os.environ.get("QWEN38_ORIN_DRAFT") == "1":
        from .draft import install as install_draft

        install_draft()
    from .kv8_attention import install as install_kv8_attention

    install_kv8_attention()
