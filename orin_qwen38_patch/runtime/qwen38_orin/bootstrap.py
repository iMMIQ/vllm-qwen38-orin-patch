# SPDX-License-Identifier: Apache-2.0
"""Check the vLLM interface version before installing runtime hooks."""


def install():
    import vllm

    if vllm.__version__.split("+")[0] != "0.20.0":
        raise RuntimeError(f"This patch targets vLLM 0.20.0; found {vllm.__version__}")
    from .backend import install as install_backend

    install_backend()
