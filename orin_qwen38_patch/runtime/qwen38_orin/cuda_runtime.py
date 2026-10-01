# SPDX-License-Identifier: Apache-2.0
"""Launch precompiled TileLang cubins on the caller's Torch CUDA stream."""

import ctypes as C
import json
from pathlib import Path
import torch


class Runtime:
    def __init__(self):
        torch.cuda.init()
        self.context_anchor = torch.empty(1, device="cuda")
        self.driver = C.CDLL("libcuda.so.1")
        self.driver.cuModuleLoad.argtypes = [C.POINTER(C.c_void_p), C.c_char_p]
        self.driver.cuModuleGetFunction.argtypes = [
            C.POINTER(C.c_void_p),
            C.c_void_p,
            C.c_char_p,
        ]
        self.driver.cuFuncSetAttribute.argtypes = [C.c_void_p, C.c_int, C.c_int]
        self.driver.cuLaunchKernel.argtypes = [
            C.c_void_p,
            *([C.c_uint] * 7),
            C.c_void_p,
            C.POINTER(C.c_void_p),
            C.c_void_p,
        ]
        root = Path(__file__).resolve().parent / "binaries"
        self.meta = json.loads((root / "manifest.json").read_text())
        self.modules = []
        self.functions = {}
        for name, meta in self.meta.items():
            module = C.c_void_p()
            function = C.c_void_p()
            self.check(
                self.driver.cuModuleLoad(
                    C.byref(module), str(root / (name + ".cubin")).encode()
                )
            )
            self.check(
                self.driver.cuModuleGetFunction(
                    C.byref(function), module, meta["symbol"].encode()
                )
            )
            if meta["shared"] > 49152:
                self.check(self.driver.cuFuncSetAttribute(function, 8, meta["shared"]))
            self.modules.append(module)
            self.functions[name] = function

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f"CUDA driver operation failed: {code}")

    def launch(self, name, *tensors, grid=None):
        meta = self.meta[name]
        assert len(tensors) == len(meta["signature"])
        values = [C.c_void_p(tensors[i].data_ptr()) for i in meta["argument_order"]]
        args = (C.c_void_p * len(values))(
            *[C.cast(C.byref(v), C.c_void_p) for v in values]
        )
        stream = torch.cuda.current_stream(tensors[0].device).cuda_stream
        self.check(
            self.driver.cuLaunchKernel(
                self.functions[name],
                *(grid or meta["grid"]),
                *meta["block"],
                meta["shared"],
                C.c_void_p(stream),
                args,
                None,
            )
        )
