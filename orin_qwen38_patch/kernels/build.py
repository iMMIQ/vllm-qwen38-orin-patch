# SPDX-License-Identifier: Apache-2.0
"""Regenerate the 24 selected SM87 kernels with TileLang 0.1.13."""

import ast
import hashlib
import json
import shutil
from pathlib import Path

from activation import activation_q8
from weights import weight_stats, lut_expand
from gemm import tiled
import fusions
from wy import wy

OUT = Path(__file__).resolve().parents[1] / "runtime/qwen38_orin/binaries"
OUT.mkdir(parents=True, exist_ok=True)
metadata = {}


def save(name, kernel):
    adapter = kernel.adapter
    host = Path(adapter.lib_generator.pypath).read_text()
    shutil.copyfile(adapter.lib_generator.libpath, OUT / (name + ".cubin"))
    tree = ast.parse(host)
    constants = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "config"
                and target.attr != "hStream"
            ):
                constants[target.attr] = ast.literal_eval(node.value)
    call = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "call"
    )
    signature = [a.arg for a in call.args.args if a.arg not in ("kernels", "stream")]
    values = next(
        n.value
        for n in ast.walk(call)
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "arg_values"
    )
    order = [n.func.value.id for n in values.elts]
    assert len(adapter.function_names) == 1
    metadata[name] = {
        "symbol": adapter.function_names[0],
        "grid": [constants["gridDim" + a] for a in "XYZ"],
        "block": [constants["blockDim" + a] for a in "XYZ"],
        "shared": constants["sharedMemBytes"],
        "signature": signature,
        "argument_order": [signature.index(a) for a in order],
        "sha256": hashlib.sha256((OUT / (name + ".cubin")).read_bytes()).hexdigest(),
    }
    print(name, flush=True)


for n, k in [(34816, 5120), (5120, 17408), (16384, 5120), (14336, 5120), (5120, 6144)]:
    bn = 64 if (n, k) in {(5120, 17408), (16384, 5120)} else 128
    save(f"stats-{n}-{k}", weight_stats(n, k))
    save(f"expandlut-{bn}-256-{n}-{k}", lut_expand(n, k))
    save(f"gemm2-{n}-{k}", tiled(n, k))
for k in [5120, 6144, 17408]:
    save(f"activation-{k}", activation_q8(256, k))
save("norm-none", fusions.gemma_norm_no_residual())
for dtype in ["float16", "float32"]:
    save("norm-" + dtype, fusions.gemma_norm(residual_dtype=dtype))
save("silu-q8", fusions.silu_mul_q8())
save("gdn-norm-q8", fusions.gdn_norm_q8())
save("wy-256-head-exp2", wy())
(OUT / "manifest.json").write_text(json.dumps(metadata, indent=2))
root = OUT.parents[2]
checksums = []
for path in sorted(root.rglob("*")):
    if (
        path.is_file()
        and path.name != "SHA256SUMS"
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    ):
        checksums.append(
            hashlib.sha256(path.read_bytes()).hexdigest()
            + "  "
            + str(path.relative_to(root))
        )
(root / "SHA256SUMS").write_text("\n".join(checksums) + "\n")
