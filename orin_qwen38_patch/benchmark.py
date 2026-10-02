# SPDX-License-Identifier: Apache-2.0
"""Measure warmed client TTFT/decode with exact token IDs and no prefix hits."""

import argparse
import json
import statistics
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8000")
parser.add_argument("--model", default="qwen38-orin")
parser.add_argument("--lengths", default="512,2048,8192")
parser.add_argument("--runs", type=int, default=3)
parser.add_argument("--outputs", type=int, default=32)
parser.add_argument("--out", default="benchmark-results.json")
args = parser.parse_args()


def post(path, payload):
    return urllib.request.urlopen(
        urllib.request.Request(
            args.url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        ),
        timeout=1200,
    )


with post(
    "/tokenize",
    {
        "model": args.model,
        "prompt": "系统按任务编号保存结果，失效后重新计算。并发请求通过互斥锁协调。\n",
    },
) as response:
    seed = json.load(response)["tokens"]
assert seed


def request(length, outputs):
    # Distinct first blocks keep this cold-prefill benchmark valid with prefix
    # caching enabled. Tokenization takes place before the request timer.
    with post(
        "/tokenize",
        {"model": args.model, "prompt": f"Request {time.time_ns()}.\n"},
    ) as response:
        prefix = json.load(response)["tokens"]
    if len(prefix) >= length:
        raise ValueError("Input length must exceed the unique request prefix")
    remaining = length - len(prefix)
    ids = prefix + (seed * ((remaining + len(seed) - 1) // len(seed)))[:remaining]
    payload = {
        "model": args.model,
        "prompt": ids,
        "max_tokens": outputs,
        "temperature": 0,
        "seed": 20261001,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
    }
    started = time.monotonic()
    first = last = None
    first_count = 0
    tokens = []
    usage = None
    finished = False
    with post("/v1/completions", payload) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            out = json.loads(data)
            if out.get("error"):
                raise RuntimeError(out["error"])
            usage = out.get("usage") or usage
            for choice in out.get("choices", []):
                new = choice.get("token_ids") or []
                if new:
                    last = time.monotonic()
                    if first is None:
                        first = last
                        first_count = len(new)
                    tokens.extend(new)
                finished |= bool(choice.get("finish_reason"))
    assert finished and usage and first and last
    assert usage["prompt_tokens"] == length and usage["completion_tokens"] == outputs
    assert len(tokens) == outputs
    assert not (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    ttft = first - started
    return {
        "input_tokens": length,
        "output_tokens": outputs,
        "ttft_s": ttft,
        "prefill_tps_client": length / ttft,
        "first_chunk_tokens": first_count,
        "decode_tps_client": (
            (outputs - first_count) / (last - first) if last > first else None
        ),
        "end_to_end_output_tps_client": outputs / (last - started),
    }


results = {
    "arguments": vars(args),
    "completed": [],
    "note": "This benchmark uses its own prompt; performance may differ from historical evidence.",
}
for length in map(int, args.lengths.split(",")):
    request(length, args.outputs)
    runs = [request(length, args.outputs) for _ in range(args.runs)]
    row = {
        "input_tokens": length,
        "runs": runs,
        "median_prefill_tps_client": statistics.median(
            r["prefill_tps_client"] for r in runs
        ),
    }
    if all(r["decode_tps_client"] is not None for r in runs):
        row["median_decode_tps_client"] = statistics.median(
            r["decode_tps_client"] for r in runs
        )
    row["median_end_to_end_output_tps_client"] = statistics.median(
        r["end_to_end_output_tps_client"] for r in runs
    )
    results["completed"].append(row)
    with open(args.out, "w") as out:
        json.dump(results, out, ensure_ascii=False, indent=2)
    print(json.dumps(row), flush=True)
