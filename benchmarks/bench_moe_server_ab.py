"""Run the same local checkpoint and request sequence on original and current sources."""

from __future__ import annotations

import argparse
import concurrent.futures
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request


def get_json(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.load(response)


def request(origin, prompt, tokens, barrier):
    body = {"model": "gpt-oss-120b", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": tokens, "ignore_eos": True, "temperature": 0.0,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": True}}
    req = urllib.request.Request(origin + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    barrier.wait(timeout=30)
    start = time.perf_counter()
    stamps, reasoning, content, usage = [], [], [], None
    with urllib.request.urlopen(req, timeout=900) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            raw = line[5:].strip()
            if raw == b"[DONE]":
                break
            item = json.loads(raw)
            if item.get("usage"):
                usage = item["usage"]
            for choice in item.get("choices", []):
                delta = choice.get("delta") or {}
                r, c = delta.get("reasoning_content") or "", delta.get("content") or ""
                if r or c:
                    stamps.append(time.perf_counter())
                    reasoning.append(r)
                    content.append(c)
    done = time.perf_counter()
    if usage is None or len(stamps) < 2:
        raise RuntimeError("request did not return enough token events")
    text = json.dumps({"reasoning": "".join(reasoning), "content": "".join(content)}, ensure_ascii=False, sort_keys=True)
    return {"start": start, "first": stamps[0], "last": stamps[-1], "done": done,
            "ttft_s": stamps[0] - start, "usage": usage, "events": len(stamps),
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "text": json.loads(text)}


def group(origin, prompts, count, tokens):
    barrier = threading.Barrier(count)
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
        rows = list(pool.map(lambda prompt: request(origin, prompt, tokens, barrier), prompts[:count]))
    span = max(r["last"] for r in rows) - min(r["first"] for r in rows)
    return {"batch": count, "requests": rows,
            "decode_tok_s": sum(r["usage"]["completion_tokens"] - 1 for r in rows) / span,
            "ttft_s": statistics.mean(r["ttft_s"] for r in rows),
            "vram_bytes": get_json(origin + "/v1/stats").get("vram_bytes")}


def variant(label, source, args, output):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    origin = f"http://127.0.0.1:{port}"
    command = [sys.executable, "-m", "freetoken.cli", "serve", "--model", str(args.model),
               "--host", "127.0.0.1", "--port", str(port), "--gpu", "0",
               "--attention-backend", "triton", "--moe-strategy", "offload",
               "--moe-cache-size", "1241", "--memory-ratio", "0.85",
               "--max-running-requests", "4", "--cuda-graph-max-bs", "4",
               "--max-seq-len-override", "8448", "--kv-reserve-tokens", "8192"]
    env = dict(os.environ, PYTHONPATH=str(source), PYTHONUNBUFFERED="1")
    logfile = args.output.parent / (args.output.stem + "_" + label + "_server.log")
    prompt = json.loads(args.prompts.read_text().splitlines()[0])["problem"]
    prompt += "\nPlease reason step by step, and put your final answer within \\boxed{}."
    prompts = [prompt if args.same_prompt else prompt + f"\nRequest identifier: {i}." for i in range(4)]
    print(f"Starting {label}: {logfile}", flush=True)
    result = {"command": command, "source": str(source), "log": str(logfile), "rows": [],
              "prompts": prompts, "sampling": {"temperature": 0.0, "ignore_eos": True, "max_tokens": args.tokens}}
    output[label] = result
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    with logfile.open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 1200
            ready = False
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"{label} exited during startup; see {logfile}")
                try:
                    health = get_json(origin + "/health")
                    ready = health.get("maintenance") == "serving" or health.get("status") == "ok"
                except (OSError, ValueError):
                    pass
                if ready:
                    break
                time.sleep(1)
            if not ready:
                raise TimeoutError(f"{label} did not become ready")
            print(f"{label} ready; warming the shared prefix at batch 1", flush=True)
            result["warmup_single"] = group(origin, prompts, 1, args.tokens)
            print(f"{label} warming batch 4", flush=True)
            result["warmup"] = group(origin, prompts, 4, args.tokens)
            for batch in (1, 4):
                for repeat in range(args.repeats):
                    row = group(origin, prompts, batch, args.tokens)
                    row["repeat"] = repeat
                    result["rows"].append(row)
                    args.output.write_text(json.dumps(output, indent=2) + "\n")
                    print(f"{label} batch={batch} repeat={repeat}: {row['decode_tok_s']:.3f} tok/s, TTFT={row['ttft_s']:.3f}s", flush=True)
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-python", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--same-prompt", action="store_true", help="identical prompt rows eliminate arrival-order effects on prefix prefill")
    parser.add_argument("--validate-existing", action="store_true", help="validate recorded response bytes without starting servers")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"model": str(args.model), "tokens": args.tokens,
              "same_prompt": args.same_prompt,
              "prompt_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
              "precision_note": "server check compares UTF-8 response bytes; kernel benchmark separately compares raw tensor bytes"}
    if args.validate_existing:
        result = json.loads(args.output.read_text())
        if result.get("same_prompt", False) != args.same_prompt:
            raise ValueError("comparison mode does not match the recorded prompts")
    else:
        variant("before", args.baseline_python.resolve(), args, result)
        variant("after", Path(__file__).resolve().parents[1] / "python", args, result)
    proof = []
    group_proof = []
    for old, new in zip(result["before"]["rows"], result["after"]["rows"]):
        if old["batch"] != new["batch"] or old["repeat"] != new["repeat"]:
            raise AssertionError("request sequence changed")
        for i, (a, b) in enumerate(zip(old["requests"], new["requests"])):
            equal = a["text"] == b["text"] and a["usage"] == b["usage"]
            proof.append({"batch": old["batch"], "repeat": old["repeat"], "request": i, "text_and_usage_equal": equal,
                          "before_sha256": a["text_sha256"], "after_sha256": b["text_sha256"]})
        signature = lambda row: Counter((json.dumps(q["text"], ensure_ascii=False, sort_keys=True).encode(),
                                         json.dumps(q["usage"], sort_keys=True)) for q in row["requests"])
        group_proof.append({"batch": old["batch"], "repeat": old["repeat"],
                            "equal": signature(old) == signature(new) if args.same_prompt else all(
                                a["text"] == b["text"] and a["usage"] == b["usage"]
                                for a, b in zip(old["requests"], new["requests"])),
                            "comparison": "response multiset for identical prompts" if args.same_prompt else "per-request bytes"})
    result["response_proof"] = proof
    result["group_response_proof"] = group_proof
    result["summary"] = [{"batch": batch,
                          **{label + "_decode_median_tok_s": statistics.median(r["decode_tok_s"] for r in result[label]["rows"] if r["batch"] == batch)
                             for label in ("before", "after")},
                          **{label + "_ttft_median_s": statistics.median(r["ttft_s"] for r in result[label]["rows"] if r["batch"] == batch)
                             for label in ("before", "after")}}
                         for batch in (1, 4)]
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2), flush=True)
    if not all(r["equal"] for r in group_proof):
        raise AssertionError("server response bytes or token counts differ")
    print(f"All {len(group_proof)} response groups match; saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
