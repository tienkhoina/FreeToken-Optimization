"""Real checkpoint policy A/B from empty cache with same sequential requests and VRAM slots."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

from bench_moe_server_ab import get_json, request

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent

PROMPTS = [
    ("math", "Find the sum of all integer bases b>9 for which 17_b divides 97_b. Explain the divisibility argument."),
    ("math", "For integer b>9, determine when b+7 divides 9b+7. Derive all possibilities carefully."),
    ("math", "Find all positive integer n such that n+5 divides 7n+5. Explain using remainders."),
    ("math", "Find all integer bases b>8 such that 15_b is a divisor of 85_b. Show the number theory steps."),
    ("coding", "Write a Python function that merges overlapping intervals and explain its time complexity."),
    ("coding", "Explain how to implement an LRU cache in Python with a hash map and doubly linked list."),
    ("coding", "Write a Python function for binary search and explain its loop invariant."),
    ("coding", "Explain how to detect a cycle in a directed graph using depth first search in Python."),
    ("math_return", "Find all positive integers n for which n+3 divides 5n+3. Use a modular arithmetic argument."),
    ("math_return", "Determine the integer bases b>9 for which 17_b divides 97_b. Give a rigorous derivation."),
]


def run(policy, args, result):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        raw = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
        if int(raw.splitlines()[0]) < 64:
            break
        time.sleep(1)
    else:
        raise RuntimeError("GPU memory from another process has not drained; refusing unequal startup budgets")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    origin = f"http://127.0.0.1:{port}"
    logpath = args.output.parent / f"{policy}_server.log"
    audit = args.output.parent / (policy + "_audit")
    audit.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", FREETOKEN_HOT_AUDIT_DIR=str(audit),
               FREETOKEN_HOT_AUDIT_TOKENS=str(args.tokens),
               PYTHONPATH=str(REPO / "benchmarks/adaptive_hot_instrument") + os.pathsep + str(REPO / "python"))
    command = [sys.executable, "-m", "freetoken.cli", "serve", "--model", str(args.model),
               "--host", "127.0.0.1", "--port", str(port), "--gpu", "0", "--attention-backend", "triton",
               "--moe-strategy", "offload", "--moe-cache-size", str(args.slots), "--moe-cache-policy", policy,
               "--memory-ratio", "0.85", "--max-running-requests", "1", "--cuda-graph-max-bs", "1",
               "--max-seq-len-override", "8448", "--kv-reserve-tokens", "8192", "--num-pages", "8192"]
    row = {"command": command, "log": str(logpath), "audit_dir": str(audit), "requests": []}
    result[policy] = row
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Starting {policy}: {logpath}", flush=True)
    with logpath.open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 1200
            ready = False
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"{policy} exited before ready; see {logpath}")
                try:
                    health = get_json(origin + "/health")
                    ready = health.get("maintenance") == "serving" or health.get("status") == "ok"
                except (OSError, ValueError):
                    pass
                if ready:
                    break
                time.sleep(1)
            if not ready:
                raise TimeoutError("server did not start")
            for index, (topic, prompt) in enumerate(PROMPTS):
                answer = request(origin, prompt, args.tokens, threading.Barrier(1))
                answer.update(index=index, topic=topic, prompt=prompt,
                              decode_tok_s=(answer["usage"]["completion_tokens"] - 1) / (answer["last"] - answer["first"]))
                row["requests"].append(answer)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                print(f"{policy} {index} {topic}: {answer['decode_tok_s']:.3f} tok/s TTFT={answer['ttft_s']:.3f}s", flush=True)
            # Finish request-boundary audit without adding another generation workload.
            time.sleep(1)
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
    row["audit"] = [json.loads(line) for path in sorted(audit.glob("*.jsonl")) for line in path.read_text().splitlines()]
    args.output.write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/gpt-oss-120b")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slots", type=int, default=1241)
    parser.add_argument("--tokens", type=int, default=128)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"model": str(args.model), "tokens": args.tokens, "slots": args.slots,
              "method": "same cold cache start, ordered topic shifts, GPU graph BS1, greedy; request-boundary counters outside decode timing",
              "prompts_sha256": hashlib.sha256(json.dumps(PROMPTS).encode()).hexdigest(), "prompts": PROMPTS}
    for policy in ("lru", "adaptive_hot"):
        run(policy, args, result)
    result["response_proof"] = [{"index": a["index"], "equal_text_and_usage": a["text"] == b["text"] and a["usage"] == b["usage"]}
                                for a, b in zip(result["lru"]["requests"], result["adaptive_hot"]["requests"])]
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["response_proof"]), flush=True)
    if not all(p["equal_text_and_usage"] for p in result["response_proof"]):
        raise AssertionError("full-model response bytes differ")


if __name__ == "__main__":
    main()
