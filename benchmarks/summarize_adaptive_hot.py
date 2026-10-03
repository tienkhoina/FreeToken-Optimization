"""Join recorded request counters with response timing, without executing GPU work."""

import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.input.read_text())
    result = {"source": str(args.input), "slots": source["slots"], "tokens": source["tokens"], "requests": [], "topics": []}
    recorded = {}
    for policy in ("lru", "adaptive_hot"):
        audit = source[policy].get("audit")
        if not audit:
            audit = [json.loads(line) for path in Path(source[policy]["audit_dir"]).glob("*.jsonl") for line in path.read_text().splitlines()]
        by_uid = {}
        for row in audit:
            if not row["uid"] or row["decode_forwards"] == 0:
                continue
            old, new = row["before"], row["after"]
            stats = [[b[i] - a[i] for i in range(3)] for a, b in zip(old["lru_stats"], new["lru_stats"])]
            active, missed, calls = [sum(r[i] for r in stats) for i in range(3)]
            by_uid[row["uid"][0]] = {"active_expert_queries": active, "missed_experts": missed, "layer_calls": calls,
                                     "miss_rate": missed / active if active else 0,
                                     "bytes_fetched": missed * new["expert_bytes"],
                                     "hot_count_before": old.get("protected_count", 0), "hot_count_after": new.get("protected_count", 0),
                                     "per_layer_stats": stats}
        recorded[policy] = by_uid
    for a, b in zip(source["lru"]["requests"], source["adaptive_hot"]["requests"]):
        index = a["index"]
        row = {"index": index, "topic": a["topic"], "response_byte_equal": a["text"] == b["text"] and a["usage"] == b["usage"]}
        for policy, request in (("lru", a), ("adaptive_hot", b)):
            row[policy] = {"decode_tok_s": request["decode_tok_s"], "ttft_s": request["ttft_s"],
                           "decode_window_s": request["last"] - request["first"],
                           **recorded[policy].get(index, {})}
        row["throughput_ratio"] = row["adaptive_hot"]["decode_tok_s"] / row["lru"]["decode_tok_s"]
        result["requests"].append(row)
    for topic in dict.fromkeys(r["topic"] for r in result["requests"]):
        rows = [r for r in result["requests"] if r["topic"] == topic]
        item = {"topic": topic, "requests": len(rows)}
        for policy in ("lru", "adaptive_hot"):
            active = sum(r[policy].get("active_expert_queries", 0) for r in rows)
            miss = sum(r[policy].get("missed_experts", 0) for r in rows)
            span = sum(r[policy]["decode_window_s"] for r in rows)
            tokens = sum(source["tokens"] - 1 for r in rows)
            item[policy] = {"aggregate_decode_tok_s": tokens / span,
                            "median_decode_tok_s": statistics.median(r[policy]["decode_tok_s"] for r in rows),
                            "miss_rate": miss / active if active else None,
                            "bytes_fetched": sum(r[policy].get("bytes_fetched", 0) for r in rows)}
        item["throughput_ratio"] = item["adaptive_hot"]["aggregate_decode_tok_s"] / item["lru"]["aggregate_decode_tok_s"]
        result["topics"].append(item)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["topics"], indent=2))


if __name__ == "__main__":
    main()
