"""Measured miss-count policies, shared by calibration and runtime profile loading."""

import math


GEOMETRY_KEYS = ("format", "hidden", "intermediate", "gpu", "cpu_workers", "cpu_isa",
                 "activation", "swiglu_alpha", "swiglu_limit", "top_k")


def recommend_fetch_counts(table, max_misses, cpu_percentile=90, minimum_gain=0.15):
    if not 0 <= minimum_gain < 1 or not 0 <= cpu_percentile <= 100:
        raise ValueError("Invalid split gain or CPU timing percentile")
    costs = {row["experts"]: row for row in table}
    if set(costs) != set(range(1, max_misses + 1)):
        raise ValueError("Calibration must cover every expert count from 1 to max_misses")
    cpu_cost = {0: 0.0}
    gpu_cost = {0: 0.0}
    for count, row in costs.items():
        samples = sorted(row["samples"]["cpu_graph_ms"])
        if not samples or any(not math.isfinite(value) or value < 0 for value in samples):
            raise ValueError("Invalid CPU branch timing samples")
        cpu_cost[count] = samples[(len(samples) - 1) * cpu_percentile // 100]
        copy_ms, gpu_ms = row["copy_ms"], row["gpu_ms"]
        if any(not math.isfinite(value) or value < 0 for value in (copy_ms, gpu_ms)):
            raise ValueError("Invalid copy/GPU timing")
        gpu_cost[count] = copy_ms + gpu_ms
    counts, estimates = [0], []
    for misses in range(1, max_misses + 1):
        choices = [{"cpu": cpu, "gpu": misses - cpu,
                    "estimated_ms": max(cpu_cost[cpu], gpu_cost[misses - cpu])}
                   for cpu in range(misses + 1)]
        best = min(choices, key=lambda choice: choice["estimated_ms"])
        if best["estimated_ms"] > choices[0]["estimated_ms"] * (1 - minimum_gain):
            best = choices[0]
        counts.append(best["gpu"])
        estimates.append({"misses": misses, "best": best, "choices": choices})
    return counts, estimates


def profile_rows(profile, experts):
    if profile.get("query_tokens", 1) != 1:
        raise ValueError("The main schedule profile must measure one-token decode")
    rows = [(1, profile["recommend_fetch_counts"])]
    for prefill in profile.get("prefill_profiles", []):
        if any(prefill.get(key) != profile.get(key) for key in GEOMETRY_KEYS):
            raise ValueError("Prefill profile geometry/hardware differs from decode calibration")
        rows.append((prefill["query_tokens"], prefill["recommend_fetch_counts"]))
    if [tokens for tokens, _ in rows] != sorted(set(tokens for tokens, _ in rows)):
        raise ValueError("Profile token buckets must be strictly increasing, starting at one")
    for tokens, counts in rows:
        if type(tokens) is not int or tokens < 1 or not 1 <= len(counts) <= experts + 1:
            raise ValueError("Invalid calibration token bucket or expert-count coverage")
        if any(type(value) is not int or value < 0 or value > misses for misses, value in enumerate(counts)):
            raise ValueError("Invalid calibrated GPU fetch counts")
    return rows


def merge_profiles(profiles, experts):
    ordered = sorted(profiles, key=lambda profile: profile.get("query_tokens", 1))
    if not ordered or ordered[0].get("query_tokens", 1) != 1:
        raise ValueError("Merging profiles requires one-token decode calibration")
    result = {**ordered[0], "prefill_profiles": ordered[1:]}
    profile_rows(result, experts)
    return result
