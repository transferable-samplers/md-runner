#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

from remd_progress import (
    REMD_ROOTS,
    TARGET_NS,
    find_remd_runs,
    replica_label,
    replica_sort_key,
    run_time_ns,
)

CACHE_PATH = Path(__file__).with_name("remd_rate_cache.json")

ETA_TARGETS_NS = (500, 1000)


def eta_days(time_ns: float, rate_ns_per_h: float | None, target_ns: float) -> float | None:
    if rate_ns_per_h is None or rate_ns_per_h <= 0:
        return None
    if time_ns >= target_ns:
        return 0.0
    return (target_ns - time_ns) / rate_ns_per_h / 24.0


def scan() -> tuple[float, dict[str, dict]]:
    now = time.time()
    runs: dict[str, dict] = {}
    for root in REMD_ROOTS:
        if not root.exists():
            continue
        tag = root.parents[1].name.replace("md-runner-remd-reference-", "")
        for run in find_remd_runs(root):
            runs[run.name] = {
                "seq": run.name.split("_")[0],
                "label": replica_label(run.name),
                "tag": tag,
                "time_ns": run_time_ns(run),
            }
    return now, runs


def load_cache() -> tuple[float | None, dict[str, dict]]:
    if not CACHE_PATH.exists():
        return None, {}
    data = json.loads(CACHE_PATH.read_text())
    return data["timestamp"], data["runs"]


def save_cache(ts: float, runs: dict[str, dict]) -> None:
    CACHE_PATH.write_text(json.dumps({"timestamp": ts, "runs": runs}, indent=2))


def main() -> None:
    prev_ts, prev_runs = load_cache()
    now_ts, now_runs = scan()

    # tag -> seq -> list of (label, time_ns, rate, name)
    by_tag: dict[str, dict[str, list[tuple[str, float, float | None, str]]]] = defaultdict(
        lambda: defaultdict(list),
    )
    for name, info in now_runs.items():
        prev = prev_runs.get(name)
        rate = None
        if prev is not None and prev_ts is not None:
            dt_h = (now_ts - prev_ts) / 3600.0
            d_ns = info["time_ns"] - prev["time_ns"]
            if dt_h > 0:
                rate = d_ns / dt_h
        by_tag[info["tag"]][info["seq"]].append(
            (info["label"], info["time_ns"], rate, name),
        )

    if prev_ts is None:
        print("# no prior cache — showing current ns only; rerun later for ns/hour")
        elapsed_str = "n/a"
    else:
        elapsed_h = (now_ts - prev_ts) / 3600.0
        elapsed_str = f"{elapsed_h:.2f} h"

    print(f"# elapsed since last scan: {elapsed_str}")

    for tag in [r.parents[1].name.replace("md-runner-remd-reference-", "") for r in REMD_ROOTS]:
        if tag not in by_tag:
            continue
        print(f"\n=== {tag} ===")
        for seq in sorted(by_tag[tag], key=lambda s: (len(s), s)):
            runs = sorted(by_tag[tag][seq], key=lambda r: replica_sort_key(r[0]))
            n_done = sum(1 for _, t, _, _ in runs if t >= TARGET_NS)
            print(f"\n{seq}  ({n_done}/{len(runs)} >= {TARGET_NS} ns)")
            for label, time_ns, rate, _ in runs:
                mark = "x" if time_ns >= TARGET_NS else " "
                rate_str = "    n/a" if rate is None else f"{rate:6.2f} ns/h"
                eta_parts = []
                for tgt in ETA_TARGETS_NS:
                    d = eta_days(time_ns, rate, tgt)
                    if d is None:
                        eta_parts.append(f"{tgt}ns: n/a")
                    else:
                        eta_parts.append(f"{tgt}ns: {d:5.1f}d")
                eta_str = "  ".join(eta_parts)
                print(f"  [{mark}] {label:11s} {time_ns:6.1f} ns   {rate_str}   {eta_str}")

    save_cache(now_ts, now_runs)
    print(f"\n# cache written to {CACHE_PATH}")


if __name__ == "__main__":
    main()
