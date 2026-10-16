"""Summarize REMD throughput from a generate_remd hydra log (openmmtools "Iteration took" lines).

Usage: python helpers/summarize_benchmark.py <log> [<log> ...]
Per-iteration cost is swap_interval * timestep_fs per replica; reports aggregate ns/day
(summed over replicas), per-replica ns/day and the projected wall time for a 2 us production run.
"""

import re
import sys

ITER_RE = re.compile(r"Iteration took ([\d.]+)s")
WARM_RE = re.compile(r"Warmup done, (\d+) iterations")
TEMPS_RE = re.compile(r"with (\d+) replicas")
CALC_RE = re.compile(r"\(([\d,]+) \* ([\d.]+) fs per swap attempt\)")
PLAT_RE = re.compile(r"Platform name: (\w+) properties: (.*)")


def summarize(path):
    text = open(path).read()
    times = [float(x) for x in ITER_RE.findall(text)]
    n_states = int(TEMPS_RE.search(text).group(1))
    steps, dt = CALC_RE.search(text).groups()
    ns_per_iter_per_replica = int(steps.replace(",", "")) * float(dt) * 1e-6
    warm = WARM_RE.search(text)
    n_warm = int(warm.group(1)) if warm else 0
    prod = times[n_warm:] if len(times) > n_warm else times
    prod = prod[1:] if len(prod) > 3 else prod  # drop first production iteration (context setup)
    mean = sum(prod) / len(prod)
    per_rep = ns_per_iter_per_replica / mean * 86400
    plat = PLAT_RE.search(text)
    hours_2us = 2000 / per_rep * 24
    print(f"{path}")
    print(f"  platform={plat.group(1) if plat else '?'} {plat.group(2) if plat else ''}  replicas={n_states}")
    print(f"  iterations timed={len(prod)} (of {len(times)} incl. {n_warm} warmup)  mean {mean:.2f} s/iter")
    print(f"  per-replica {per_rep:.1f} ns/day   aggregate {per_rep * n_states:.0f} ns/day")
    print(f"  projected 2 us production: {hours_2us:.1f} h = {hours_2us / 24:.2f} days")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        summarize(p)
