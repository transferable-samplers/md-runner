#!/usr/bin/env python3
"""Summarise round trips by (length, replica count) and against swap rate.

Usage: round_trips_summary.py [round_trips.csv ...]   (default: round_trips_all.csv)
Writes round_trips_vs_size.png."""
from __future__ import annotations

import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

size = {r["sequence"]: r for r in csv.DictReader(open("swap_rates_size.csv"))}


def load(path: str) -> list[dict]:
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        s = size[r["sequence"]]
        r.update(length=int(s["length"]), n_atoms=int(s["n_atoms"]), mean_rate=float(s["mean_rate"]), min_rate=float(s["min_rate"]))
        r["trips"] = np.array([int(x) for x in r["round_trips"].split()])
    return rows


inputs = sys.argv[1:] or ["round_trips_all.csv"]
rows = [r for path in inputs for r in load(path)]
seen = set()
rows = [r for r in rows if not (r["sequence"] in seen or seen.add(r["sequence"]))]
print(f"{len(rows)} runs from {', '.join(inputs)}")

mean_trips = np.array([r["trips"].mean() for r in rows])
min_trips = np.array([r["trips"].min() for r in rows])
length = np.array([r["length"] for r in rows])
n_states = np.array([int(r["n_states"]) for r in rows])
mean_rate = np.array([r["mean_rate"] for r in rows])

print(f"{'len':>4} {'n_st':>4} {'runs':>5} | mean trips/replica: {'mean':>5} {'p5':>5} {'p95':>5} | worst replica: {'p5':>4} | runs w/ replica<5: {'%':>5} | runs w/ replica=0")
for L in sorted(set(length)):
    for n in sorted(set(n_states[length == L])):
        m = (length == L) & (n_states == n)
        print(
            f"{L:>4} {n:>4} {m.sum():>5} | {mean_trips[m].mean():>25.1f} {np.percentile(mean_trips[m], 5):>5.1f} {np.percentile(mean_trips[m], 95):>5.1f} "
            f"| {np.percentile(min_trips[m], 5):>18.0f} | {100 * (min_trips[m] < 5).mean():>22.1f} | {(min_trips[m] == 0).sum()}"
        )
print(f"overall: mean trips/replica {mean_trips.mean():.1f}, median {np.median(mean_trips):.1f}; "
      f"runs with any replica <5 trips: {(min_trips < 5).sum()} ({100 * (min_trips < 5).mean():.2f}%); with a replica at 0: {(min_trips == 0).sum()}")
print(f"runs where no replica completed a round trip: {(np.array([r['trips'].max() for r in rows]) == 0).sum()}")

ink, muted = "#333333", "#777777"
fig, (ax_len, ax_rate) = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
pair_colors = {"lower": "#2a78d6", "higher": "#eb6834"}
for L in sorted(set(length)):
    options = sorted(set(n_states[length == L]))
    for n in options:
        which = "lower" if n == options[0] else "higher"
        x = L + (-0.17 if which == "lower" else 0.17)
        v = mean_trips[(length == L) & (n_states == n)]
        lo, hi = np.percentile(v, [5, 95])
        ax_len.plot([x, x], [lo, hi], color=pair_colors[which], linewidth=2, solid_capstyle="round")
        ax_len.plot(x, np.median(v), "o", color=pair_colors[which], markersize=6, markeredgecolor="white", markeredgewidth=1)
        ax_len.annotate(str(n), (x, hi), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8, color=ink)
for which, label in [("lower", "lower replica count for that length"), ("higher", "higher replica count")]:
    ax_len.plot([], [], "o-", color=pair_colors[which], linewidth=2, label=label)
ax_len.legend(frameon=False, fontsize=9, loc="upper right")
ax_len.set_xticks(sorted(set(length)))
ax_len.set_xlabel("Sequence length (residues)")
ax_len.set_ylabel("Round trips per replica in 25 ns")
ax_len.set_title("By (length, replica count): median and 5th-95th percentile\nnumber above each bar = replicas", fontsize=10, loc="left")

ramp = dict(zip(sorted(set(n_states)), ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]))
for n in sorted(set(n_states)):
    m = n_states == n
    ax_rate.scatter(mean_rate[m], mean_trips[m], s=4, color=ramp[n], alpha=0.4, linewidths=0, rasterized=True)
    ax_rate.annotate(f"{n} replicas", (np.percentile(mean_rate[m], 97), np.percentile(mean_trips[m], 97)),
                     xytext=(4, 4), textcoords="offset points", fontsize=9, color=ink)
ax_rate.set_xlabel("Mean swap acceptance rate of run")
ax_rate.set_title("Round trips against swap acceptance\nshade = replica count", fontsize=10, loc="left")
for ax in (ax_len, ax_rate):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=muted)
ax_len.set_ylim(bottom=0)
fig.suptitle(f"REMD round trips (state 0 -> top -> 0) per replica, {len(rows):,} runs", fontsize=12, x=0.01, ha="left")
fig.tight_layout()
fig.savefig("round_trips_vs_size.png", dpi=150)
