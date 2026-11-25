"""Draw three offline figures from matrix JSONL. Print the source numbers."""

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from time_gym.policies import BUDGET_POLICIES, ETA_POLICIES
from time_gym.report_baseline import budget_efficiency

COLORS = {"oracle": "#3c414e", "fair": "#007ea7", "fixed-15": "#6d5192", "random": "#9c5900",
          "finish-now": "#536b5d", "reread": "#b77079", "save-unread": "#b43d4a", "wide-eta": "#b43d4a",
          "self-fulfill-eta": "#9c5900", "inkling": "#2c7b51"}
OUTCOMES = ("api_error", "request_limit", "unparseable_output", "text_only", "overrun")


def failures(row):
    if row["outcome"] in OUTCOMES:
        yield row["outcome"]
    for field, label in (("unsupported", "Unsupported claim"), ("reread_minutes", "Re-read"),
                         ("retry_minutes", "Failed-source retry"), ("unused_useful_minutes", "Unused useful time"),
                         ("invalid_actions", "Invalid action"), ("format_errors", "Format error")):
        if row[field]:
            yield label
    if row["task"] == "eta":
        if not row["scope_complete"]:
            yield "Incomplete scope"
        if row["eta_covered"] is False:
            yield "ETA miss"
        if row["eta"] is None:
            yield "Missing ETA"


def summarize(rows):
    budgets = {p: {b: mean(r["value"] for r in rows if r["policy"] == p and r["budget"] == b)
                   for b in sorted({r["budget"] for r in rows if r["policy"] == p and r["budget"] is not None})}
               for p in (*BUDGET_POLICIES, "inkling") if any(r["policy"] == p and r["budget"] for r in rows)}
    eta = {}
    for policy in (*ETA_POLICIES, "inkling"):
        group = [r for r in rows if r["policy"] == policy and r["task"] == "eta"]
        if group:
            eta[policy] = {"episodes": len(group), "scope_complete": sum(r["scope_complete"] for r in group),
                           "forecasts": sum(r["eta"] is not None for r in group), "reward": mean(r["reward"] for r in group)}
            for key in ("eta_width", "eta_covered", "eta_bias", "eta_interval_score"):
                values = [r[key] for r in group if r[key] is not None]
                eta[policy][key] = mean(values) if values else None
    return {"episodes": len(rows), "budget_value": budgets, "eta": eta,
            "failure_counts": Counter(f for r in rows for f in failures(r)),
            "efficiency": budget_efficiency(rows) if {15, 60} <= budgets.get("fair", {}).keys() else {}}


def figure_columns(splits, title, subtitle):
    fig, axes = plt.subplots(1, len(splits), figsize=(5.2 * len(splits), 4.4), squeeze=False)
    fig.subplots_adjust(top=0.75, bottom=0.24, left=0.12, right=0.98, wspace=0.40)
    fig.suptitle(title, x=0.12, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.12, 0.88, subtitle, fontsize=9, color="#4d5964")
    for split, ax in zip(splits, axes[0]):
        ax.set_title({"dev": "Development", "held-out": "Held-out family"}[split], loc="left", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e9ed")
        ax.set_axisbelow(True)
    return fig, axes[0]


def draw(rows, output):
    splits = {s: summarize([r for r in rows if r["split"] == s]) for s in ("dev", "held-out")
              if any(r["split"] == s for r in rows)}
    if not splits:
        raise ValueError("No matrix rows.")
    output.mkdir(parents=True, exist_ok=True)
    if any(s["budget_value"] for s in splits.values()):
        fig, axes = figure_columns(splits, "More time increases report value for the reference policies",
                                   "Mean verified weight minus unsupported weight, at each simulated budget.")
        for ax, summary in zip(axes, splits.values()):
            for p, values in summary["budget_value"].items():
                style = {"reread": (":", "o"), "finish-now": ("--", "x")}.get(p, ("-", "o"))
                ax.plot(list(values), list(values.values()), linestyle=style[0], marker=style[1],
                        color=COLORS[p], label=p, markersize=5)
            ax.set(xlabel="Budget (min)", ylabel="Mean report value", xticks=sorted({
                b for values in summary["budget_value"].values() for b in values}))
        lo = min(ax.get_ylim()[0] for ax in axes)
        hi = max(ax.get_ylim()[1] for ax in axes)
        for ax in axes:
            ax.set_ylim(lo, hi)
        handles = dict((h.get_label(), h) for ax in axes for h in ax.get_legend_handles_labels()[0])
        fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=min(4, len(handles)), frameon=False)
        fig.savefig(output / "budget-value.png", dpi=160)
        plt.close(fig)

    if any(s["eta"] for s in splits.values()):
        fig, axes = figure_columns(splits, "Coverage alone does not make an ETA useful",
                                   "Intervals with no forecast are excluded. The dashed line shows 80% target coverage.")
        max_width = max((m["eta_width"] or 0 for s in splits.values() for m in s["eta"].values()), default=0)
        for ax, summary in zip(axes, splits.values()):
            ax.axhline(0.8, color="#7a858d", linestyle="--", lw=1)
            for p, m in summary["eta"].items():
                if m["eta_width"] is not None:
                    x, y = 1 + m["eta_width"], m["eta_covered"]
                    ax.scatter(x, y, label=p, color=COLORS[p], zorder=3)
            ticks = [n for n in (1, 11, 101, 1001, 10001) if n <= max(30, 1.5 * max_width)]
            ax.set(xscale="log", xlim=(0.72, max(30, 1.5 * max_width)), ylim=(-0.08, 1.08), xticks=ticks,
                   xticklabels=[str(x - 1) for x in ticks], xlabel="Mean interval width (min, log spacing)",
                   ylabel="Fraction covered")
            ax.minorticks_off()
        handles = dict((h.get_label(), h) for ax in axes for h in ax.get_legend_handles_labels()[0])
        if handles:
            fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=min(4, len(handles)), frameon=False)
        fig.savefig(output / "eta-calibration.png", dpi=160)
        plt.close(fig)

    fig, axes = figure_columns(splits, "Episode flags by type",
                               "An episode can have more than one flag. Zero-count types are not shown.")
    labels = sorted({f for s in splits.values() for f in s["failure_counts"]})
    for ax, summary in zip(axes, splits.values()):
        if labels:
            values = [100 * summary["failure_counts"][f] / summary["episodes"] for f in labels]
            ax.barh(labels, values, color=COLORS["fair"])
            ax.invert_yaxis()
        else:
            ax.text(0.5, 0.5, "No episode flags", ha="center", transform=ax.transAxes)
        ax.set(xlim=(0, 100), xlabel="Percent of all episodes in split")
    fig.subplots_adjust(left=0.22, wspace=0.76)
    fig.savefig(output / "failure-types.png", dpi=160)
    plt.close(fig)
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("figures"))
    args = parser.parse_args()
    rows = []
    for path in args.inputs:
        with path.open() as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    print(json.dumps(draw(rows, args.output), indent=2))


if __name__ == "__main__":
    main()
