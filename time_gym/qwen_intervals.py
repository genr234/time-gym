"""Paired case bootstrap for the Qwen budget comparisons."""

import argparse
import json
from pathlib import Path

import numpy as np

from time_gym.qwen_eval import DEV_EVAL_SEEDS, SAMPLE_SEEDS
from time_gym.report_baseline import matrix


def paired_intervals(before, after, fair, oracle, *, draws=20_000, seed=41, include_traces=False):
    def values(rows):
        result = {}
        for row in rows:
            if row["budget"] not in (15, 60):
                continue
            key = row["family"], row["seed"], row["sample_seed"], row["budget"]
            if key in result:
                raise ValueError(f"Duplicate episode: {key}")
            result[key] = row["value"]
        return result

    a, b, f, o = map(values, (before, after, fair, oracle))
    cases = sorted({key[:2] for key in a})
    expected = {(*case, sample, budget) for case in cases for sample in SAMPLE_SEEDS for budget in (15, 60)}
    if not cases or any(v.keys() != expected for v in (a, b, f, o)):
        raise ValueError("Need the same complete case, sample, and budget pairs for all policies.")

    if include_traces:
        def trace_rows(rows):
            keyed = {(r["family"], r["seed"], r["sample_seed"], r["budget"]): r for r in rows}
            full = expected | {(*case, sample, budget) for case in cases for sample in SAMPLE_SEEDS
                               for budget in (30, None)}
            if keyed.keys() != full or len(keyed) != len(rows):
                raise ValueError("Need complete paired traces at all budgets and ETA.")
            return keyed

        traces = list(map(trace_rows, (before, after)))

    def case_values(case):
        k15 = [(*case, sample, 15) for sample in SAMPLE_SEEDS]
        k60 = [(*case, sample, 60) for sample in SAMPLE_SEEDS]
        delta15 = np.mean([b[k] - a[k] for k in k15])
        delta60 = np.mean([b[k] - a[k] for k in k60])
        fair_gain = np.mean([f[y] - f[x] for x, y in zip(k15, k60)])
        oracle_value = np.mean([o[k] for k in k60])
        budget = (delta60, delta60 - delta15, fair_gain, oracle_value)
        if not include_traces:
            return budget

        def task_values(rows):
            eta = [rows[(*case, sample, None)] for sample in SAMPLE_SEEDS]
            reread = [rows[(*case, sample, b)]["reread_minutes"] for sample in SAMPLE_SEEDS for b in (15, 30, 60, None)]
            return (100 * np.mean([r["eta_actual"] is not None for r in eta]),
                    100 * np.mean([r["eta_covered"] is True for r in eta]), np.mean(reread))

        return (*budget, *(b - a for a, b in zip(*map(task_values, traces), strict=True)))

    data = np.array([case_values(case) for case in cases])
    rng = np.random.default_rng(seed)
    strata = [np.array([i for i, case in enumerate(cases) if case[0] == family])
              for family in sorted({case[0] for case in cases})]
    sample = np.concatenate([rng.choice(indices, size=(draws, len(indices)), replace=True) for indices in strata], axis=1)

    def metrics(totals):
        value, gain, fair_gain, oracle_value, *trace_metrics = totals.T
        if np.any(fair_gain <= 0) or np.any(oracle_value <= 0):
            raise ValueError("Bootstrap requires positive fair gain and oracle value.")
        return np.array([value, gain / fair_gain, 100 * value / oracle_value, *trace_metrics]).T

    point = metrics(data.mean(axis=0))
    low, high = np.quantile(metrics(data[sample].mean(axis=1)), [0.025, 0.975], axis=0)
    names = ("value60", "efficiency", "oracle_share_pp")
    if include_traces:
        names += ("eta_completion_pp", "eta_coverage_all_pp", "reread_minutes_per_episode")
    return dict(zip(names, zip(point.tolist(), low.tolist(), high.tolist())))


def value_gap(before, after, *, draws=20_000, seed=41):
    """Paired 60-minute value gap. Resample whole cases. Reject incomplete pairs."""
    def index(rows):
        result = {}
        for row in rows:
            if row.get("budget") != 60:
                raise ValueError("Use only 60-minute rows.")
            key = row["family"], row["seed"], row["sample_seed"]
            if key in result:
                raise ValueError(f"Duplicate episode: {key}")
            result[key] = row
        return result

    left, right = index(before), index(after)
    if not left or left.keys() != right.keys():
        raise ValueError("Need the same complete case and sample pairs.")
    cases = sorted({key[:2] for key in left})
    samples = sorted({key[2] for key in left})
    if any(len([key for key in left if key[:2] == case]) != len(samples) for case in cases):
        raise ValueError("Every case needs the same sample seeds.")
    gaps = np.array([np.mean([right[(*case, sample)]["value"] - left[(*case, sample)]["value"] for sample in samples])
                     for case in cases])
    draws = np.random.default_rng(seed).choice(len(cases), size=(draws, len(cases)), replace=True)
    point = float(gaps.mean())
    low, high = np.quantile(gaps[draws].mean(axis=1), [0.025, 0.975])
    paired = [right[key]["value"] - left[key]["value"] for key in sorted(left)]
    return {
        "value_gap": (point, float(low), float(high)),
        "wins": sum(gap > 0 for gap in paired),
        "ties": sum(gap == 0 for gap in paired),
        "losses": sum(gap < 0 for gap in paired),
        "parse_failures": {name: sum(row["outcome"] == "unparseable_output" for row in rows)
                           for name, rows in (("before", before), ("after", after))},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("dev", "held-out"))
    parser.add_argument("--include-traces", action="store_true", help="Include ETA and re-read intervals.")
    parser.add_argument("--value-gap", action="store_true", help="Report a 60-minute paired value gap.")
    parser.add_argument("traces", nargs="+", type=Path, help="Base, SFT, and optionally GRPO JSONL, in that order.")
    args = parser.parse_args()
    seeds = DEV_EVAL_SEEDS if args.split == "dev" else None
    refs = list(matrix(args.split, seeds=seeds, task="budget", samples=len(SAMPLE_SEEDS), sample_seed=SAMPLE_SEEDS[0]))
    fair, oracle = ([r for r in refs if r["policy"] == policy] for policy in ("fair", "oracle"))
    models = [[json.loads(line) for line in path.read_text().splitlines()] for path in args.traces]
    labels = ["base", "sft", "grpo"][:len(models)]
    if args.value_gap:
        if len(models) != 2:
            parser.error("The value gap needs the base file, then the tuned file.")
        print(value_gap(models[0], models[1]))
        return
    if len(models) not in (2, 3) or any(
        not rows or any(r["policy"] != label or r["split"] != args.split for r in rows)
        for rows, label in zip(models, labels)
    ):
        parser.error("Use matching base, SFT, and optionally GRPO traces.")
    for i in range(1, len(models)):
        results = paired_intervals(models[i-1], models[i], fair, oracle, include_traces=args.include_traces)
        print(f"{labels[i]} − {labels[i-1]}: {results}")


if __name__ == "__main__":
    main()
