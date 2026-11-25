"""Paired case bootstrap for the Qwen budget comparisons."""

import argparse
import json
from pathlib import Path

import numpy as np

from time_gym.qwen_eval import DEV_EVAL_SEEDS, SAMPLE_SEEDS
from time_gym.report_baseline import matrix


def paired_intervals(before, after, fair, oracle, *, draws=20_000, seed=41):
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

    def case_values(case):
        k15 = [(*case, sample, 15) for sample in SAMPLE_SEEDS]
        k60 = [(*case, sample, 60) for sample in SAMPLE_SEEDS]
        delta15 = np.mean([b[k] - a[k] for k in k15])
        delta60 = np.mean([b[k] - a[k] for k in k60])
        fair_gain = np.mean([f[y] - f[x] for x, y in zip(k15, k60)])
        oracle_value = np.mean([o[k] for k in k60])
        return delta60, delta60 - delta15, fair_gain, oracle_value

    data = np.array([case_values(case) for case in cases])
    rng = np.random.default_rng(seed)
    strata = [np.array([i for i, case in enumerate(cases) if case[0] == family])
              for family in sorted({case[0] for case in cases})]
    sample = np.concatenate([rng.choice(indices, size=(draws, len(indices)), replace=True) for indices in strata], axis=1)

    def metrics(totals):
        value, gain, fair_gain, oracle_value = totals.T
        if np.any(fair_gain <= 0) or np.any(oracle_value <= 0):
            raise ValueError("Bootstrap requires positive fair gain and oracle value.")
        return np.array([value, gain / fair_gain, 100 * value / oracle_value]).T

    point = metrics(data.mean(axis=0))
    low, high = np.quantile(metrics(data[sample].mean(axis=1)), [0.025, 0.975], axis=0)
    names = ("value60", "efficiency", "oracle_share_pp")
    return dict(zip(names, zip(point.tolist(), low.tolist(), high.tolist())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("dev", "held-out"))
    parser.add_argument("traces", nargs="+", type=Path, help="Base, SFT, and optionally GRPO JSONL, in that order.")
    args = parser.parse_args()
    seeds = DEV_EVAL_SEEDS if args.split == "dev" else None
    refs = list(matrix(args.split, seeds=seeds, task="budget", samples=len(SAMPLE_SEEDS), sample_seed=SAMPLE_SEEDS[0]))
    fair, oracle = ([r for r in refs if r["policy"] == policy] for policy in ("fair", "oracle"))
    models = [[json.loads(line) for line in path.read_text().splitlines()] for path in args.traces]
    labels = ["base", "sft", "grpo"][:len(models)]
    if len(models) not in (2, 3) or any(
        not rows or any(r["policy"] != label or r["split"] != args.split for r in rows)
        for rows, label in zip(models, labels)
    ):
        parser.error("Use matching base, SFT, and optionally GRPO traces.")
    for i in range(1, len(models)):
        print(f"{labels[i]} − {labels[i-1]}: {paired_intervals(models[i-1], models[i], fair, oracle)}")


if __name__ == "__main__":
    main()
