"""Matched Qwen 4B and 9B comparison. This module does not start a GPU job."""

import json

from time_gym.cases import DEV_FAMILIES, HELD_OUT_FAMILIES, generate
from time_gym.policies import forecast, run_policy
from time_gym.qwen_eval import DEV_EVAL_SEEDS, MODELS, SAMPLE_SEEDS, THINKING_TEMPERATURE
from time_gym.qwen_intervals import paired_intervals
from time_gym.qwen_sft import fair_examples
from time_gym.report import ReportEpisode
from time_gym.report_baseline import matrix

# Seeds 80-99 check the forecast. Seeds 200-211 are reserved for a later frozen test.
CHECK_SEEDS = range(80, 100)
FRESH_SEEDS = range(200, 212)
SETTINGS = {
    "thinking": True,
    "temperature": THINKING_TEMPERATURE,
    "max_tokens": 1024,
    "guided_json": False,
    "context": 65536,
    "budgets": (15, 30, 60, None),
    "eval_seeds": DEV_EVAL_SEEDS,
    "sample_seeds": SAMPLE_SEEDS,
}


def teacher_rows(family, seeds):
    rows = []
    for seed in seeds:
        episode = ReportEpisode(generate(family, seed), None)
        run_policy(episode, "fair")
        rows.append(episode.metrics())
    return rows


def summarize(rows):
    complete = [row for row in rows if row["eta_actual"] is not None]
    return {
        "n": len(rows),
        "coverage": sum(row["eta_reference_covered"] for row in complete) / len(complete),
        "width": sum(row["eta_width"] for row in complete) / len(complete),
        "loss": sum(row["eta_interval_score"] for row in complete) / len(complete),
    }


def compare_forecasts():
    """Report the catalog recovery rule. Do not use frozen branch seeds to choose it."""
    return {
        "detour_train_20_79": summarize(teacher_rows("detour", range(20, 80))),
        "detour_check_80_99": summarize(teacher_rows("detour", CHECK_SEEDS)),
        "other_check_80_99": summarize([row for family in DEV_FAMILIES for row in teacher_rows(family, CHECK_SEEDS)]),
        "frozen_branch_not_used_for_fit": summarize(teacher_rows(HELD_OUT_FAMILIES[0], range(100, 112))),
    }


def arms():
    return [{"size": size, "model": model, "revision": revision, "role": role, **SETTINGS}
            for size, (model, revision) in MODELS.items() for role in ("base", "sft")]


def mock_scores(size, role):
    """Stand in for a GPU result. A wrong pairing must not look like a real gain."""
    rows = []
    for family in DEV_FAMILIES:
        for seed in DEV_EVAL_SEEDS:
            case = generate(family, seed)
            for sample in SAMPLE_SEEDS:
                for budget in (15, 60):
                    episode = ReportEpisode(case, budget)
                    run_policy(episode, "fair" if role == "sft" else "fixed-15")
                    metrics = episode.metrics()
                    if role == "base":
                        metrics["value"] = 0
                    rows.append({"policy": f"{size}-{role}", "split": "dev", "sample_seed": sample, **metrics})
    return rows


def main():
    examples = list(fair_examples(thinking=True))
    print(json.dumps({"forecast": compare_forecasts(), "teacher_turns": len(examples), "arms": arms()}, indent=2))
    refs = list(matrix("dev", seeds=DEV_EVAL_SEEDS, task="budget", samples=len(SAMPLE_SEEDS), sample_seed=SAMPLE_SEEDS[0]))
    fair = [row for row in refs if row["policy"] == "fair"]
    oracle = [row for row in refs if row["policy"] == "oracle"]
    print("mock 4b-sft minus 9b-base", paired_intervals(mock_scores("9b", "base"), mock_scores("4b", "sft"), fair, oracle, draws=20))
    print("example forecast", forecast(ReportEpisode(generate("detour", 80), None).brief()))


if __name__ == "__main__":
    main()
