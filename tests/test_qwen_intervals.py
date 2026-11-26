import unittest

from time_gym.qwen_intervals import paired_intervals, value_gap


class QwenIntervalTests(unittest.TestCase):
    def test_bootstrap_resamples_case_pairs_and_reference_denominators(self):
        def rows(at_15, at_60):
            return [{"family": "branch", "seed": case, "sample_seed": sample, "budget": budget, "value": value}
                    for case in (0, 1) for sample in (41, 42)
                    for budget, value in ((15, at_15[case]), (60, at_60[case]))]

        base = rows((2, 20), (4, 40))
        sft = rows((3, 22), (7, 39))
        fair = rows((1, 2), (3, 12))
        oracle = rows((2, 4), (10, 50))
        result = paired_intervals(base, sft, fair, oracle, draws=1_000)
        self.assertEqual(result["value60"], (1.0, -1.0, 3.0))
        self.assertEqual(result["efficiency"], (-1 / 12, -0.3, 1.0))
        self.assertAlmostEqual(result["oracle_share_pp"][0], 100 / 30)
        self.assertEqual(result["oracle_share_pp"][1:], (-2.0, 30.0))
        with self.assertRaisesRegex(ValueError, "complete case"):
            paired_intervals(base, sft[:-1], fair, oracle)

    def test_eta_coverage_counts_incomplete_tasks_and_rereads_cover_all_budgets(self):
        def rows(tuned):
            return [{"family": "chain", "seed": 0, "sample_seed": sample, "budget": budget,
                     "value": 1 if budget == 60 else 0,
                     "eta_actual": 39 if budget is None and (sample == 41 or tuned) else None,
                     "eta_covered": tuned and budget is None and sample == 42,
                     "reread_minutes": 12 if tuned and budget == 30 and sample == 42 else 0}
                    for sample in (41, 42) for budget in (15, 30, 60, None)]

        base, sft = rows(False), rows(True)
        result = paired_intervals(base, sft, base, base, draws=10, include_traces=True)
        self.assertEqual(result["eta_completion_pp"], (50, 50, 50))
        self.assertEqual(result["eta_coverage_all_pp"], (50, 50, 50))
        self.assertEqual(result["reread_minutes_per_episode"], (1.5, 1.5, 1.5))

    def test_value_gap_resamples_cases_and_rejects_a_missing_pair(self):
        def rows(shift):
            return [{"family": "detour", "seed": seed, "sample_seed": sample, "budget": 60,
                     "value": seed + shift, "outcome": "finished"}
                    for seed in (200, 201) for sample in (41, 42)]

        gap = value_gap(rows(0), rows(2), draws=50)
        self.assertEqual(gap["value_gap"], (2, 2, 2))
        self.assertEqual((gap["wins"], gap["ties"], gap["losses"], gap["parse_failures"]), (4, 0, 0, {"before": 0, "after": 0}))
        with self.assertRaisesRegex(ValueError, "complete case"):
            value_gap(rows(0)[:-1], rows(2))
        expected = {("detour", seed, sample) for seed in (200, 201) for sample in (41, 42)}
        with self.assertRaisesRegex(ValueError, "expected case"):
            value_gap(rows(0)[:-1], rows(2)[:-1], expected=expected)
