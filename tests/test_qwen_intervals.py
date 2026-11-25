import unittest

from time_gym.qwen_intervals import paired_intervals


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
