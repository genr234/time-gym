import json
import unittest

from time_gym.qwen_eval import evaluate


class QwenEvaluationTests(unittest.TestCase):
    def test_rounds_feed_back_tool_results_and_keep_episode_errors_local(self):
        batches = []

        def sample(chats, seeds):
            batches.append((len(chats), seeds[:2]))
            if len(batches) == 1:
                return ["not JSON", *(['{"tool":"estimate","args":{"low":9,"high":20}}'] * (len(chats) - 1))]
            self.assertTrue(all(json.loads(chat[-1]["content"].removeprefix("Tool result: "))["elapsed_minutes"] <= 1
                                for chat in chats))
            return ['{"tool":"finish","args":{}}'] * len(chats)

        rows = evaluate(sample)
        self.assertEqual(batches, [(48, [41000, 42000]), (47, [42001, 41001])])
        self.assertEqual(rows[0]["outcome"], "unparseable_output")
        self.assertTrue(all(row["outcome"] == "finished" for row in rows[1:]))
        self.assertEqual({row["split"] for row in rows}, {"dev"})
