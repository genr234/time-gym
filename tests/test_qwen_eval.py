import json
import unittest

from time_gym.cases import generate
from time_gym.qwen_eval import action_schema, evaluate, initial_chat
from time_gym.qwen_sft import fair_examples
from time_gym.report import ReportEpisode


class QwenEvaluationTests(unittest.TestCase):
    def test_action_schema_pairs_each_allowed_tool_with_its_arguments(self):
        for eta in (False, True):
            variants = {s["properties"]["tool"]["const"]: s for s in action_schema(eta=eta)["anyOf"]}
            expected = {"read_source": ["source_id"], "save_draft": ["findings"], "finish": []}
            if eta:
                expected["estimate"] = ["low", "high"]
            self.assertEqual(set(variants), set(expected))
            for name, fields in expected.items():
                self.assertEqual(variants[name]["required"], ["tool", "args"])
                self.assertFalse(variants[name]["additionalProperties"])
                self.assertEqual(variants[name]["properties"]["args"]["required"], fields)

    def test_citation_count_is_validated_by_the_simulator_not_the_decoder(self):
        schema = action_schema(eta=False)
        save = next(s for s in schema["anyOf"] if s["properties"]["tool"]["const"] == "save_draft")
        finding = save["properties"]["args"]["properties"]["findings"]["items"]
        self.assertEqual(finding["properties"]["source_ids"], {"type": "array", "items": {"type": "string"}})
        episode = ReportEpisode(generate("corroborate", 0), 15)
        episode.step("save_draft", {"findings": [{"topic": "Example", "value": "18 May", "source_ids": ["06-77"]}]})
        metrics = episode.metrics()
        self.assertEqual((metrics["elapsed"], metrics["invalid_actions"], metrics["unsupported"], metrics["value"]),
                         (1, 1, 0, 0))

    def test_prompt_lists_only_allowed_actions_and_cites_every_stage(self):
        for family, stages in (("chain", 2), ("conflict", 3)):
            for budget in (15, None):
                chat = initial_chat(ReportEpisode(generate(family, 1), budget))
                self.assertNotIn("estimate", chat[0]["content"])
                text = chat[1]["content"]
                self.assertTrue(text.startswith("ETA task:" if budget is None else "Budget task: you have 15 minutes."))
                actions = {a["tool"]: a["args"] for a in map(json.loads, text.split("\n\n")[1].splitlines()[1:])}
                expected = {"read_source", "save_draft", "finish"}
                self.assertEqual(set(actions), expected | ({"estimate"} if budget is None else set()))
                self.assertEqual(len(actions["save_draft"]["findings"][0]["source_ids"]), stages)

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
        self.assertEqual(batches, [(96, [41000, 42000]), (95, [42001, 41001])])
        self.assertEqual(rows[0]["outcome"], "unparseable_output")
        self.assertTrue(all(row["outcome"] == "finished" for row in rows[1:]))
        self.assertEqual({row["split"] for row in rows}, {"dev"})

    def test_sft_uses_same_chat_and_feedback_as_evaluation(self):
        examples = iter(fair_examples())
        first, output = next(examples)
        second, _ = next(examples)
        case = generate("corroborate", 4)
        self.assertEqual(first, initial_chat(ReportEpisode(case, 15)))
        self.assertEqual(second[:2], first)
        self.assertEqual(second[2], {"role": "assistant", "content": output})
        self.assertTrue(second[3]["content"].startswith("Tool result: "))
