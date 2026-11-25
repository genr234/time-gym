import json
from types import SimpleNamespace
import unittest

from time_gym.cases import generate
from time_gym.policies import run_policy
from time_gym.qwen_eval import PROGRESS_EVAL_SEEDS, act, action_schema, evaluate, initial_chat
from time_gym.qwen_grpo import advantages, rollout_group, training_tasks
from time_gym.qwen_sft import fair_examples
from time_gym.report import ReportEpisode


class QwenEvaluationTests(unittest.TestCase):
    def test_thinking_parses_one_action_and_preserves_raw_output(self):
        action = '{"tool":"finish","args":{}}'
        for text, outcome in (
            (f'consider {{}}\n</think>\n```json\n{action}\n```', "finished"),
            (f'<think>done</think>{action}', "finished"),
            (action, "finished"),
            (f'<think>not closed {action}', "unparseable_output"),
            (f'done</think>{action} {{}}', "unparseable_output"),
        ):
            with self.subTest(text=text):
                episode = ReportEpisode(generate("chain", 0), 15)
                chat = initial_chat(episode)
                act(episode, chat, text, thinking=True)
                self.assertEqual(episode.outcome, outcome)
                self.assertEqual(chat[2]["content"], text)

    def test_grpo_groups_use_training_cases_and_terminal_rewards(self):
        tasks = training_tasks()
        self.assertEqual({(case.split, case.seed) for case, _ in tasks}, {("dev", s) for s in range(4, 12)})
        case, budget = tasks[0]
        source = case.topics[0].sources[0].id
        steps = iter([["read_source", "finish"], ["finish"]])

        def sample(chats, seeds):
            return [SimpleNamespace(prompt_token_ids=[1], outputs=[SimpleNamespace(token_ids=[2], text=json.dumps(
                {"tool": tool, "args": {"source_id": source} if tool == "read_source" else {}}))])
                    for tool in next(steps)]

        group = rollout_group(case, budget, sample, 0, size=2)
        self.assertEqual([len(tokens) for _, tokens in group], [2, 1])
        self.assertEqual([r["outcome"] for r, _ in group], ["finished", "finished"])
        self.assertEqual(advantages([r["reward"] for r, _ in group]), [0, 0])
        self.assertGreater(advantages([-0.2, 0.0, 0.1])[-1], 0)

    def test_thinking_training_sees_detours_but_not_evaluation_cases(self):
        tasks = training_tasks(thinking=True)
        self.assertEqual({(c.split, c.family, c.seed) for c, _ in tasks},
                         {("dev", f, s) for f in ("corroborate", "conflict", "chain", "detour") for s in range(20, 80)})
        episode = ReportEpisode(generate("detour", 23))
        run_policy(episode, "fair")
        self.assertTrue(episode.scope_complete)
        self.assertTrue(any(a.get("source", {}).get("status") in {"archived", "failed"}
                            for a in episode.actions))
        examples = list(fair_examples(thinking=True))
        self.assertTrue(2000 <= len(examples) <= 3000)
        self.assertTrue(any(" archived" in note and " opens " in note for _, note in examples))

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

    def test_held_out_evaluation_uses_the_frozen_branch_matrix(self):
        rows = evaluate(lambda chats, seeds: ['{"tool":"finish","args":{}}'] * len(chats),
                        split="held-out")
        self.assertEqual(len(rows), 96)
        self.assertEqual({(r["family"], r["seed"]) for r in rows}, {("branch", seed) for seed in range(100, 112)})
        self.assertEqual({(r["budget"], r["sample_seed"]) for r in rows},
                         {(budget, seed) for budget in (15, 30, 60, None) for seed in (41, 42)})

    def test_progress_reports_only_observed_reads_and_does_not_change_the_episode(self):
        target = initial_chat(ReportEpisode(generate("chain", 12), 60))

        def run(progress):
            prompts = []

            def sample(chats, seeds):
                turn = len(prompts)
                index = chats.index(target) if turn == 0 else 0
                prompts.append([m["content"] for m in chats[index]])
                tool = '{"tool":"finish","args":{}}'
                outputs = [tool] * len(chats)
                if turn < 2:
                    source_id = ("12-13", "12-67")[turn]
                    outputs[index] = json.dumps({"tool": "read_source", "args": {"source_id": source_id}})
                return outputs

            rows = evaluate(sample, seeds=PROGRESS_EVAL_SEEDS, progress=progress)
            return rows, prompts

        control, plain = run(False)
        treatment, summary = run(True)
        self.assertEqual(control, treatment)
        self.assertEqual(len(treatment), 96)
        self.assertEqual({(r["split"], r["seed"]) for r in treatment}, {("dev", s) for s in range(12, 16)})
        self.assertEqual(summary[:2], plain[:2])  # A rejected gated read adds no progress.
        self.assertEqual(summary[2][:-1], plain[2][:-1])
        self.assertEqual(summary[2][-1], plain[2][-1] +
                         "\n\nSearch progress (records already read):\n12-67: archived; opens 12-13")

    def test_sft_uses_same_chat_and_feedback_as_evaluation(self):
        examples = iter(fair_examples())
        first, output = next(examples)
        second, _ = next(examples)
        case = generate("corroborate", 4)
        self.assertEqual(first, initial_chat(ReportEpisode(case, 15)))
        self.assertEqual(second[:2], first)
        self.assertEqual(second[2], {"role": "assistant", "content": output})
        self.assertTrue(second[3]["content"].startswith("Tool result: "))
