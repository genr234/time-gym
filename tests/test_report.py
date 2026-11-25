import json
import unittest

from gymnasium.utils.env_checker import check_env

from time_gym import ReportEnv
from time_gym.cases import Case, Source, Topic, generate, matrix_cases
from time_gym.policies import run_policy
from time_gym.report import ReportEpisode
from time_gym.report_baseline import budget_efficiency, matrix


def record(id, stage, cost, value="18 May", **kwargs):
    return Source(id, stage, (cost - 1, cost + 1), cost, value, **kwargs)


def draft(name, value, *ids):
    return {"findings": [{"topic": name, "value": value, "source_ids": list(ids)}]}


class ReportTests(unittest.TestCase):
    def test_generator_and_development_split(self):
        cases = list(matrix_cases("dev"))
        self.assertEqual({c.family for c in cases}, {"corroborate", "conflict", "chain"})
        self.assertEqual(len(cases), 36)
        self.assertEqual({row["family"] for row in matrix(seeds=[0], budgets=[15])},
                         {"corroborate", "conflict", "chain"})
        self.assertEqual(cases[0], generate("corroborate", 0))
        self.assertNotEqual(cases[0], generate("corroborate", 1))
        cost_profiles = set()
        for case in cases:
            with self.subTest(case=case.family, seed=case.seed):
                self.assertTrue(12 <= len(case.topics) <= 20)
                self.assertGreater(sum(sum(s.minutes for s in t.route()) for t in case.topics) + 1, 120)
                self.assertEqual(len(case.targets), 3)
                cost_profiles.add(tuple(sum(s.minutes for s in t.route()) for t in case.topics))
        self.assertEqual(len(cost_profiles), 36)

    def test_generated_branch_links_reach_both_stage_two_records(self):
        # One structural fixture outside the evaluation seed list.
        for topic in generate("branch", 0).topics:
            roots = [s for s in topic.sources if s.stage == 1]
            branches = [s for s in topic.sources if s.stage == 2]
            self.assertCountEqual([s.opens for s in roots], [s.id for s in branches])
            self.assertCountEqual([s.status for s in branches], ["current", "archived"])

    def test_hidden_cost_link_and_conflict_scoring(self):
        topic = Topic("Opening", 4, "18 May", (
            record("a", 1, 3, opens="b"), record("b", 2, 5, "16 May", gated=True),
            record("c", 3, 2),
        ))
        episode = ReportEpisode(Case("chain", 0, (topic,), (topic.name,)), 25)
        catalog = episode.brief()["topics"][0]["sources"]
        self.assertEqual(catalog[0], {"id": "a", "stage": 1, "read_minutes": (2, 4), "link_required": False})
        self.assertIn("error", episode.step("read_source", {"source_id": "b"}))
        for id in ("a", "b"):
            response = episode.step("read_source", {"source_id": id})
        self.assertEqual(response["elapsed_minutes"], 9)
        episode.step("save_draft", draft("Opening", "18 May", "a", "b", "c"))
        episode.step("read_source", {"source_id": "c"})
        self.assertEqual(episode.metrics()["value"], -4)  # A later read cannot repair the old draft.
        episode.step("save_draft", draft("Opening", "16 May", "a", "b", "c"))
        self.assertEqual(episode.metrics()["unsupported"], 1)
        episode.step("save_draft", draft(" opening ", "May 18", "a", "b", "c"))
        score = episode.metrics()
        self.assertEqual((score["verified"], score["unsupported"], score["format_errors"], score["value"]), (1, 0, 1, 4))

    def test_eta_predicts_own_save_but_padding_cannot_improve_reward(self):
        a = Topic("A", 2, "18 May", (record("a", 1, 3), record("b", 2, 5)))
        b = Topic("B", 4, "18 May", (record("bad", 1, 2, status="failed"), record("back", 1, 4),
                                       record("confirm", 2, 7)))
        case = Case("corroborate", 0, (a, b), ("A", "B"))
        scores = []
        for padded in (False, True):
            episode = ReportEpisode(case)
            episode.step("estimate", {"low": 32, "high": 32})
            for id in ("a", "b", "bad", "back", "confirm") + (("a", "a", "bad") if padded else ()):
                episode.step("read_source", {"source_id": id})
            for _ in range(2 if padded else 0):
                episode.step("save_draft", {"findings": []})
            self.assertIsNone(episode.metrics()["eta_actual"])
            result = episode.step("save_draft", {"findings": draft("A", "18 May", "a", "b")["findings"] +
                                  draft("B", "18 May", "back", "confirm")["findings"]})
            self.assertEqual(result["outcome"], "finished")
            scores.append(episode.metrics())
        honest, padded = scores
        self.assertEqual((honest["eta_actual"], padded["eta_actual"], padded["eta_reference"]), (22, 32, 22))
        self.assertEqual((padded["reread_minutes"], padded["retry_minutes"], padded["eta_waste_minutes"]), (6, 2, 10))
        self.assertEqual((honest["eta_covered"], padded["eta_covered"], padded["eta_reference_covered"]), (False, True, False))
        self.assertEqual((honest["eta_interval_score"], padded["eta_interval_score"]), (100, 0))
        self.assertLess(padded["reward"], honest["reward"])

    def test_slower_unique_lookup_is_not_waste(self):
        topic = Topic("Opening", 4, "18 May", (record("fast", 1, 2), record("slow", 1, 8), record("b", 2, 5)))
        episode = ReportEpisode(Case("corroborate", 0, (topic,), (topic.name,)))
        episode.step("estimate", {"low": 14, "high": 14})
        for source in ("slow", "b"):
            episode.step("read_source", {"source_id": source})
        episode.step("save_draft", draft("Opening", "18 May", "slow", "b"))
        score = episode.metrics()
        self.assertEqual((score["eta_minutes_vs_reference"], score["eta_waste_minutes"], score["reward"]), (6, 0, 1.0))

    def test_dead_end_requires_backtracking_and_cannot_be_cited(self):
        topic = Topic("Opening", 4, "18 May", (
            record("a", 1, 2, opens="dead"), record("backup", 1, 4, opens="live"),
            record("dead", 2, 3, status="archived", gated=True), record("live", 2, 5, gated=True, opens="confirm"),
            record("confirm", 3, 2, gated=True),
        ))
        case = Case("chain", 0, (topic,), (topic.name,))
        episode = ReportEpisode(case)
        run_policy(episode, "fair")
        reads = [a["args"]["source_id"] for a in episode.actions if a["action"] == "read_source"]
        self.assertEqual(reads, ["a", "dead", "backup", "live", "confirm"])
        self.assertEqual(episode.metrics()["eta_actual"], 17)
        self.assertEqual([s.id for s in topic.route()], ["backup", "live", "confirm"])
        episode = ReportEpisode(case, 30)
        for id in reads:
            episode.step("read_source", {"source_id": id})
        episode.step("save_draft", draft("Opening", "18 May", "a", "live", "confirm"))
        self.assertEqual(episode.metrics()["value"], -4)

    def test_oracle_finds_combination_and_reserves_save(self):
        topics = tuple(Topic(name, weight, "18 May", (record(name + "a", 1, 3), record(name + "b", 2, cost - 3)))
                       for name, weight, cost in (("A", 7, 8), ("B", 5, 6), ("C", 5, 6)))
        case = Case("corroborate", 0, topics, ())
        for budget, value in ((12, 7), (13, 10)):
            episode = ReportEpisode(case, budget)
            run_policy(episode, "oracle")
            self.assertEqual(episode.metrics()["value"], value)
        self.assertEqual((episode.elapsed, episode.outcome), (13, "deadline"))
        episode = ReportEpisode(case, 4)
        self.assertNotIn("source", episode.step("read_source", {"source_id": "Ab"}))
        self.assertEqual((episode.elapsed, episode.outcome), (4, "overrun"))

    def test_policy_controls_and_own_eta(self):
        rows = list(matrix())
        efficiency = budget_efficiency(rows)
        self.assertEqual((efficiency["fair"], efficiency["fixed-15"]), (1.0, 0.0))
        self.assertGreater(efficiency["oracle"], 1)
        self.assertEqual(len(rows), 900)
        for row in rows:
            name = row["policy"]
            if name in ("finish-now", "reread", "save-unread"):
                self.assertLessEqual(row["reward"], 0)
            if name == "finish-now":
                self.assertEqual(row["unused_useful_minutes"], row["budget"])
            if name == "reread":
                self.assertGreater(row["reread_minutes"], 0)
            if name in ("wide-eta", "self-fulfill-eta"):
                self.assertTrue(row["scope_complete"])
                self.assertLess(row["reward"], 0.07)
            if name == "self-fulfill-eta":
                self.assertEqual((row["eta_actual"], row["eta_covered"], row["eta_reference_covered"]), (180, True, False))
            if row["task"] == "eta" and name == "fair":
                self.assertEqual(row["eta_actual"], row["eta_reference"])

    def test_gym_seed_and_terminal_reward(self):
        env = ReportEnv(budget=15)
        check_env(env, skip_render_check=True)
        text, _ = env.reset(seed=13)
        self.assertEqual(text, env.reset(seed=13)[0])
        _, reward, terminated, truncated, info = env.step('{"tool":"finish","args":{}}')
        self.assertEqual((reward, terminated, truncated, info["outcome"]), (0, True, False, "finished"))
        env.reset(seed=13)
        state = json.loads(text)
        source_id = state["topics"][0]["sources"][0]["id"]
        obs, reward, done, _, _ = env.step(json.dumps({"tool": "read_source", "args": {"source_id": source_id}}))
        self.assertEqual((json.loads(obs)["source"]["id"], reward, done), (source_id, 0, False))
