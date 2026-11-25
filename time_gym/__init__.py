"""Time Gym's Gymnasium interface for JSON text actions and observations."""

import json
from string import printable

import gymnasium as gym

from time_gym.cases import DEV_FAMILIES, generate
from time_gym.report import INSTRUCTIONS, ReportEpisode


def request_cap(budget):
    return 120 if budget is None else budget + 2


class ReportEnv(gym.Env):
    """Actions are JSON objects with a tool name and an args object."""

    def __init__(self, family=DEV_FAMILIES[0], budget=60):
        self.family, self.budget = family, budget
        self.action_space = gym.spaces.Text(16_384, charset=printable)
        self.observation_space = gym.spaces.Text(32_768, charset=printable)
        self._done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        case_seed = int(self.np_random.integers(2**32))
        self._episode = ReportEpisode(generate(self.family, case_seed), self.budget)
        self._done = False
        return json.dumps({"instructions": INSTRUCTIONS, **self._episode.brief()}), {}

    def step(self, action):
        if self._done:
            raise gym.error.ResetNeeded("Call reset() before step().")
        try:
            parsed = json.loads(action) if self.action_space.contains(action) else None
        except (ValueError, RecursionError):
            parsed = None
        if isinstance(parsed, dict) and isinstance(parsed.get("tool"), str) and isinstance(parsed.get("args"), dict):
            observation = self._episode.step(parsed["tool"], parsed["args"])
        else:
            observation = {"error": "Expected JSON with a tool name and an args object."}
            self._episode.outcome = "unparseable_output"
        if not self._episode.outcome and len(self._episode.actions) >= request_cap(self.budget):
            self._episode.outcome = "request_limit"
        outcome = self._episode.outcome
        observation["outcome"] = outcome
        self._done = outcome is not None
        info = self._episode.metrics() if self._done else {}
        truncated = outcome == "request_limit"
        return json.dumps(observation), info.get("reward", 0.0), self._done and not truncated, truncated, info
