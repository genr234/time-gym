"""Shared chat protocol and batched Time Gym evaluation for Qwen."""

import json

from time_gym import request_cap
from time_gym.cases import BUDGETS, matrix_cases
from time_gym.report import INSTRUCTIONS, ReportEpisode

MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SAMPLE_SEEDS = (41, 42)
DEV_EVAL_SEEDS = (0, 1)
PROMPT = INSTRUCTIONS + (
    ' Output one JSON object per turn, with no prose: {"tool":"name","args":{}}. '
    'Tools: read_source(source_id); save_draft(findings), each with topic, value, source_ids; '
    'estimate(low, high) for ETA only; finish() to deliver your last saved report. '
    'Each save replaces the draft and costs one minute. Read tool results before your next call.'
)


def initial_chat(episode):
    return [{"role": "system", "content": PROMPT}, {"role": "user", "content": json.dumps(episode.brief())}]


def act(episode, chat, output):
    chat.append({"role": "assistant", "content": output})
    try:
        action = json.loads(output)
        tool, args = action["tool"], action["args"]
        if not isinstance(tool, str) or not isinstance(args, dict):
            raise ValueError("Invalid action fields")
    except (ValueError, TypeError, KeyError, RecursionError):
        episode.outcome = "unparseable_output"
        return
    result = episode.step(tool, args)
    chat.append({"role": "user", "content": "Tool result: " + json.dumps(result)})
    if not episode.outcome and len(episode.actions) >= request_cap(episode.budget):
        episode.outcome = "request_limit"


def evaluate(sample, split="dev", policy="base"):
    """Sample one action per active episode per round. Keep raw tool traces."""
    seeds = DEV_EVAL_SEEDS if split == "dev" else None
    runs = []
    for case in matrix_cases(split, seeds):
        for budget in (*BUDGETS, None):
            for seed in SAMPLE_SEEDS:
                episode = ReportEpisode(case, budget)
                runs.append((episode, initial_chat(episode), seed))
    pending = runs
    turn = 0
    while pending:
        outputs = sample([chat for _, chat, _ in pending], [seed * 1000 + turn for _, _, seed in pending])
        for (episode, chat, _), output in zip(pending, outputs, strict=True):
            act(episode, chat, output)
        pending = [(e, h, s) for e, h, s in pending if not e.outcome]
        turn += 1
    return [{"policy": policy, "sample_seed": seed, **episode.metrics(), "draft": episode.draft,
             "actions": episode.actions, "messages": chat} for episode, chat, seed in runs]
