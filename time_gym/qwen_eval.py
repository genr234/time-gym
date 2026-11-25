"""Shared chat protocol and batched Time Gym evaluation for Qwen."""

import json

from time_gym import request_cap
from time_gym.cases import BUDGETS, matrix_cases
from time_gym.report import ReportEpisode

MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SAMPLE_SEEDS = (41, 42)
DEV_EVAL_SEEDS = (0, 1, 2, 3)
PROMPT = (
    "Reply with exactly one JSON action object per turn. Read the tool result before your next action. "
    "Read costs are hidden within their listed ranges. Failed reads still cost time. "
    "For each finding, cite one current record you read from every stage of the same topic. "
    "Use the last stage's value to settle disagreement. Cite the record that opens each gated record. "
    "If a link leads to an archived record, try another path. Unsupported claims subtract value. "
    "Each save costs one minute and replaces the draft."
)


def initial_chat(episode):
    brief = episode.brief()
    if episode.budget is None:
        task = ("ETA task: call estimate exactly once, first. Give an 80% interval for your own time to verify "
                "and save all listed topics. The first complete verified save ends the task. "
                "Re-reads, retries, invalid actions, and extra saves lower reward.")
        examples = ['{"tool":"estimate","args":{"low":42,"high":67}}']
    else:
        task = f"Budget task: you have {episode.budget} minutes. Do not call estimate. Maximize verified report value."
        examples = []
    stages = max(s["stage"] for t in brief["topics"] for s in t["sources"])
    examples.extend([
        '{"tool":"read_source","args":{"source_id":"example-1"}}',
        json.dumps({"tool": "save_draft", "args": {"findings": [{"topic": "Example opening date",
                    "value": "18 May", "source_ids": [f"example-{i}" for i in range(1, stages + 1)]}]}}),
        '{"tool":"finish","args":{}}',
    ])
    user = task + "\n\nAllowed actions (examples only; use actual catalog IDs and read values):\n" + "\n".join(examples)
    return [{"role": "system", "content": PROMPT}, {"role": "user", "content": user + "\n\nCatalog:\n" + json.dumps(brief)}]


def action_schema(*, eta):
    """Constrain action syntax, not evidence, timing, or the choice of source."""
    def obj(**properties):
        return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}

    string = {"type": "string"}
    integer = {"type": "integer"}
    finding = obj(topic=string, value=string, source_ids={"type": "array", "items": string, "minItems": 2, "maxItems": 3})
    actions = {"read_source": obj(source_id=string), "save_draft": obj(findings={"type": "array", "items": finding}),
               "finish": obj()}
    if eta:
        actions["estimate"] = obj(low=integer, high=integer)
    return {"anyOf": [obj(tool={"const": name}, args=args) for name, args in actions.items()]}


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
