"""Shared chat protocol and batched Time Gym evaluation for Qwen."""

import json
import re

from time_gym import request_cap
from time_gym.cases import BUDGETS, matrix_cases
from time_gym.report import ReportEpisode

MODELS = {
    "4b": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
    "9b": ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
}
SIZE_ADAPTERS = {"4b": "/cache/time-gym-sft-v3-4b", "9b": "/cache/time-gym-sft-v3-9b"}
MODEL, REVISION = MODELS["4b"]
SAMPLE_SEEDS = (41, 42)
DEV_EVAL_SEEDS = (0, 1, 2, 3)
PROGRESS_EVAL_SEEDS = (12, 13, 14, 15)
THINKING_TEMPERATURE = 0.6
PROMPT = (
    "Reply with exactly one JSON action object per turn. Read the tool result before your next action. "
    "Read costs are hidden within their listed ranges. Failed reads still cost time. "
    "For each finding, cite one current record you read from every stage of the same topic. "
    "Use the last stage's value to settle disagreement. Cite the record that opens each gated record. "
    "If a link leads to an archived record, try another path. Unsupported claims subtract value. "
    "Each save costs one minute and replaces the draft."
)


def initial_chat(episode, *, thinking=False):
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
    system = PROMPT
    if thinking:
        system += " Keep deliberation to at most two short sentences. Output one action and stop. Wait for the next tool result."
    return [{"role": "system", "content": system}, {"role": "user", "content": user + "\n\nCatalog:\n" + json.dumps(brief)}]


def action_schema(*, eta):
    """Constrain action syntax, not evidence, timing, or the choice of source."""
    def obj(**properties):
        return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}

    string = {"type": "string"}
    integer = {"type": "integer"}
    finding = obj(topic=string, value=string, source_ids={"type": "array", "items": string})
    actions = {"read_source": obj(source_id=string), "save_draft": obj(findings={"type": "array", "items": finding}),
               "finish": obj()}
    if eta:
        actions["estimate"] = obj(low=integer, high=integer)
    return {"anyOf": [obj(tool={"const": name}, args=args) for name, args in actions.items()]}


def act(episode, chat, output, *, thinking=False):
    chat.append({"role": "assistant", "content": output})
    try:
        text = output.strip()
        if thinking:
            if "</think>" in text:
                text = text.split("</think>", 1)[1].strip()
            fence = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
            if fence:
                text = fence[1]
        action = json.loads(text)
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


def _with_progress(episode, chat):
    if not episode.seen:
        return chat
    # Keep only the latest summary in the prompt. The stored chat stays unchanged.
    records = [f"{s['id']}: {s['status']}" + (f"; opens {s['opens']}" if s.get("opens") else "")
               for s in episode.seen.values()]
    return [*chat[:-1], {**chat[-1], "content": chat[-1]["content"] +
                        "\n\nSearch progress (records already read):\n" + "\n".join(records)}]


def evaluate(sample, split="dev", policy="base", *, seeds=None, progress=False, thinking=False, budgets=None,
             sample_seeds=None, limit=None):
    """Sample one action per active episode per round. Keep raw tool traces."""
    if split == "dev" and seeds is None:
        seeds = DEV_EVAL_SEEDS
    runs = []
    for case in matrix_cases(split, seeds):
        for budget in (*BUDGETS, None) if budgets is None else budgets:
            for seed in SAMPLE_SEEDS if sample_seeds is None else sample_seeds:
                episode = ReportEpisode(case, budget)
                runs.append((episode, initial_chat(episode, thinking=thinking), seed))
                if limit is not None and len(runs) >= limit:
                    break
            if limit is not None and len(runs) >= limit:
                break
        if limit is not None and len(runs) >= limit:
            break
    pending = runs
    turn = 0
    while pending:
        outputs = sample([_with_progress(episode, chat) if progress else chat for episode, chat, _ in pending],
                         [seed * 1000 + turn for _, _, seed in pending])
        for (episode, chat, _), output in zip(pending, outputs, strict=True):
            act(episode, chat, output, thinking=thinking)
        pending = [(e, h, s) for e, h, s in pending if not e.outcome]
        turn += 1
    return [{"policy": policy, "sample_seed": seed, **episode.metrics(), "draft": episode.draft,
             "actions": episode.actions, "messages": chat} for episode, chat, seed in runs]
