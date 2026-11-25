"""Run the offline matrix or collect approved OpenRouter samples as JSONL."""

import argparse
import json
import os
from collections import defaultdict
from http.client import HTTPException
from urllib.request import Request, urlopen

from time_gym import request_cap
from time_gym.cases import BUDGETS, matrix_cases
from time_gym.policies import BUDGET_POLICIES, ETA_POLICIES, run_policy
from time_gym.report import INSTRUCTIONS, ReportEpisode

MODEL = "thinkingmachines/inkling-2-mini"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


def tool(name, description, **properties):
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": list(properties), "additionalProperties": False}}}


STRING = {"type": "string"}
MINUTES = {"type": "integer", "minimum": 1, "maximum": 10_000}
TOOLS = [
    tool("estimate", "Give the ETA interval in simulated minutes, in the ETA task only.", low=MINUTES, high=MINUTES),
    tool("read_source", "Read a catalog source. The result gives its actual cost.", source_id=STRING),
    tool("save_draft", "Replace the complete draft. This takes one minute.", findings={
        "type": "array", "items": {"type": "object", "properties": {
            "topic": STRING, "value": STRING, "source_ids": {
                "type": "array", "items": STRING, "minItems": 2, "maxItems": 3}},
            "required": ["topic", "value", "source_ids"], "additionalProperties": False}}),
    tool("finish", "Deliver the last saved draft and end the episode."),
]


def openrouter_sample(messages, api_key, *, seed, reasoning):
    request = Request(ENDPOINT, json.dumps({
        "model": MODEL, "messages": messages, "tools": TOOLS, "tool_choice": "auto",
        "provider": {"require_parameters": True}, "temperature": 0.7, "seed": seed,
        "reasoning": {"enabled": reasoning}, "max_tokens": 4096,
    }).encode(), {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    with urlopen(request, timeout=90) as response:
        body = json.load(response)
    if isinstance(body, dict) and body.get("error"):
        raise OSError("OpenRouter returned an error response.")
    return body["choices"][0]["message"]


def run_inkling(episode, sample, *, max_requests):
    messages = [{"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": json.dumps(episode.brief())}]
    requests, error_type = 0, None
    while not episode.outcome and requests < max_requests:
        requests += 1
        try:
            reply = sample(messages)
        except (OSError, HTTPException) as exc:
            episode.outcome, error_type = "api_error", type(exc).__name__
            break
        except (ValueError, TypeError, KeyError, IndexError, RecursionError) as exc:
            episode.outcome, error_type = "unparseable_output", type(exc).__name__
            break
        if not isinstance(reply, dict):
            episode.outcome = "unparseable_output"
            break
        messages.append({"role": "assistant", **{k: reply[k] for k in
                         ("content", "tool_calls", "reasoning_details") if k in reply}})
        calls = reply.get("tool_calls")
        if calls is None or calls == []:
            episode.outcome = "text_only" if isinstance(reply.get("content"), str) else "unparseable_output"
            break
        try:
            if not isinstance(calls, list) or len(calls) != 1:
                raise ValueError("Expected one tool call.")
            call = calls[0]
            name, call_id = call["function"]["name"], call["id"]
            args = json.loads(call["function"]["arguments"])
            if not isinstance(name, str) or not isinstance(call_id, str) or not isinstance(args, dict):
                raise ValueError("Invalid tool message.")
        except (KeyError, TypeError, ValueError, RecursionError) as exc:
            episode.outcome, error_type = "unparseable_output", type(exc).__name__
            break
        result = episode.step(name, args)
        messages.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(result)})
    if not episode.outcome:
        episode.outcome = "request_limit"
    return {"requests": requests, "max_requests": max_requests, "error_type": error_type, "messages": messages}


def matrix(split="dev", *, seeds=None, budgets=BUDGETS, task="both", policy=None, samples=1,
           sample_seed=0, live=None, reasoning=False, max_requests=None):
    for case in matrix_cases(split, seeds):
        tasks = []
        if task in ("both", "budget"):
            tasks.extend((budget, BUDGET_POLICIES) for budget in budgets)
        if task in ("both", "eta"):
            tasks.append((None, ETA_POLICIES))
        for budget, policies in tasks:
            for name in ("inkling",) if live is not None else policies:
                if policy and name != policy:
                    continue
                for index in range(samples):
                    seed = sample_seed + index
                    episode = ReportEpisode(case, budget)
                    settings = {}
                    if live is not None:
                        settings = {"model": MODEL, "reasoning": reasoning, **run_inkling(
                            episode, lambda messages: live(messages, seed=seed, reasoning=reasoning),
                            max_requests=max_requests or request_cap(budget))}
                    else:
                        run_policy(episode, name, sample_seed=seed + case.seed)
                    yield {"policy": name, "sample_seed": seed, **settings, **episode.metrics(),
                           "draft": episode.draft, "actions": episode.actions}


def budget_efficiency(rows, reference="fair"):
    """Compare gains on the same complete set of matched cases and sample seeds."""
    paired = defaultdict(dict)
    for row in rows:
        if row["budget"] in (15, 60):
            key = (row["family"], row["seed"], row["sample_seed"], row["budget"])
            if key in paired[row["policy"]]:
                raise ValueError("Duplicate matrix row. Analyze each reasoning setting separately.")
            paired[row["policy"]][key] = row["value"]
    keys = paired[reference].keys()
    if not keys or any((*key[:-1], 75 - key[-1]) not in keys for key in keys):
        raise ValueError("Need matched reference rows at 15 and 60 minutes.")
    if any(p.keys() != keys for p in paired.values()):
        raise ValueError("Policies must use the same cases and sample seeds.")
    gains = {name: sum(values[k] * (1 if k[-1] == 60 else -1) for k in keys)
             for name, values in paired.items()}
    if gains[reference] <= 0:
        raise ValueError("Reference must gain value from the larger budget.")
    return {name: gain / gains[reference] for name, gain in gains.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "held-out"), default="dev")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--budgets", nargs="+", type=int, default=BUDGETS)
    parser.add_argument("--task", choices=("both", "budget", "eta"), default="both")
    parser.add_argument("--policy", choices=sorted({*BUDGET_POLICIES, *ETA_POLICIES, "inkling"}))
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--reasoning", choices=("on", "off"), default="off")
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--allow-paid", action="store_true")
    args = parser.parse_args()
    if min(*args.budgets, args.samples, args.max_requests if args.max_requests is not None else 1) < 1:
        parser.error("Use positive budgets, sample counts and request limits.")
    live = None
    if args.policy == "inkling":
        if not args.allow_paid or not (key := os.environ.get("OPENROUTER_API_KEY")):
            parser.error("Live calls require spend approval, --allow-paid and OPENROUTER_API_KEY.")
        def live(messages, **settings):
            return openrouter_sample(messages, key, **settings)

    for row in matrix(args.split, seeds=args.seeds, budgets=args.budgets, task=args.task, policy=args.policy,
                      samples=args.samples, sample_seed=args.sample_seed, live=live,
                      reasoning=args.reasoning == "on", max_requests=args.max_requests):
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
