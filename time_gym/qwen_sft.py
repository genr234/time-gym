"""Build completion-only SFT examples from fair development traces."""

import json

from time_gym.cases import BUDGETS, DEV_SEEDS, matrix_cases, thinking_training_cases
from time_gym.policies import run_policy
from time_gym.qwen_eval import DEV_EVAL_SEEDS, act, initial_chat
from time_gym.report import ReportEpisode

FLA = "flash-linear-attention==0.4.2"
CAUSAL_CONV = ("https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1/"
               "causal_conv1d-1.6.2.post1%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl")


def _decision_note(episode, action):
    args = action["args"]
    time = (f"{episode.budget - episode.elapsed} minutes left." if episode.budget is not None
            else f"{episode.elapsed} minutes used; verify every target.")
    match action["action"]:
        case "read_source":
            topic, source = episode.sources[args["source_id"]]
            prior = [s for s in episode.seen.values() if s["topic"] == topic.name]
            evidence = ", ".join(f"{s['id']} {s['status']}" + (f" opens {s['opens']}" if s.get("opens") else "")
                                 for s in prior)
            if evidence:
                time += f" Read so far for this topic: {evidence}."
            return (f"{time} Read {source.id}, stage {source.stage}. The topic is worth {topic.weight}; "
                    f"this read costs at most {source.bounds[1]} minutes.")
        case "estimate":
            return f"Estimate {args['low']} to {args['high']} minutes from the catalog ranges, including the save and reserve reads."
        case "save_draft":
            return f"{time} Save {len(args['findings'])} complete findings with the records already read."
        case _:
            return f"{time} Finish without spending more time on incomplete findings."


def fair_examples(*, thinking=False):
    train_seeds = [seed for seed in DEV_SEEDS if seed not in DEV_EVAL_SEEDS]
    cases = thinking_training_cases() if thinking else matrix_cases("dev", train_seeds)
    for i, case in enumerate(cases):
        budgets = [(*BUDGETS, None)[i % 4]] if thinking else (*BUDGETS, None)
        if thinking and i % 8 == 0:
            budgets.append(60)
        for budget in budgets:
            teacher = ReportEpisode(case, budget)
            run_policy(teacher, "fair")
            replay = ReportEpisode(case, budget)
            chat = initial_chat(replay, thinking=thinking)
            for action in teacher.actions:
                output = json.dumps({"tool": action["action"], "args": action["args"]}, separators=(",", ":"))
                if thinking:
                    output = f"<think>\n{_decision_note(replay, action)}\n</think>\n\n{output}"
                yield chat.copy(), output
                act(replay, chat, output, thinking=thinking)
            if replay.metrics() != teacher.metrics():
                raise ValueError("SFT trace did not replay through the evaluation protocol.")


def tokenized_examples(tokenizer, *, thinking=False):
    for chat, output in fair_examples(thinking=thinking):
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        if thinking:
            if not prompt.endswith("<think>\n"):
                raise ValueError("The tokenizer must open the thinking block.")
            output = output.removeprefix("<think>\n")
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        completion_ids = tokenizer.encode(output + tokenizer.eos_token, add_special_tokens=False)
        yield {"input_ids": prompt_ids + completion_ids,
               "completion_mask": [0] * len(prompt_ids) + [1] * len(completion_ids)}
