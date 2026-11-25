"""Build completion-only SFT examples from fair development traces."""

import json

from time_gym.cases import BUDGETS, DEV_SEEDS, matrix_cases
from time_gym.policies import run_policy
from time_gym.qwen_eval import DEV_EVAL_SEEDS, act, initial_chat
from time_gym.report import ReportEpisode


def fair_examples():
    train_seeds = [seed for seed in DEV_SEEDS if seed not in DEV_EVAL_SEEDS]
    for case in matrix_cases("dev", train_seeds):
        for budget in (*BUDGETS, None):
            teacher = ReportEpisode(case, budget)
            run_policy(teacher, "fair")
            replay = ReportEpisode(case, budget)
            chat = initial_chat(replay)
            for action in teacher.actions:
                output = json.dumps({"tool": action["action"], "args": action["args"]}, separators=(",", ":"))
                yield chat.copy(), output
                act(replay, chat, output)
            if replay.metrics() != teacher.metrics():
                raise ValueError("SFT trace did not replay through the evaluation protocol.")


def tokenized_examples(tokenizer):
    for chat, output in fair_examples():
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        completion_ids = tokenizer.encode(output + tokenizer.eos_token, add_special_tokens=False)
        yield {"input_ids": prompt_ids + completion_ids,
               "completion_mask": [0] * len(prompt_ids) + [1] * len(completion_ids)}
