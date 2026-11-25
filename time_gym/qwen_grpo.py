"""Budget-only GRPO groups and a single on-policy update per group."""

import json
import sys
from pathlib import Path
from statistics import mean, pstdev

from time_gym.cases import BUDGETS, DEV_SEEDS, matrix_cases, thinking_training_cases
from time_gym.qwen_eval import DEV_EVAL_SEEDS, MODEL, REVISION, THINKING_TEMPERATURE, act, initial_chat
from time_gym.report import ReportEpisode


def training_tasks(*, thinking=False):
    seeds = [seed for seed in DEV_SEEDS if seed not in DEV_EVAL_SEEDS]
    cases = thinking_training_cases() if thinking else matrix_cases("dev", seeds)
    return [(case, budget) for case in cases for budget in BUDGETS]


def rollout_group(case, budget, sample, seed, size=8, *, thinking=False):
    episodes = [ReportEpisode(case, budget) for _ in range(size)]
    runs = [(e, initial_chat(e, thinking=thinking), [], seed + i) for i, e in enumerate(episodes)]
    active = runs
    turn = 0
    while active:
        outputs = sample([r[1] for r in active], [r[3] * 1000 + turn for r in active])
        for (episode, chat, tokens, _), output in zip(active, outputs, strict=True):
            completion = output.outputs[0]
            tokens.append((output.prompt_token_ids, list(completion.token_ids)))
            act(episode, chat, completion.text, thinking=thinking)
        active = [run for run in active if not run[0].outcome]
        turn += 1
    return [(e.metrics(), tokens) for e, _, tokens, _ in runs]


def advantages(rewards):
    center, scale = mean(rewards), pstdev(rewards) + 1e-4
    return [(reward - center) / scale for reward in rewards]


def train_worker(adapter, output, mode="plain"):
    """Worker uses Transformers 5; vLLM runs in the main process with Transformers 4."""
    import torch
    from peft import PeftModel
    from torch.nn.functional import cross_entropy
    from transformers import Qwen3_5ForConditionalGeneration

    thinking = mode == "thinking"
    model = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, adapter, is_trainable=True).cuda().train()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5 if thinking else 1e-5)
    temperature = THINKING_TEMPERATURE if thinking else 1.0
    for line in sys.stdin:
        source = Path(line.strip())
        group = json.loads(source.read_text())
        weights = advantages([metrics["reward"] for metrics, _ in group])
        optimizer.zero_grad(set_to_none=True)
        if any(weights):
            for (metrics, turns), weight in zip(group, weights, strict=True):
                count = sum(len(ids) for _, ids in turns)
                if not weight or not count:
                    continue
                for prompt, completion in turns:
                    if not completion:
                        continue
                    ids = torch.tensor([prompt + completion], device="cuda")
                    logits = model(input_ids=ids, logits_to_keep=len(completion) + 1).logits[0, :-1]
                    # One update on fresh rollouts: the on-policy ratio starts at 1.
                    loss = cross_entropy(logits.float() / temperature, ids[0, -len(completion):], reduction="sum")
                    (loss * (weight / (len(group) * count))).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
        model.save_pretrained(output, save_embedding_layers=False)
        source.with_suffix(".done").touch()


if __name__ == "__main__":
    train_worker(*sys.argv[1:])
