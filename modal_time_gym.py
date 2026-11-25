"""Bounded on-demand Modal jobs for Qwen and Time Gym."""

import json
from pathlib import Path
from time import monotonic

import modal

app = modal.App("time-gym-qwen")
cache = modal.Volume.from_name("time-gym-qwen-weights", create_if_missing=True)
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.19.0", "gymnasium>=1.0,<2")
    .env({"HF_HOME": "/cache", "HF_HUB_DISABLE_XET": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
)
eval_image = vllm_image.add_local_python_source("time_gym")
GPU = "L40S"
EVAL_TIMEOUT = 1500


@app.function(image=eval_image, cpu=2, memory=4096, timeout=120, startup_timeout=120, scaledown_window=2)
def check_image():
    start = monotonic()
    try:
        import time_gym
        import torch
        import vllm
        from vllm import ModelRegistry

        assert not torch.cuda.is_available(), "The preflight must run without a GPU"
        arch = "Qwen3_5ForConditionalGeneration"
        assert arch in ModelRegistry.get_supported_archs(), f"Missing model registration: {arch}"
        print(f"vLLM {vllm.__version__}; registered {arch}; imported {time_gym.__name__}", flush=True)
    finally:
        print(f"CPU_JOB check_image 2vCPU 4GiB {monotonic() - start:.1f}s", flush=True)


def inference(adapter_enabled, memory_fraction=0.86):
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from time_gym.qwen_eval import MODEL, REVISION, action_schema

    engine = LLM(model=MODEL, revision=REVISION, language_model_only=True, dtype="bfloat16",
                 max_model_len=8192, gpu_memory_utilization=memory_fraction, enforce_eager=True,
                 enable_lora=adapter_enabled)
    tokenizer = engine.get_tokenizer()
    schemas = {eta: action_schema(eta=eta) for eta in (True, False)}

    def sample(chats, seeds, lora):
        prompts = tokenizer.apply_chat_template(chats, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)
        params = [SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=384, seed=seed,
                                 structured_outputs=StructuredOutputsParams(json=schemas[chat[1]["content"].startswith("ETA task:")]))
                  for chat, seed in zip(chats, seeds, strict=True)]
        return engine.generate(prompts, params, use_tqdm=False, lora_request=lora)

    return sample


@app.function(image=eval_image, gpu=GPU, cpu=4, memory=16384, timeout=EVAL_TIMEOUT, startup_timeout=360,
              scaledown_window=2, max_containers=1, volumes={"/cache": cache})
def baseline(adapter: str = "", split: str = "dev", policy: str = "base"):
    from vllm.lora.request import LoRARequest
    from time_gym.qwen_eval import evaluate

    start = monotonic()
    try:
        sample = inference(bool(adapter))
        lora = LoRARequest(policy, 1, adapter) if adapter else None
        return evaluate(lambda chats, seeds: [r.outputs[0].text for r in sample(chats, seeds, lora)],
                        split=split, policy=policy)
    finally:
        print(f"GPU_JOB baseline {GPU} {monotonic() - start:.1f}s", flush=True)


@app.local_entrypoint()
def main(output: str, adapter: str = "", split: str = "dev", policy: str = ""):
    if split not in ("dev", "held-out"):
        raise ValueError("Use dev or held-out.")
    rows = baseline.remote(adapter, split, policy or ("sft" if adapter else "base"))
    Path(output).write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"Wrote {len(rows)} model episodes to {output}")


rl_image = vllm_image.run_commands(
    "python -m pip install uv",
    "uv venv --system-site-packages /opt/train",
    "uv pip install --python /opt/train/bin/python transformers==5.3.0 peft==0.18.1",
).add_local_python_source("time_gym")
SFT_ADAPTER = "/cache/time-gym-sft-v1"
RL_ADAPTER = "/cache/time-gym-grpo-v1"


@app.function(image=rl_image, gpu="H100", cpu=8, memory=32768, timeout=5400, startup_timeout=360,
              scaledown_window=2, max_containers=1, volumes={"/cache": cache})
def grpo(sft_reward: float, sft_value60: float):
    from random import Random
    from statistics import mean
    from subprocess import PIPE, Popen
    from tempfile import TemporaryDirectory
    from time import sleep

    from vllm.lora.request import LoRARequest

    from time_gym.qwen_eval import evaluate
    from time_gym import qwen_grpo

    start = monotonic()
    try:
        sample = inference(True, memory_fraction=0.40)
        tasks = qwen_grpo.training_tasks()
        Random(41).shuffle(tasks)
        reward, value60 = sft_reward, sft_value60
        with TemporaryDirectory() as directory:
            work = Path(directory)
            adapter = str(work / "adapter")
            worker = Popen(["/opt/train/bin/python", "-u", "-m", "time_gym.qwen_grpo", SFT_ADAPTER, adapter],
                           stdin=PIPE, text=True, cwd=Path(qwen_grpo.__file__).parent.parent)
            try:
                for step, (case, budget) in enumerate(tasks[:30], 1):
                    lora = LoRARequest(f"grpo-{step - 1}", step, adapter if step > 1 else SFT_ADAPTER)
                    group = qwen_grpo.rollout_group(case, budget, lambda chats, seeds: sample(chats, seeds, lora), step * 8)
                    data = work / "group.json"
                    done = data.with_suffix(".done")
                    done.unlink(missing_ok=True)
                    data.write_text(json.dumps(group))
                    worker.stdin.write(str(data) + "\n")
                    worker.stdin.flush()
                    while not done.exists():
                        if worker.poll() is not None:
                            raise RuntimeError(f"GRPO trainer exited with code {worker.returncode}.")
                        sleep(0.25)
                    print(f"GRPO step {step}: terminal reward {mean(m['reward'] for m, _ in group):.4f}", flush=True)
                    if step not in (15, 30):
                        continue
                    lora = LoRARequest(f"check-{step}", step + 100, adapter)
                    rows = evaluate(lambda chats, seeds: [r.outputs[0].text for r in sample(chats, seeds, lora)],
                                    policy="grpo")
                    budget_rows = [r for r in rows if r["budget"] is not None]
                    new_reward = mean(r["reward"] for r in budget_rows)
                    new_value = mean(r["value"] for r in rows if r["budget"] == 60)
                    print(f"GRPO check {step}: reward={new_reward:.4f}, value60={new_value:.4f}, "
                          f"reread={sum(r['reread_minutes'] for r in rows)}, "
                          f"unused={sum(r['unused_useful_minutes'] for r in rows)}, "
                          f"unsupported={sum(r['unsupported'] for r in rows)}, "
                          f"eta_waste={sum(r['eta_waste_minutes'] or 0 for r in rows)}", flush=True)
                    if new_reward > reward and new_value <= value60:
                        return {"step": step, "stopped": True, "rows": rows}
                    reward, value60 = new_reward, new_value
                from shutil import copytree
                copytree(adapter, RL_ADAPTER)
                cache.commit()
                return {"step": step, "stopped": False, "rows": rows}
            finally:
                worker.terminate()
                worker.wait()
    finally:
        print(f"GPU_JOB grpo H100 {monotonic() - start:.1f}s", flush=True)


@app.local_entrypoint()
def train_rl(sft_results: str, output: str):
    from statistics import mean

    rows = [json.loads(line) for line in Path(sft_results).read_text().splitlines()]
    if len(rows) != 96 or any(r["split"] != "dev" or r["policy"] != "sft" for r in rows):
        raise ValueError("Use the 96 development SFT results.")
    result = grpo.remote(mean(r["reward"] for r in rows if r["budget"] is not None),
                         mean(r["value"] for r in rows if r["budget"] == 60))
    Path(output).write_text("".join(json.dumps(row) + "\n" for row in result["rows"]))
    print(f"GRPO step {result['step']}; rejected by reward check: {result['stopped']}")
