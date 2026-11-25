"""Bounded on-demand Modal jobs for Qwen and Time Gym."""

import json
from pathlib import Path
from time import monotonic

import modal

from modal_sft import CAUSAL_CONV, FLA

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


def inference(adapter_enabled, memory_fraction=0.86, context=8192, *, thinking=False):
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from time_gym.qwen_eval import MODEL, REVISION, THINKING_TEMPERATURE, action_schema

    engine = LLM(model=MODEL, revision=REVISION, language_model_only=True, dtype="bfloat16",
                 max_model_len=context, gpu_memory_utilization=memory_fraction, enforce_eager=True,
                 enable_lora=adapter_enabled)
    tokenizer = engine.get_tokenizer()
    schemas = {eta: action_schema(eta=eta) for eta in (True, False)}

    def sample(chats, seeds, lora):
        prompts = tokenizer.apply_chat_template(chats, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=thinking)
        params = [SamplingParams(temperature=THINKING_TEMPERATURE if thinking else 0.7, top_p=1.0 if thinking else 0.8,
                                 top_k=-1 if thinking else 20, max_tokens=1024 if thinking else 384, seed=seed,
                                 skip_special_tokens=not thinking,
                                 structured_outputs=None if thinking else StructuredOutputsParams(json=schemas[chat[1]["content"].startswith("ETA task:")]))
                  for chat, seed in zip(chats, seeds, strict=True)]
        return engine.generate(prompts, params, use_tqdm=False, lora_request=lora)

    return sample


@app.function(image=eval_image, gpu=GPU, cpu=4, memory=16384, timeout=EVAL_TIMEOUT, startup_timeout=360,
              scaledown_window=2, max_containers=1, volumes={"/cache": cache})
def baseline(adapter: str = "", split: str = "dev", policy: str = "base", thinking: bool = False):
    from vllm.lora.request import LoRARequest
    from time_gym.qwen_eval import evaluate

    start = monotonic()
    try:
        sample = inference(bool(adapter), context=65536 if thinking else (32768 if split == "held-out" else 8192),
                           thinking=thinking)
        lora = LoRARequest(policy, 1, adapter) if adapter else None
        return evaluate(lambda chats, seeds: [r.outputs[0].text for r in sample(chats, seeds, lora)],
                        split=split, policy=policy, thinking=thinking)
    finally:
        print(f"GPU_JOB baseline {GPU} {monotonic() - start:.1f}s", flush=True)


@app.local_entrypoint()
def main(output: str, adapter: str = "", split: str = "dev", policy: str = "", progress_study: bool = False,
         thinking: bool = False):
    if split not in ("dev", "held-out"):
        raise ValueError("Use dev or held-out.")
    if progress_study:
        if adapter or split != "dev" or policy or thinking:
            raise ValueError("The progress study uses the fixed SFT adapter and development cases.")
        rows = compare_progress.remote()
    else:
        rows = baseline.remote(adapter, split, policy or ("sft" if adapter else "base"), thinking)
    Path(output).write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"Wrote {len(rows)} model episodes to {output}")


rl_image = vllm_image.run_commands(
    "python -m pip install uv",
    "uv venv /opt/train",
    "uv pip install --python /opt/train/bin/python torch==2.10.0 transformers==5.3.0 peft==0.18.1 'gymnasium>=1.0,<2'",
    f"uv pip install --python /opt/train/bin/python {FLA} '{CAUSAL_CONV}'",
    "/opt/train/bin/python -c 'from causal_conv1d import causal_conv1d_fn; from peft import PeftModel'",
).add_local_python_source("time_gym")
SFT_ADAPTER = "/cache/time-gym-sft-v1"
RL_ADAPTER = "/cache/time-gym-grpo-v1"
THINKING_SFT_ADAPTER = "/cache/time-gym-sft-v2"
THINKING_RL_ADAPTER = "/cache/time-gym-grpo-v2"


@app.function(image=eval_image, gpu=GPU, cpu=4, memory=16384, timeout=1350, startup_timeout=360,
              retries=0, scaledown_window=2, max_containers=1, volumes={"/cache": cache})
def compare_progress():
    from vllm.lora.request import LoRARequest
    from time_gym.qwen_eval import PROGRESS_EVAL_SEEDS, evaluate

    start = monotonic()
    try:
        sample = inference(True, context=32768)
        lora = LoRARequest("sft", 1, SFT_ADAPTER)

        def sample_text(chats, seeds):
            return [r.outputs[0].text for r in sample(chats, seeds, lora)]

        return evaluate(sample_text, policy="sft", seeds=PROGRESS_EVAL_SEEDS) + evaluate(
            sample_text, policy="sft-progress", seeds=PROGRESS_EVAL_SEEDS, progress=True)
    finally:
        print(f"GPU_JOB compare_progress {GPU} {monotonic() - start:.1f}s", flush=True)


@app.function(image=rl_image, gpu="H100", cpu=8, memory=32768, timeout=5400, startup_timeout=360,
              retries=0, scaledown_window=2, max_containers=1, volumes={"/cache": cache})
def grpo(sft_reward: float, sft_value60: float, thinking: bool = False):
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
        sample = inference(True, memory_fraction=0.40, context=65536 if thinking else 8192, thinking=thinking)
        tasks = qwen_grpo.training_tasks(thinking=thinking)
        Random(41).shuffle(tasks)
        sft_adapter = THINKING_SFT_ADAPTER if thinking else SFT_ADAPTER
        rl_adapter = THINKING_RL_ADAPTER if thinking else RL_ADAPTER
        limit = 100 if thinking else 30
        reward, value60 = sft_reward, sft_value60
        with TemporaryDirectory() as directory:
            work = Path(directory)
            adapter = str(work / "adapter")
            worker = Popen(["/opt/train/bin/python", "-u", "-m", "time_gym.qwen_grpo", sft_adapter, adapter,
                            "thinking" if thinking else "plain"],
                           stdin=PIPE, text=True, cwd=Path(qwen_grpo.__file__).parent.parent)
            try:
                for step, (case, budget) in enumerate(tasks[:limit], 1):
                    lora = LoRARequest(f"grpo-{step - 1}", step, adapter if step > 1 else sft_adapter)
                    group = qwen_grpo.rollout_group(case, budget, lambda chats, seeds: sample(chats, seeds, lora),
                                                    step * 8, thinking=thinking)
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
                    time_limit = thinking and monotonic() - start >= 3900  # Reserve time for the full dev check.
                    if step % 15 and step != limit and not time_limit:
                        continue
                    lora = LoRARequest(f"check-{step}", step + 1000, adapter)
                    rows = evaluate(lambda chats, seeds: [r.outputs[0].text for r in sample(chats, seeds, lora)],
                                    policy="grpo", thinking=thinking)
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
                    if time_limit:
                        break
                from shutil import copytree
                copytree(adapter, rl_adapter)
                cache.commit()
                return {"step": step, "stopped": False, "rows": rows}
            finally:
                worker.terminate()
                worker.wait()
    finally:
        print(f"GPU_JOB grpo H100 {monotonic() - start:.1f}s", flush=True)


@app.local_entrypoint()
def train_rl(sft_results: str, output: str, thinking: bool = False, base_results: str = ""):
    from statistics import mean

    def dev_results(path, policy):
        parsed = [json.loads(line) for line in Path(path).read_text().splitlines()]
        if len(parsed) != 96 or any(r["split"] != "dev" or r["policy"] != policy for r in parsed):
            raise ValueError(f"Use the 96 development {policy} results.")
        if thinking and sum(r["outcome"] == "unparseable_output" for r in parsed) > 0.2 * len(parsed):
            raise ValueError(f"Fix {policy} parse failures before GRPO.")
        return parsed

    rows = dev_results(sft_results, "sft")
    if thinking:
        if not base_results:
            raise ValueError("Thinking GRPO needs the matching base development results.")
        dev_results(base_results, "base")
    result = grpo.remote(mean(r["reward"] for r in rows if r["budget"] is not None),
                         mean(r["value"] for r in rows if r["budget"] == 60), thinking)
    Path(output).write_text("".join(json.dumps(row) + "\n" for row in result["rows"]))
    print(f"GRPO step {result['step']}; rejected by reward check: {result['stopped']}")
