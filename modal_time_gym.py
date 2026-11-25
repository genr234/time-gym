"""Bounded on-demand Modal jobs for Qwen and Time Gym."""

import json
from pathlib import Path
from time import monotonic

import modal

app = modal.App("time-gym-qwen")
cache = modal.Volume.from_name("time-gym-qwen-weights", create_if_missing=True)
eval_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.19.0", "gymnasium>=1.0,<2")
    .env({"HF_HOME": "/cache", "HF_HUB_DISABLE_XET": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
    .add_local_python_source("time_gym")
)
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


@app.function(image=eval_image, gpu=GPU, cpu=4, memory=16384, timeout=EVAL_TIMEOUT, startup_timeout=360,
              scaledown_window=2, max_containers=1, volumes={"/cache": cache})
def baseline(adapter: str = ""):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import StructuredOutputsParams
    from time_gym.qwen_eval import MODEL, REVISION, action_schema, evaluate

    start = monotonic()
    try:
        engine = LLM(model=MODEL, revision=REVISION, language_model_only=True, dtype="bfloat16",
                     max_model_len=8192, gpu_memory_utilization=0.86, enforce_eager=True,
                     enable_lora=bool(adapter))
        tokenizer = engine.get_tokenizer()
        schemas = {eta: action_schema(eta=eta) for eta in (True, False)}
        lora = LoRARequest("sft", 1, adapter) if adapter else None

        def sample(chats, seeds):
            prompts = tokenizer.apply_chat_template(chats, tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=False)
            params = [SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=384, seed=seed,
                                     structured_outputs=StructuredOutputsParams(json=schemas[chat[1]["content"].startswith("ETA task:")]))
                      for chat, seed in zip(chats, seeds, strict=True)]
            return [result.outputs[0].text for result in engine.generate(prompts, params, use_tqdm=False, lora_request=lora)]

        return evaluate(sample, policy="sft" if adapter else "base")
    finally:
        print(f"GPU_JOB baseline {GPU} {monotonic() - start:.1f}s", flush=True)


@app.local_entrypoint()
def main(output: str, adapter: str = ""):
    rows = baseline.remote(adapter)
    Path(output).write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"Wrote {len(rows)} model episodes to {output}")
