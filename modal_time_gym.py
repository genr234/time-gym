"""Bounded on-demand Modal jobs for Qwen and Time Gym."""

import json
from pathlib import Path
from time import monotonic

import modal

app = modal.App("time-gym-qwen")
cache = modal.Volume.from_name("time-gym-qwen-weights", create_if_missing=True)
eval_image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.19.0", add_python=None)
    .entrypoint([])
    .pip_install("gymnasium>=1.0,<2")
    .env({"HF_HOME": "/cache", "HF_HUB_DISABLE_XET": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
    .add_local_python_source("time_gym")
)
GPU = "L40S"
EVAL_TIMEOUT = 1500


@app.function(image=eval_image, gpu=GPU, cpu=4, memory=16384, timeout=EVAL_TIMEOUT, startup_timeout=360,
              scaledown_window=2, max_containers=1, volumes={"/cache": cache})
def baseline():
    from vllm import LLM, SamplingParams
    from time_gym.qwen_eval import MODEL, REVISION, evaluate

    start = monotonic()
    try:
        engine = LLM(model=MODEL, revision=REVISION, language_model_only=True, dtype="bfloat16",
                     max_model_len=8192, gpu_memory_utilization=0.86, enforce_eager=True)
        tokenizer = engine.get_tokenizer()

        def sample(chats, seeds):
            prompts = tokenizer.apply_chat_template(chats, tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=False)
            params = [SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=384, seed=seed)
                      for seed in seeds]
            return [result.outputs[0].text for result in engine.generate(prompts, params, use_tqdm=False)]

        return evaluate(sample)
    finally:
        print(f"GPU_JOB baseline {GPU} {monotonic() - start:.1f}s", flush=True)


@app.local_entrypoint()
def main(output: str):
    rows = baseline.remote()
    Path(output).write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"Wrote {len(rows)} model episodes to {output}")
