"""Train one bounded Qwen LoRA on Modal. No evaluation or RL starts here."""

from time import monotonic

import modal

from time_gym.qwen_sft import CAUSAL_CONV, FLA

app = modal.App("time-gym-qwen-sft")
cache = modal.Volume.from_name("time-gym-qwen-weights", create_if_missing=True)
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .uv_pip_install("torch==2.10.0", "transformers==5.3.0", "trl==0.29.0", "peft==0.18.1", "gymnasium>=1.0,<2")
    .env({"HF_HOME": "/cache", "HF_HUB_DISABLE_XET": "1"})
)
GPU = "L40S"
ADAPTER = "/cache/time-gym-sft-v1"
THINKING_ADAPTER = "/cache/time-gym-sft-v2"
thinking_image = image.uv_pip_install(FLA, CAUSAL_CONV).add_local_python_source("time_gym")
image = image.add_local_python_source("time_gym")


@app.function(image=image, gpu=GPU, cpu=4, memory=24576, timeout=2400, startup_timeout=360,
              scaledown_window=2, max_containers=1, volumes={"/cache": cache})
def train():
    return _train(GPU, ADAPTER, thinking=False)


@app.function(image=thinking_image, cpu=2, timeout=120, startup_timeout=120, scaledown_window=2)
def check_thinking_image():
    from importlib.metadata import version
    from causal_conv1d import causal_conv1d_fn

    print(f"Imported {causal_conv1d_fn.__name__}; installed FLA {version('flash-linear-attention')}.")


@app.function(image=thinking_image, gpu="H100", cpu=4, memory=24576, timeout=6200, startup_timeout=360,
              retries=0, scaledown_window=2, max_containers=1, volumes={"/cache": cache})
def train_thinking():
    return _train("H100", THINKING_ADAPTER, thinking=True)


def _train(gpu, adapter, *, thinking):
    from datasets import Dataset
    from peft import LoraConfig
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from trl import SFTConfig, SFTTrainer

    from time_gym.qwen_eval import MODEL, REVISION
    from time_gym.qwen_sft import tokenized_examples

    start = monotonic()
    try:
        if thinking:
            from transformers.models.qwen3_5.modeling_qwen3_5 import is_fast_path_available

            if not is_fast_path_available:
                raise RuntimeError("Qwen is missing its fast training kernels on GPU.")
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        data = Dataset.from_list(list(tokenized_examples(tokenizer, thinking=thinking)))
        longest = max(map(len, data["input_ids"]))
        if longest > 8192:
            raise ValueError(f"SFT context exceeds evaluation limit: {longest}")
        print(f"SFT examples: {len(data)}; longest: {longest} tokens", flush=True)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16)
        model.config.use_cache = False
        config = SFTConfig(output_dir="/tmp/time-gym-sft", max_steps=-1 if thinking else 240,
                           num_train_epochs=2, per_device_train_batch_size=1,
                           gradient_accumulation_steps=8 if thinking else 1,
                           learning_rate=2e-4 if thinking else 1e-4,
                           lr_scheduler_type="cosine" if thinking else "linear", bf16=True, gradient_checkpointing=True,
                           gradient_checkpointing_kwargs={"use_reentrant": False},
                           dataset_kwargs={"skip_prepare_dataset": True}, completion_only_loss=True,
                           logging_steps=20, save_strategy="no", report_to="none", seed=41)
        trainer = SFTTrainer(model=model, args=config, train_dataset=data, processing_class=tokenizer,
                             peft_config=LoraConfig(r=16 if thinking else 8, lora_alpha=32 if thinking else 16,
                                                    task_type="CAUSAL_LM",
                                                    target_modules="all-linear" if thinking else ["q_proj", "v_proj"]))
        result = trainer.train()
        trainer.save_model(adapter)
        cache.commit()
        print(f"Saved LoRA to {adapter}: {result.metrics}", flush=True)
    finally:
        print(f"GPU_JOB sft {gpu} {monotonic() - start:.1f}s", flush=True)
