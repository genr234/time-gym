"""Train one bounded Qwen LoRA on Modal. No evaluation or RL starts here."""

from time import monotonic

import modal

app = modal.App("time-gym-qwen-sft")
cache = modal.Volume.from_name("time-gym-qwen-weights", create_if_missing=True)
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .uv_pip_install("torch==2.10.0", "transformers==5.3.0", "trl==0.29.0", "peft==0.18.1", "gymnasium>=1.0,<2")
    .env({"HF_HOME": "/cache", "HF_HUB_DISABLE_XET": "1"})
    .add_local_python_source("time_gym")
)
GPU = "L40S"
ADAPTER = "/cache/time-gym-sft-v1"


@app.function(image=image, gpu=GPU, cpu=4, memory=24576, timeout=2400, startup_timeout=360,
              scaledown_window=2, max_containers=1, volumes={"/cache": cache})
def train():
    from datasets import Dataset
    from peft import LoraConfig
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from trl import SFTConfig, SFTTrainer

    from time_gym.qwen_eval import MODEL, REVISION
    from time_gym.qwen_sft import tokenized_examples

    start = monotonic()
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        data = Dataset.from_list(list(tokenized_examples(tokenizer)))
        longest = max(map(len, data["input_ids"]))
        if longest > 8192:
            raise ValueError(f"SFT context exceeds evaluation limit: {longest}")
        print(f"SFT examples: {len(data)}; longest: {longest} tokens", flush=True)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16)
        model.config.use_cache = False
        config = SFTConfig(output_dir="/tmp/time-gym-sft", max_steps=240, per_device_train_batch_size=1,
                           learning_rate=1e-4, bf16=True, gradient_checkpointing=True,
                           gradient_checkpointing_kwargs={"use_reentrant": False},
                           dataset_kwargs={"skip_prepare_dataset": True}, completion_only_loss=True,
                           logging_steps=20, save_strategy="no", report_to="none", seed=41)
        trainer = SFTTrainer(model=model, args=config, train_dataset=data, processing_class=tokenizer,
                             peft_config=LoraConfig(r=8, lora_alpha=16, task_type="CAUSAL_LM",
                                                    target_modules=["q_proj", "v_proj"]))
        result = trainer.train()
        trainer.save_model(ADAPTER)
        cache.commit()
        print(f"Saved LoRA to {ADAPTER}: {result.metrics}", flush=True)
    finally:
        print(f"GPU_JOB sft {GPU} {monotonic() - start:.1f}s", flush=True)
