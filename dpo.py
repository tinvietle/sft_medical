#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOConfig, DPOTrainer

from sft import (
    SYSTEM_PROMPT,
    get_torch_dtype,
    load_env,
    login_services,
    print_gpu_memory,
    resolve_model_source,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a DPO adapter starting from an existing SFT adapter and a JSON preference dataset.",
    )
    parser.add_argument("--dataset-path", default="smol_merged.json", help="JSON file containing case_text, answer, and smol_answer.")
    parser.add_argument("--model-id", required=True, help="Base model identifier or local path.")
    parser.add_argument("--model-cache-dir", default=None, help="Optional local directory where the base model snapshot will be downloaded and reused.")
    parser.add_argument("--adapter-id", required=True, help="Existing SFT adapter repo ID or local path used to initialize the policy and reference adapters.")
    parser.add_argument("--output-dir", required=True, help="Directory to save the trained DPO adapter.")
    parser.add_argument("--hub-model-id", default=None, help="Hub repo ID for the trained DPO adapter.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit loading and train with standard precision.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter.")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum sequence length for DPO training.")
    parser.add_argument("--rows-per-group", type=int, default=5, help="Dataset grouping size for eval split. The first row in each group is used for eval.")
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=40)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--wandb-project", required=True, help="Weights & Biases project name.")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--no-push-to-hub", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.rows_per_group < 2:
        raise ValueError("--rows-per-group must be at least 2.")
    if not args.no_push_to_hub and not args.hub_model_id:
        raise ValueError("--hub-model-id is required unless --no-push-to-hub is set.")


def load_preference_dataset(dataset_path: str) -> Dataset:
    payload = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected {dataset_path} to contain a JSON list.")

    records: list[dict[str, str]] = []
    for index, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Expected item {index} in {dataset_path} to be a JSON object.")

        case_text = row.get("case_text")
        chosen = row.get("answer")
        rejected = row.get("smol_answer")

        if not isinstance(case_text, str) or not case_text.strip():
            raise ValueError(f"Expected item {index} in {dataset_path} to contain a non-empty string `case_text`.")
        if not isinstance(chosen, str) or not chosen.strip():
            raise ValueError(f"Expected item {index} in {dataset_path} to contain a non-empty string `answer`.")
        if not isinstance(rejected, str) or not rejected.strip():
            raise ValueError(f"Expected item {index} in {dataset_path} to contain a non-empty string `smol_answer`.")

        records.append(
            {
                "case_text": case_text.strip(),
                "answer": chosen.strip(),
                "smol_answer": rejected.strip(),
            }
        )

    if len(records) < 2:
        raise ValueError("Preference dataset must contain at least 2 rows.")
    return Dataset.from_list(records)


def create_dpo_messages(row: dict[str, str]) -> dict[str, list[dict[str, str]]]:
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["case_text"]},
        ],
        "chosen": [
            {"role": "assistant", "content": row["answer"]},
        ],
        "rejected": [
            {"role": "assistant", "content": row["smol_answer"]},
        ],
    }


def build_datasets(dataset_path: str, rows_per_group: int, seed: int) -> tuple[Dataset, Dataset]:
    dataset = load_preference_dataset(dataset_path)
    dpo_dataset = dataset.map(create_dpo_messages, remove_columns=dataset.column_names)

    train_indices: list[int] = []
    eval_indices: list[int] = []
    for start in range(0, len(dpo_dataset), rows_per_group):
        chunk = list(range(start, min(start + rows_per_group, len(dpo_dataset))))
        if len(chunk) < 2:
            raise ValueError(
                f"Chunk starting at row {start} has only {len(chunk)} item(s). "
                "Increase dataset size or adjust --rows-per-group."
            )
        eval_indices.append(chunk[0])
        train_indices.extend(chunk[1:])

    train_dataset = dpo_dataset.select(train_indices).shuffle(seed=seed)
    eval_dataset = dpo_dataset.select(eval_indices)
    return train_dataset, eval_dataset


def build_quantization_config(args: argparse.Namespace):
    from sft import load_model_and_tokenizer

    # Reuse the SFT model-loading logic to keep quantization behavior aligned.
    model_source = resolve_model_source(args)
    model, tokenizer = load_model_and_tokenizer(args, model_source=model_source)
    return model, tokenizer


def load_policy_model(args: argparse.Namespace):
    base_model, tokenizer = build_quantization_config(args)
    base_model.name_or_path = args.model_id
    base_model.config.name_or_path = args.model_id

    model = PeftModel.from_pretrained(
        base_model,
        args.adapter_id,
        adapter_name="policy",
        is_trainable=True,
    )
    model.load_adapter(
        args.adapter_id,
        adapter_name="ref",
        is_trainable=False,
    )
    model.set_adapter("policy")
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    all_params = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Trainable params: {trainable_params:,} || "
        f"All params: {all_params:,} || "
        f"Trainable%: {100 * trainable_params / all_params:.2f}%"
    )
    return model, tokenizer


def build_training_args(args: argparse.Namespace, report_to: str) -> DPOConfig:
    return DPOConfig(
        output_dir=args.output_dir,
        beta=args.beta,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        report_to=report_to,
        bf16=args.dtype == "bfloat16",
        fp16=args.dtype == "float16",
        model_adapter_name="policy",
        ref_adapter_name="ref",
        max_length=args.max_length,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        push_to_hub=not args.no_push_to_hub,
        hub_model_id=args.hub_model_id,
        seed=args.seed,
    )


def save_and_push_outputs(
    trainer: DPOTrainer,
    model: PeftModel,
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
) -> None:
    trainer.save_model(args.output_dir)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.no_push_to_hub:
        return

    trainer.push_to_hub()
    tokenizer.push_to_hub(args.hub_model_id)


def main() -> None:
    load_env()
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    report_to = login_services(
        push_to_hub=not args.no_push_to_hub,
        use_wandb=not args.disable_wandb,
        wandb_project=args.wandb_project,
    )

    train_dataset, eval_dataset = build_datasets(
        dataset_path=args.dataset_path,
        rows_per_group=args.rows_per_group,
        seed=args.seed,
    )
    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")

    model, tokenizer = load_policy_model(args)
    training_args = build_training_args(args, report_to=report_to)

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    baseline_reserved_gb = print_gpu_memory("Before training")
    trainer_stats = trainer.train()
    print(f"Training runtime: {trainer_stats.metrics['train_runtime']:.2f} seconds")
    print_gpu_memory("After training", baseline_reserved_gb=baseline_reserved_gb)

    save_and_push_outputs(trainer=trainer, model=model, tokenizer=tokenizer, args=args)


if __name__ == "__main__":
    main()
