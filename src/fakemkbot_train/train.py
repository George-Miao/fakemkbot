from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


class EncodedDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, examples: list[dict[str, torch.Tensor]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training")
    return torch.device("cuda")


def quantization_config() -> BitsAndBytesConfig:
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


def read_messages(path: Path) -> list[list[dict[str, str]]]:
    records: list[list[dict[str, str]]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value: Any = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path} line {line_number} is not valid JSON") from error

            messages = value.get("messages") if isinstance(value, dict) else None
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"{path} line {line_number} has no message list")
            if any(
                not isinstance(message, dict)
                or not isinstance(message.get("role"), str)
                or not isinstance(message.get("content"), str)
                for message in messages
            ):
                raise ValueError(f"{path} line {line_number} has an invalid message")
            if messages[-1]["role"] != "assistant":
                raise ValueError(f"{path} line {line_number} must end with an assistant message")
            records.append(messages)

    if not records:
        raise ValueError(f"{path} contains no training records")
    return records


def encode_messages(
    messages: list[dict[str, str]],
    tokenizer: Any,
    *,
    max_length: int,
) -> dict[str, torch.Tensor] | None:
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    full = tokenizer(
        full_text,
        add_special_tokens=False,
        max_length=max_length,
        truncation=True,
    )
    prompt = tokenizer(
        prompt_text,
        add_special_tokens=False,
        max_length=max_length,
        truncation=True,
    )

    input_ids = list(full["input_ids"])
    attention_mask = list(full["attention_mask"])
    prompt_length = min(len(prompt["input_ids"]), len(input_ids))
    labels = [-100] * prompt_length + input_ids[prompt_length:]
    if not labels or all(label == -100 for label in labels):
        return None

    padding = max_length - len(input_ids)
    input_ids.extend([tokenizer.pad_token_id] * padding)
    attention_mask.extend([0] * padding)
    labels.extend([-100] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def load_dataset(path: Path, tokenizer: Any, *, max_length: int) -> EncodedDataset:
    examples = [
        encoded
        for messages in read_messages(path)
        if (encoded := encode_messages(messages, tokenizer, max_length=max_length)) is not None
    ]
    if not examples:
        raise ValueError(f"{path} has no assistant text within max length {max_length}")
    return EncodedDataset(examples)


def train(
    *,
    model_name: str,
    train_file: Path,
    validation_file: Path,
    output_dir: Path,
    epochs: float,
    learning_rate: float,
    max_length: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    seed: int,
) -> None:
    require_cuda()
    random.seed(seed)
    set_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = load_dataset(train_file, tokenizer, max_length=max_length)
    validation_dataset = load_dataset(validation_file, tokenizer, max_length=max_length)

    use_bf16 = torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config(),
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
        ),
    )
    model.print_trainable_parameters()

    output_dir.mkdir(parents=True, exist_ok=True)
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=use_bf16,
        fp16=not use_bf16,
        tf32=True,
        optim="adamw_torch_fused",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        remove_unused_columns=False,
        seed=seed,
        data_seed=seed,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=default_data_collator,
        processing_class=tokenizer,
    )
    print(
        f"Training {len(train_dataset)} records on {torch.cuda.get_device_name(0)}; "
        f"validation records: {len(validation_dataset)}"
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    (output_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "base_model": model_name,
                "train_records": len(train_dataset),
                "validation_records": len(validation_dataset),
                "epochs": epochs,
                "max_length": max_length,
                "quantization": "nf4",
                "seed": seed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a Qwen style adapter on CUDA")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train-file", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=Path("data/processed/validation.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/mk-style"))
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(
        model_name=args.model,
        train_file=args.train_file,
        validation_file=args.validation_file,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
