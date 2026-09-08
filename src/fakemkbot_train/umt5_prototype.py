from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model
from peft.utils.other import prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

DEFAULT_MODEL = "google/umt5-small"
SENTINEL_TOKENS = ("<extra_id_0>", "<extra_id_1>", "<extra_id_2>")
LORA_TARGET_MODULES = ["q", "k", "v", "o", "wi_0", "wi_1", "wo"]


@dataclass(frozen=True)
class AnchorExample:
    source: str
    anchor: str
    left: str
    right: str


@dataclass(frozen=True)
class SampleResult:
    source: str
    anchor: str
    raw_output: str
    generated: str | None
    left: str | None
    right: str | None
    parsed: bool
    contains_anchor: bool
    duplicated_anchor: bool
    exact_reconstruction: bool
    source_similarity: float


class EncodedDataset(Dataset[dict[str, list[int]]]):
    def __init__(self, examples: list[dict[str, list[int]]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the UMT5 prototype")
    return torch.device("cuda")


def quantization_config() -> BitsAndBytesConfig:
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


def read_texts(path: Path, *, limit: int | None) -> list[str]:
    texts: list[str] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value: Any = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path} line {line_number} is not valid JSON") from error

            messages = value.get("messages") if isinstance(value, dict) else None
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{path} line {line_number} has no message list")
            message = messages[-1]
            text = message.get("content") if isinstance(message, dict) else None
            if not isinstance(text, str) or not text:
                raise ValueError(f"{path} line {line_number} has no assistant text")
            if any(token in text for token in SENTINEL_TOKENS):
                continue
            texts.append(text)
            if limit is not None and len(texts) >= limit:
                break

    if not texts:
        raise ValueError(f"{path} contains no usable messages")
    return texts


def anchor_ranges(length: int, count: int, offset: int) -> list[tuple[int, int]]:
    width = max(1, min(4, round(length / 3)))
    candidates = [
        (0, min(length, width)),
        (max(0, (length - width) // 2), min(length, (length - width) // 2 + width)),
        (max(0, length - width), length),
    ]
    unique = list(dict.fromkeys((start, end) for start, end in candidates if start < end))
    if unique:
        rotation = offset % len(unique)
        unique = unique[rotation:] + unique[:rotation]
    return unique[:count]


def make_anchor_examples(texts: list[str], *, anchors_per_message: int) -> list[AnchorExample]:
    examples: list[AnchorExample] = []
    for index, text in enumerate(texts):
        for start, end in anchor_ranges(len(text), anchors_per_message, index):
            examples.append(
                AnchorExample(
                    source=text,
                    anchor=text[start:end],
                    left=text[:start],
                    right=text[end:],
                )
            )
    return examples


def conditioning_text(anchor: str) -> str:
    escaped = anchor.replace("<extra_id_", "<extra\u200b_id_")
    return f"{SENTINEL_TOKENS[0]}{escaped}{SENTINEL_TOKENS[1]}"


def target_text(example: AnchorExample) -> str:
    return (
        f"{SENTINEL_TOKENS[0]}{example.left}{SENTINEL_TOKENS[1]}{example.right}{SENTINEL_TOKENS[2]}"
    )


def encode_examples(
    examples: list[AnchorExample],
    tokenizer: Any,
    *,
    max_length: int,
) -> EncodedDataset:
    encoded: list[dict[str, list[int]]] = []
    for example in examples:
        inputs = tokenizer(conditioning_text(example.anchor), add_special_tokens=True)
        labels = tokenizer(text_target=target_text(example), add_special_tokens=True)
        if len(inputs["input_ids"]) > max_length or len(labels["input_ids"]) > max_length:
            continue
        encoded.append(
            {
                "input_ids": list(inputs["input_ids"]),
                "attention_mask": list(inputs["attention_mask"]),
                "labels": list(labels["input_ids"]),
            }
        )
    if not encoded:
        raise ValueError("No anchor examples fit within the configured maximum length")
    return EncodedDataset(encoded)


def load_quantized_base(model_name: str) -> Any:
    return AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        quantization_config=quantization_config(),
        device_map={"": 0},
    )


def train_adapter(
    *,
    model_name: str,
    tokenizer: Any,
    train_examples: list[AnchorExample],
    validation_examples: list[AnchorExample],
    output_dir: Path,
    epochs: float,
    learning_rate: float,
    max_length: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    seed: int,
    resume_from_checkpoint: Path | None,
) -> tuple[Any, dict[str, float]]:
    train_dataset = encode_examples(train_examples, tokenizer, max_length=max_length)
    validation_dataset = encode_examples(validation_examples, tokenizer, max_length=max_length)
    model = load_quantized_base(model_name)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
        ),
    )
    model.print_trainable_parameters()

    output_dir.mkdir(parents=True, exist_ok=True)
    use_bf16 = torch.cuda.is_bf16_supported()
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
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        ),
        processing_class=tokenizer,
    )
    print(
        f"Training {len(train_dataset)} anchor records on {torch.cuda.get_device_name(0)}; "
        f"validation records: {len(validation_dataset)}"
    )
    train_result = trainer.train(
        resume_from_checkpoint=(
            str(resume_from_checkpoint) if resume_from_checkpoint is not None else None
        )
    )
    evaluation = trainer.evaluate()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    train_loss_name = "resume_train_loss" if resume_from_checkpoint is not None else "train_loss"
    metrics = {
        train_loss_name: float(train_result.metrics["train_loss"]),
        "eval_loss": float(evaluation["eval_loss"]),
    }
    return model, metrics


def load_adapter(output_dir: Path) -> tuple[Any, Any]:
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Prototype adapter does not exist: {output_dir}")
    config = PeftConfig.from_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(output_dir, use_fast=True)
    model = PeftModel.from_pretrained(
        load_quantized_base(config.base_model_name_or_path), output_dir
    )
    return model, tokenizer


def sentinel_ids(tokenizer: Any) -> tuple[int, int, int]:
    ids = tuple(tokenizer.convert_tokens_to_ids(token) for token in SENTINEL_TOKENS)
    if any(token_id == tokenizer.unk_token_id for token_id in ids):
        raise RuntimeError("The tokenizer does not provide the required sentinel tokens")
    return ids


def parse_spans(token_ids: list[int], tokenizer: Any) -> tuple[str, str] | None:
    first, second, third = sentinel_ids(tokenizer)
    pad = tokenizer.pad_token_id
    eos = tokenizer.eos_token_id
    core = list(token_ids)
    while core and core[0] == pad:
        core.pop(0)
    while core and core[-1] in {pad, eos}:
        core.pop()
    if not core or core[0] != first or core[-1] != third:
        return None
    if core.count(first) != 1 or core.count(second) != 1 or core.count(third) != 1:
        return None
    second_index = core.index(second)
    third_index = core.index(third)
    if not 0 < second_index < third_index:
        return None
    left_ids = core[1:second_index]
    right_ids = core[second_index + 1 : third_index]
    left = tokenizer.decode(
        left_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    right = tokenizer.decode(
        right_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    return left, right


def generate_sample(
    model: Any,
    tokenizer: Any,
    example: AnchorExample,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> SampleResult:
    inputs = tokenizer(conditioning_text(example.anchor), return_tensors="pt").to("cuda")
    first, _, third = sentinel_ids(tokenizer)
    decoder_input_ids = torch.tensor(
        [[tokenizer.pad_token_id, first]],
        dtype=torch.long,
        device="cuda",
    )
    end_of_document_id = tokenizer.convert_tokens_to_ids("[eod]")
    bad_words_ids = None if end_of_document_id == tokenizer.unk_token_id else [[end_of_document_id]]
    set_seed(seed)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            decoder_input_ids=decoder_input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[tokenizer.eos_token_id, third],
            bad_words_ids=bad_words_ids,
        )
    spans = parse_spans(output[0].tolist(), tokenizer)
    raw_output = tokenizer.decode(
        output[0],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if spans is None:
        return SampleResult(
            source=example.source,
            anchor=example.anchor,
            raw_output=raw_output,
            generated=None,
            left=None,
            right=None,
            parsed=False,
            contains_anchor=False,
            duplicated_anchor=False,
            exact_reconstruction=False,
            source_similarity=0.0,
        )

    left, right = spans
    generated = f"{left}{example.anchor}{right}"
    return SampleResult(
        source=example.source,
        anchor=example.anchor,
        raw_output=raw_output,
        generated=generated,
        left=left,
        right=right,
        parsed=True,
        contains_anchor=example.anchor in generated,
        duplicated_anchor=example.anchor in left or example.anchor in right,
        exact_reconstruction=generated == example.source,
        source_similarity=SequenceMatcher(None, generated, example.source).ratio(),
    )


def evaluate_samples(
    model: Any,
    tokenizer: Any,
    examples: list[AnchorExample],
    *,
    sample_count: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> tuple[list[SampleResult], dict[str, float | int]]:
    model.eval()
    model.config.use_cache = True
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    selected = list(examples)
    random.Random(seed).shuffle(selected)
    selected = selected[:sample_count]
    results = [
        generate_sample(
            model,
            tokenizer,
            example,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed + index,
        )
        for index, example in enumerate(selected)
    ]
    parsed = [result for result in results if result.parsed]
    metrics: dict[str, float | int] = {
        "samples": len(results),
        "parsed": len(parsed),
        "parse_rate": len(parsed) / len(results) if results else 0.0,
        "inclusion_rate": (
            sum(result.contains_anchor for result in results) / len(results) if results else 0.0
        ),
        "duplicate_anchor_rate": (
            sum(result.duplicated_anchor for result in parsed) / len(parsed) if parsed else 0.0
        ),
        "exact_reconstruction_rate": (
            sum(result.exact_reconstruction for result in parsed) / len(parsed) if parsed else 0.0
        ),
        "mean_source_similarity": (
            sum(result.source_similarity for result in parsed) / len(parsed) if parsed else 0.0
        ),
    }
    return results, metrics


def write_results(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    training_metrics: dict[str, float],
    sample_metrics: dict[str, float | int],
    samples: list[SampleResult],
) -> None:
    summary_path = output_dir / "prototype_summary.json"
    previous: dict[str, Any] = {}
    if args.skip_train and summary_path.is_file():
        value = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            previous = value
    effective_training = training_metrics or previous.get("training", {})
    effective_epochs = previous.get("epochs", args.epochs) if args.skip_train else args.epochs
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**2
    summary = {
        "prototype": True,
        "base_model": args.model,
        "train_file": str(args.train_file),
        "validation_file": str(args.validation_file),
        "anchors_per_message": args.anchors_per_message,
        "epochs": effective_epochs,
        "max_length": args.max_length,
        "quantization": "nf4",
        "lora_rank": 16,
        "lora_targets": LORA_TARGET_MODULES,
        "seed": args.seed,
        "training": effective_training,
        "samples": sample_metrics,
        "custom_prompts": len(args.prompt),
        "peak_allocated_mib": peak_allocated,
        "peak_reserved_mib": peak_reserved,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "prototype_samples.jsonl").open("w", encoding="utf-8") as destination:
        for sample in samples:
            destination.write(json.dumps(asdict(sample), ensure_ascii=False, separators=(",", ":")))
            destination.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PROTOTYPE: train and evaluate UMT5 dual-sentinel generation on CUDA"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train-file", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=Path("data/processed/validation.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/umt5-prototype"),
    )
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--anchors-per-message", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--limit-train-messages", type=int)
    parser.add_argument("--limit-validation-messages", type=int)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--resume-from-checkpoint", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    require_cuda()
    random.seed(args.seed)
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()

    train_texts = read_texts(args.train_file, limit=args.limit_train_messages)
    validation_texts = read_texts(
        args.validation_file,
        limit=args.limit_validation_messages,
    )
    train_examples = make_anchor_examples(
        train_texts,
        anchors_per_message=args.anchors_per_message,
    )
    validation_examples = make_anchor_examples(
        validation_texts,
        anchors_per_message=args.anchors_per_message,
    )

    if args.skip_train:
        model, tokenizer = load_adapter(args.output_dir)
        training_metrics: dict[str, float] = {}
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        model, training_metrics = train_adapter(
            model_name=args.model,
            tokenizer=tokenizer,
            train_examples=train_examples,
            validation_examples=validation_examples,
            output_dir=args.output_dir,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            seed=args.seed,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )

    samples, sample_metrics = evaluate_samples(
        model,
        tokenizer,
        validation_examples,
        sample_count=args.sample_count,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )
    for index, prompt in enumerate(args.prompt):
        custom = generate_sample(
            model,
            tokenizer,
            AnchorExample(source=prompt, anchor=prompt, left="", right=""),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed + len(samples) + index,
        )
        samples.append(custom)
        print(
            json.dumps(
                {"prompt": prompt, "parsed": custom.parsed, "generated": custom.generated},
                ensure_ascii=False,
            )
        )
    write_results(
        args.output_dir,
        args=args,
        training_metrics=training_metrics,
        sample_metrics=sample_metrics,
        samples=samples,
    )


if __name__ == "__main__":
    main()
