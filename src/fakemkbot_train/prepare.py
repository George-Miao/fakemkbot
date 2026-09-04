from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, TypedDict

SYSTEM_PROMPT = "你只模仿 MK 的聊天口吻。保持原始、具体、口语化, 不解释, 不润色。"
USER_PROMPT = "写一句 MK 可能说的全新消息, 只输出正文。"


class ChatMessage(TypedDict):
    role: str
    content: str


class TrainingRecord(TypedDict):
    messages: list[ChatMessage]


def parse_raw_line(value: Any, line_number: int) -> tuple[str, bool]:
    if not isinstance(value, dict):
        raise ValueError(f"Line {line_number} must contain a JSON object")

    text = value.get("text")
    is_forwarded = value.get("is_forwarded")
    if not isinstance(text, str):
        raise ValueError(f"Line {line_number} has no string text field")
    if not isinstance(is_forwarded, bool):
        raise ValueError(f"Line {line_number} has no boolean is_forwarded field")

    return text.replace("\r\n", "\n").replace("\r", "\n").strip(), is_forwarded


def load_records(path: Path, *, forwarded_only: bool = True) -> list[TrainingRecord]:
    records: list[TrainingRecord] = []
    seen: set[str] = set()

    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Line {line_number} is not valid JSON") from error

            text, is_forwarded = parse_raw_line(value, line_number)
            if not text or (forwarded_only and not is_forwarded) or text in seen:
                continue
            seen.add(text)
            records.append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_PROMPT},
                        {"role": "assistant", "content": text},
                    ]
                }
            )

    return records


def split_records(
    records: Sequence[TrainingRecord],
    *,
    validation_ratio: float,
    seed: int,
) -> tuple[list[TrainingRecord], list[TrainingRecord]]:
    if len(records) < 2:
        raise ValueError("At least two unique messages are required")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("Validation ratio must be between zero and one")

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    validation_count = min(len(shuffled) - 1, max(1, round(len(shuffled) * validation_ratio)))
    validation = shuffled[:validation_count]
    train = shuffled[validation_count:]
    return train, validation


def write_jsonl(path: Path, records: Iterable[TrainingRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            encoding="utf-8",
            delete=False,
        ) as destination:
            temporary_path = Path(destination.name)
            for record in records:
                destination.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def prepare_dataset(
    input_path: Path,
    output_dir: Path,
    *,
    validation_ratio: float = 0.05,
    seed: int = 42,
    forwarded_only: bool = True,
) -> tuple[int, int]:
    records = load_records(input_path, forwarded_only=forwarded_only)
    train, validation = split_records(
        records,
        validation_ratio=validation_ratio,
        seed=seed,
    )
    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "validation.jsonl", validation)
    return len(train), len(validation)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Telegram text for style fine-tuning")
    parser.add_argument("--input", type=Path, default=Path("data/raw/messages.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-non-forwarded",
        action="store_true",
        help="Include messages that are not forwarded",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_count, validation_count = prepare_dataset(
        args.input,
        args.output_dir,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        forwarded_only=not args.include_non_forwarded,
    )
    print(f"Prepared {train_count} train and {validation_count} validation records")


if __name__ == "__main__":
    main()
