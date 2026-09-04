from __future__ import annotations

import json
from pathlib import Path

import pytest

from fakemkbot_train.prepare import load_records, prepare_dataset, split_records


def write_raw(path: Path, values: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )


def assistant_text(record: dict[str, object]) -> str:
    messages = record["messages"]
    assert isinstance(messages, list)
    message = messages[-1]
    assert isinstance(message, dict)
    text = message["content"]
    assert isinstance(text, str)
    return text


def test_load_records_keeps_unique_forwarded_text(tmp_path: Path) -> None:
    source = tmp_path / "messages.jsonl"
    write_raw(
        source,
        [
            {"id": 1, "text": "  first\r\nline  ", "is_forwarded": True},
            {"id": 2, "text": "first\nline", "is_forwarded": True},
            {"id": 3, "text": "not target", "is_forwarded": False},
            {"id": 4, "text": "  ", "is_forwarded": True},
        ],
    )

    records = load_records(source)

    assert [assistant_text(record) for record in records] == ["first\nline"]


def test_split_is_deterministic_and_disjoint(tmp_path: Path) -> None:
    source = tmp_path / "messages.jsonl"
    write_raw(
        source,
        [{"id": index, "text": f"message {index}", "is_forwarded": True} for index in range(20)],
    )
    records = load_records(source)

    first_train, first_validation = split_records(records, validation_ratio=0.2, seed=7)
    second_train, second_validation = split_records(records, validation_ratio=0.2, seed=7)

    assert first_train == second_train
    assert first_validation == second_validation
    assert len(first_train) == 16
    assert len(first_validation) == 4
    assert {assistant_text(record) for record in first_train}.isdisjoint(
        {assistant_text(record) for record in first_validation}
    )


def test_prepare_writes_valid_chat_records(tmp_path: Path) -> None:
    source = tmp_path / "messages.jsonl"
    output = tmp_path / "processed"
    write_raw(
        source,
        [{"id": index, "text": f"message {index}", "is_forwarded": True} for index in range(4)],
    )

    assert prepare_dataset(source, output, validation_ratio=0.25) == (3, 1)
    train_records = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    validation_records = [
        json.loads(line) for line in (output / "validation.jsonl").read_text().splitlines()
    ]

    assert len(train_records) == 3
    assert len(validation_records) == 1
    assert all(record["messages"][-1]["role"] == "assistant" for record in train_records)


def test_invalid_raw_record_has_line_number(tmp_path: Path) -> None:
    source = tmp_path / "messages.jsonl"
    source.write_text('{"text": 5, "is_forwarded": true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Line 1"):
        load_records(source)
