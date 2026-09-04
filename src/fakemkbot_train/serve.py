from __future__ import annotations

import argparse
import json
import secrets
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, TextIO

from fakemkbot_train.generate import Generator
from fakemkbot_train.prepare import USER_PROMPT


class TextGenerator(Protocol):
    def generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> str: ...


def write_reply(destination: TextIO, value: dict[str, object]) -> None:
    destination.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    destination.write("\n")
    destination.flush()


def request_prompt(value: Any) -> str:
    if not isinstance(value, dict) or not isinstance(value.get("prompt"), str):
        raise ValueError("Request must contain a string prompt")
    return value["prompt"].strip() or USER_PROMPT


def serve(
    generator: TextGenerator,
    source: TextIO,
    destination: TextIO,
    *,
    max_new_tokens: int = 80,
    temperature: float = 0.9,
    top_p: float = 0.9,
    next_seed: Callable[[], int] = lambda: secrets.randbelow(2**31),
) -> None:
    write_reply(destination, {"type": "ready"})
    for line in source:
        try:
            prompt = request_prompt(json.loads(line))
            text = generator.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=next_seed(),
            )
            if not text:
                raise RuntimeError("The model generated no text")
            reply: dict[str, object] = {"type": "result", "text": text}
        except Exception as error:
            reply = {"type": "error", "message": str(error)}
        write_reply(destination, reply)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the MK style model over JSON lines")
    parser.add_argument("--adapter", type=Path, default=Path("artifacts/mk-style"))
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.9)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    generator = Generator(args.adapter)
    serve(
        generator,
        sys.stdin,
        sys.stdout,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )


if __name__ == "__main__":
    main()
