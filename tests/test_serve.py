from __future__ import annotations

import io
import json

from fakemkbot_train.prepare import USER_PROMPT
from fakemkbot_train.serve import serve


class FakeGenerator:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> str:
        self.prompts.append(prompt)
        assert max_new_tokens == 80
        assert temperature == 0.9
        assert top_p == 0.9
        assert seed == 7
        return "太有创意了"


def replies(destination: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in destination.getvalue().splitlines()]


def test_serves_one_result_for_each_request() -> None:
    generator = FakeGenerator()
    destination = io.StringIO()

    serve(
        generator,
        io.StringIO(f'{{"prompt":"{USER_PROMPT}"}}\n'),
        destination,
        next_seed=lambda: 7,
    )

    assert generator.prompts == [USER_PROMPT]
    assert replies(destination) == [
        {"type": "ready"},
        {"type": "result", "text": "太有创意了"},
    ]


def test_reports_bad_requests_and_keeps_serving() -> None:
    generator = FakeGenerator()
    destination = io.StringIO()
    source = io.StringIO('{}\n{"prompt":"next"}\n')

    serve(generator, source, destination, next_seed=lambda: 7)

    assert generator.prompts == ["next"]
    assert replies(destination) == [
        {"type": "ready"},
        {"type": "error", "message": "Request must contain a string prompt"},
        {"type": "result", "text": "太有创意了"},
    ]


def test_empty_prompt_uses_training_prompt() -> None:
    generator = FakeGenerator()

    serve(generator, io.StringIO('{"prompt":""}\n'), io.StringIO(), next_seed=lambda: 7)

    assert generator.prompts == [USER_PROMPT]
