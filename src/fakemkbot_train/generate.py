from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from fakemkbot_train.prepare import SYSTEM_PROMPT, USER_PROMPT
from fakemkbot_train.train import quantization_config, require_cuda


class Generator:
    def __init__(self, adapter: Path) -> None:
        require_cuda()
        if not adapter.is_dir():
            raise FileNotFoundError(f"Adapter directory does not exist: {adapter}")

        config = PeftConfig.from_pretrained(adapter)
        tokenizer = AutoTokenizer.from_pretrained(adapter, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            quantization_config=quantization_config(),
            attn_implementation="sdpa",
            device_map={"": 0},
        )
        self.model = PeftModel.from_pretrained(base_model, adapter)
        self.model.eval()
        self.tokenizer = tokenizer

    def generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        set_seed(seed)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def generate(
    adapter: Path,
    *,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> str:
    generator = Generator(adapter)
    return generator.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate text with the trained style adapter")
    parser.add_argument("--adapter", type=Path, default=Path("artifacts/mk-style"))
    parser.add_argument("--prompt", default=USER_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = generate(
        args.adapter,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )
    if not result:
        raise RuntimeError("The model generated no text")
    print(result)


if __name__ == "__main__":
    main()
