# fakemkbot

`fakemkbot` collects forwarded Telegram messages, trains a small language model on their writing style, and serves generated text through a Telegram inline bot.

Generated text is an imitation. It is not an authentic quote. Use the project only with permission from the person whose messages form the dataset.

## Requirements

- Nix with flakes enabled
- An NVIDIA GPU with CUDA support
- Telegram API credentials and a bot token
- `direnv` is optional

The tested system uses an NVIDIA RTX 3070. Training and inference use CUDA. There is no CPU fallback.

## Setup

Create the local environment file. Do not commit it.

```sh
cp .env.example .env
```

Set these values in `.env`:

- `API_ID`: Telegram application ID
- `API_HASH`: Telegram application hash
- `BOT_TOKEN`: token from BotFather
- `CHAT_NAME`: public username of the source chat

Enter the development shell:

```sh
nix develop
```

The shell installs Rust, Python 3.12, `uv`, CUDA runtime libraries, and project dependencies. With direnv, run `direnv allow` once instead.

## Data Flow

```text
Telegram chat
  -> fakemkbot-collector
  -> data/raw/messages.jsonl
  -> fakemkbot-prepare
  -> data/processed/{train,validation}.jsonl
  -> fakemkbot-train
  -> artifacts/mk-style
  -> fakemkbot-worker
  -> Telegram inline bot
```

`data/`, `artifacts/`, `.env`, and Telegram session files are private and excluded from Git.

## Collect Messages

The collector keeps non-empty forwarded text and writes ordered JSONL records.

```sh
cargo run -p fakemkbot-collector --release
```

The default output is `data/raw/messages.jsonl`. The default session is `data/telegram.session`.

For a small connection test:

```sh
cargo run -p fakemkbot-collector --release -- --limit 10
```

Use the complete history for real training. `--max-id ID` sets a known upper message ID when automatic discovery is not suitable.

## Prepare the Dataset

```sh
uv run fakemkbot-prepare
```

This command removes duplicate or empty text and creates a deterministic training and validation split under `data/processed/`.

## Train

```sh
uv run fakemkbot-train
```

The default training path uses `Qwen/Qwen2.5-1.5B-Instruct` with NF4 QLoRA. It writes the selected adapter and training summary to `artifacts/mk-style/`.

## Generate One Message

```sh
uv run fakemkbot-generate --adapter artifacts/mk-style
```

Use `--seed` for repeatable output during evaluation:

```sh
uv run fakemkbot-generate --adapter artifacts/mk-style --seed 42
```

## Run the Inline Bot

Train the adapter first, then start the bot inside the Nix shell:

```sh
cargo run -p fakemkbot --release
```

The Rust bot starts one persistent `fakemkbot-worker` process. The worker loads the model once and handles generation requests in sequence.

## Architecture

- `src/fakemkbot_collector/`: Rust Telegram history collector and JSONL exporter
- `src/fakemkbot_train/`: Python dataset preparation, QLoRA training, generation, and worker protocol
- `src/fakemkbot/`: Rust Telegram inline bot and Python worker manager
- `tests/`: Python dataset and worker protocol tests
- `AGENTS.md`: repository conventions for coding assistants

## Checks

Run all commands inside `nix develop`:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
uv run ruff check src/fakemkbot_train tests
uv run pytest
nix flake check --no-build
```

Python tests do not load the CUDA model. Verify a trained adapter separately with `fakemkbot-generate` or `fakemkbot-worker`.
