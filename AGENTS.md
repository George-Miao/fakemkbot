# Repository Guidelines

## Project Overview

`fakemkbot` collects forwarded text from one Telegram chat, fine-tunes a small language model on MK's writing style, and serves generated quotes through a Telegram inline bot. `../realmkbot` is the Telegram behavior reference.

Never open, print, or commit `.env`. It contains only `API_HASH`, `API_ID`, `BOT_TOKEN`, and `CHAT_NAME`. Treat `data/`, Telegram session files, and trained adapters as private generated artifacts.

## Architecture & Data Flow

Keep collection, training, and serving as separate processes:

1. `src/fakemkbot_collector/src/main.rs` signs in with Grammers, resolves `CHAT_NAME`, scans message IDs, and writes forwarded text to `data/raw/messages.jsonl`.
2. `src/fakemkbot_train/prepare.py` creates deterministic train and validation JSONL files. `train.py` fine-tunes the Qwen LoRA adapter on CUDA.
3. `src/fakemkbot_train/generate.py` owns model loading and generation. `serve.py` loads the model once and exchanges newline-delimited JSON on standard input and output.
4. `src/fakemkbot/src/model.rs` starts `fakemkbot-worker`, waits for its `ready` reply, and serializes GPU requests through one child process.
5. `src/fakemkbot/src/main.rs` receives Grammers inline queries, generates one quote, and answers with exactly one Telegram `Article`.

The worker protocol uses `ready`, `result`, and `error` replies. Keep standard output exclusive to protocol JSON. Send model diagnostics to standard error.

## Key Directories

- `src/fakemkbot/`: Rust Telegram inline bot.
- `src/fakemkbot_collector/`: Rust Telegram history collector.
- `src/fakemkbot_train/`: Python dataset, training, generation, and model-worker modules.
- `tests/`: fast Python dataset and worker-protocol tests.
- `data/`: private Telegram sessions and dataset files. Never commit it.
- `artifacts/`: private adapters and checkpoints. Never commit them.

## Development Commands

Run commands in `nix develop` or after direnv loads `.envrc`.

```sh
nix develop
uv sync --dev
cargo run -p fakemkbot-collector --release -- --output data/raw/messages.jsonl
uv run fakemkbot-prepare
uv run fakemkbot-train
uv run fakemkbot-generate --adapter artifacts/mk-style
cargo run -p fakemkbot --release
uv run pytest
uv run ruff check src/fakemkbot_train tests
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

Use `--limit` only for a collector smoke test. Use the complete history for final training. The bot expects the trained adapter at `artifacts/mk-style` unless `--adapter` overrides it.

## Code Conventions & Common Patterns

- Format Rust with `rustfmt`. Use `snake_case` functions and modules, `PascalCase` types, `anyhow::Context` at process, I/O, and network boundaries, and Tokio for Grammers and child-process work.
- Keep the Telegram update loop and model worker sequential. The GPU model handles one request at a time.
- Keep secrets in one configuration boundary. Never log configuration structs, environment mappings, tokens, or Telegram sessions.
- Format Python with Ruff. Add type hints to public functions. Use `snake_case` functions and modules and `PascalCase` classes.
- Pass model settings and paths as arguments. Do not add global mutable model state or a dependency injection framework.
- Keep the worker JSON protocol small and tagged. Flush every reply. Return errors as protocol messages so one bad request does not stop the worker.
- Preserve training text. Only normalize line endings and outer whitespace.
- Keep dataset splits and training seeds deterministic. Bot generation seeds can vary per request.
- Write completed datasets through a temporary file and atomic rename.

## Important Files

- `Cargo.toml`: Rust workspace and shared dependency versions.
- `Cargo.lock`: exact Rust dependency resolution.
- `src/fakemkbot/src/main.rs`: Telegram authorization, update loop, and one-result inline answer.
- `src/fakemkbot/src/model.rs`: persistent Python worker lifecycle and JSON protocol.
- `src/fakemkbot_collector/src/main.rs`: Telegram history scan, filtering, and JSONL export.
- `src/fakemkbot_train/prepare.py`: dataset schema, filtering, and split policy.
- `src/fakemkbot_train/train.py`: Qwen 1.5B NF4 QLoRA settings and CUDA checks.
- `src/fakemkbot_train/generate.py`: reusable adapter loading and generation.
- `src/fakemkbot_train/serve.py`: persistent model worker entry point.
- `pyproject.toml` and `uv.lock`: Python commands and exact dependencies.
- `flake.nix` and `.envrc`: Rust, Python, uv, and CUDA development environment.

## Runtime/Tooling Preferences

- Use Nix flakes and `.envrc` for system tools and CUDA library paths. Rust comes from `rust-overlay` with the `rust-src` component and `RUST_SRC_PATH` configured.
- Use `uv` for Python dependencies and commands. Do not use `pip` directly.
- Use the root Cargo workspace for both Rust binaries. Grammers is the required Telegram client.
- Run the bot inside the Nix shell. Its child model worker needs the same CUDA and C++ runtime libraries.
- Train and generate on the NVIDIA RTX 3070. Use the configured NF4 QLoRA path so the 1.5B model fits. Do not add a CPU fallback.
- Load the adapter once per bot process. Never load the base model for every inline query.

## Testing & QA

Pytest covers dataset rules and the worker JSON protocol. Rust tests cover collector scan boundaries and worker reply decoding. Telegram authentication, update streaming, and CUDA inference need smoke tests because unit tests cannot prove them.

Before a completed serving change:

1. Run Ruff, Pytest, rustfmt, Clippy, and `cargo test --workspace`.
2. Start `fakemkbot-worker` and send one JSON request through the persistent process.
3. Confirm the worker returns one non-empty `result`.
4. Start `cargo run -p fakemkbot --release` and confirm `Ready for Telegram inline queries`.
5. Test one real inline query when a Telegram user session is available.

There is no numeric coverage threshold. Test stable contracts, not exact generated prose.
