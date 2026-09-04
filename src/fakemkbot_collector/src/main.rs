use std::{
    env,
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use grammers_client::Client;
use grammers_mtsender::SenderPool;
use grammers_session::{storages::SqliteSession, types::PeerRef};
use serde::Serialize;

#[derive(Parser)]
#[command(about = "Collect forwarded Telegram text as JSONL")]
struct Args {
    #[arg(long, default_value = "data/raw/messages.jsonl")]
    output: PathBuf,

    #[arg(long, default_value = "data/telegram.session")]
    session: PathBuf,

    #[arg(
        long,
        default_value_t = 0,
        help = "Stop after this many records; zero means all"
    )]
    limit: usize,

    #[arg(long, help = "Override the latest Telegram message ID")]
    max_id: Option<i32>,

    #[arg(
        long,
        default_value_t = 11,
        help = "Stop after this many consecutive absent message IDs"
    )]
    max_empty_run: usize,
}

struct Config {
    api_id: i32,
    api_hash: String,
    bot_token: String,
    chat_name: String,
}

impl Config {
    fn from_env() -> Result<Self> {
        Ok(Self {
            api_id: required_env("API_ID")?
                .parse()
                .context("API_ID must be an integer")?,
            api_hash: required_env("API_HASH")?,
            bot_token: required_env("BOT_TOKEN")?,
            chat_name: required_env("CHAT_NAME")?,
        })
    }
}

#[derive(Debug, Serialize, PartialEq, Eq)]
struct MessageRecord {
    id: i32,
    text: String,
    is_forwarded: bool,
}

fn required_env(name: &str) -> Result<String> {
    let value = env::var(name).with_context(|| format!("Missing environment variable {name}"))?;
    if value.trim().is_empty() {
        return Err(anyhow!("Environment variable {name} is empty"));
    }
    Ok(value)
}

fn clean_record(id: i32, text: &str, is_forwarded: bool) -> Option<MessageRecord> {
    let text = text.trim();
    if !is_forwarded || text.is_empty() {
        return None;
    }
    Some(MessageRecord {
        id,
        text: text.to_owned(),
        is_forwarded,
    })
}

async fn connect(config: &Config, session_path: &Path) -> Result<(Client, PeerRef)> {
    if let Some(parent) = session_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .with_context(|| format!("Failed to create {}", parent.display()))?;
    }
    let session = Arc::new(
        SqliteSession::open(session_path)
            .await
            .context("Failed to load Telegram session")?,
    );
    let pool = SenderPool::new(session, config.api_id);
    let client = Client::new(pool.handle);
    tokio::spawn(pool.runner.run());
    client
        .bot_sign_in(&config.bot_token, &config.api_hash)
        .await
        .map_err(|error| anyhow!("Telegram authorization failed: {error:?}"))?;
    let chat = client
        .resolve_username(&config.chat_name)
        .await
        .context("Failed to resolve Telegram chat")?
        .ok_or_else(|| anyhow!("Telegram chat could not be resolved"))?
        .to_ref()
        .await
        .map_err(|error| anyhow!("Failed to create a Telegram peer reference: {error}"))?
        .ok_or_else(|| anyhow!("Telegram chat has no access hash"))?;
    Ok((client, chat))
}

fn message_id_batch(next_id: i32, max_id: Option<i32>) -> Vec<i32> {
    let last_id = max_id.map_or_else(
        || next_id.saturating_add(99),
        |max_id| next_id.saturating_add(99).min(max_id),
    );
    (next_id..=last_id).collect()
}

fn reached_history_end(consecutive_empty: &mut usize, has_text: bool, maximum: usize) -> bool {
    if has_text {
        *consecutive_empty = 0;
        false
    } else {
        *consecutive_empty += 1;
        *consecutive_empty >= maximum
    }
}

async fn collect(
    client: &Client,
    chat: PeerRef,
    max_id: Option<i32>,
    limit: usize,
    max_empty_run: usize,
) -> Result<Vec<MessageRecord>> {
    if max_id.is_some_and(|max_id| max_id <= 0) {
        return Err(anyhow!("Maximum message ID must be greater than zero"));
    }
    if max_empty_run == 0 {
        return Err(anyhow!("Maximum empty run must be greater than zero"));
    }

    let mut records = Vec::new();
    let mut next_id = 1_i32;
    let mut consecutive_empty = 0_usize;
    'history: loop {
        let ids = message_id_batch(next_id, max_id);
        let Some(&last_id) = ids.last() else {
            break;
        };
        let messages = client
            .get_messages_by_id(chat, &ids)
            .await
            .context("Failed to read chat messages by ID")?;
        let mut messages = messages.into_iter();

        for _ in &ids {
            let message = messages.next().flatten();
            let has_text = message
                .as_ref()
                .is_some_and(|message| !message.text().trim().is_empty());
            if reached_history_end(&mut consecutive_empty, has_text, max_empty_run) {
                break 'history;
            }
            if !has_text {
                continue;
            }
            let message = message.expect("a text message was checked above");
            if let Some(record) = clean_record(
                message.id(),
                message.text(),
                message.forward_header().is_some(),
            ) {
                records.push(record);
                if limit != 0 && records.len() >= limit {
                    break 'history;
                }
            }
        }

        if last_id == i32::MAX {
            break;
        }
        next_id = last_id + 1;
    }
    Ok(records)
}

fn write_records(path: &Path, records: &[MessageRecord]) -> Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).with_context(|| format!("Failed to create {}", parent.display()))?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow!("Output path needs a file name"))?;
    let temporary_path = parent.join(format!(".{file_name}.tmp"));

    let result = (|| -> Result<()> {
        let file = File::create(&temporary_path)
            .with_context(|| format!("Failed to create {}", temporary_path.display()))?;
        let mut writer = BufWriter::new(file);
        for record in records {
            serde_json::to_writer(&mut writer, record).context("Failed to encode message")?;
            writer.write_all(b"\n").context("Failed to write message")?;
        }
        writer.flush().context("Failed to flush message file")?;
        writer
            .get_ref()
            .sync_all()
            .context("Failed to sync message file")?;
        drop(writer);
        fs::rename(&temporary_path, path)
            .with_context(|| format!("Failed to replace {}", path.display()))?;
        Ok(())
    })();

    if result.is_err() {
        let _ = fs::remove_file(&temporary_path);
    }
    result
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    let args = Args::parse();
    let config = Config::from_env()?;
    let (client, chat) = connect(&config, &args.session).await?;
    if let Some(max_id) = args.max_id {
        println!("Scanning message IDs 1 through {max_id}");
    } else {
        println!("Scanning message IDs from 1 until history ends");
    }
    let records = collect(&client, chat, args.max_id, args.limit, args.max_empty_run).await?;
    if records.is_empty() {
        return Err(anyhow!("No forwarded text messages were found"));
    }
    write_records(&args.output, &records)?;
    println!("Collected {} forwarded text messages", records.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{MessageRecord, clean_record, message_id_batch, reached_history_end};

    #[test]
    fn keeps_forwarded_non_empty_text() {
        assert_eq!(
            clean_record(7, "  hello  ", true),
            Some(MessageRecord {
                id: 7,
                text: "hello".to_owned(),
                is_forwarded: true,
            })
        );
    }

    #[test]
    fn rejects_non_forwarded_or_empty_text() {
        assert_eq!(clean_record(7, "hello", false), None);
        assert_eq!(clean_record(7, " \n ", true), None);
    }

    #[test]
    fn scans_a_batch_past_the_stale_read_marker() {
        let ids = message_id_batch(1301, None);

        assert_eq!(ids.first(), Some(&1301));
        assert_eq!(ids.last(), Some(&1400));
        assert!(ids.contains(&1348));
    }

    #[test]
    fn caps_a_batch_at_an_explicit_maximum() {
        let ids = message_id_batch(1301, Some(1348));

        assert_eq!(ids.last(), Some(&1348));
    }

    #[test]
    fn stops_after_consecutive_empty_messages() {
        let mut consecutive_empty = 0;

        for _ in 0..10 {
            assert!(!reached_history_end(&mut consecutive_empty, false, 11));
        }
        assert!(reached_history_end(&mut consecutive_empty, false, 11));
        assert!(!reached_history_end(&mut consecutive_empty, true, 11));
        assert_eq!(consecutive_empty, 0);
    }
}
