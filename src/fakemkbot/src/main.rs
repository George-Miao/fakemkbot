mod model;

use std::{env, path::PathBuf, sync::Arc};

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use grammers_client::{
    Client,
    update::{Article, Update},
};
use grammers_mtsender::SenderPool;
use grammers_session::storages::SqliteSession;
use model::Model;

#[derive(Parser)]
#[command(about = "Serve generated MK quotes as Telegram inline results")]
struct Args {
    #[arg(long, default_value = "artifacts/mk-style")]
    adapter: PathBuf,

    #[arg(long, default_value = "data/fakemkbot.session")]
    session: PathBuf,
}

struct Config {
    api_id: i32,
    api_hash: String,
    bot_token: String,
}

impl Config {
    fn from_env() -> Result<Self> {
        let api_id = required_env("API_ID")?
            .parse()
            .context("API_ID must be an integer")?;
        Ok(Self {
            api_id,
            api_hash: required_env("API_HASH")?,
            bot_token: required_env("BOT_TOKEN")?,
        })
    }
}

fn required_env(name: &str) -> Result<String> {
    env::var(name).with_context(|| format!("Missing required environment variable {name}"))
}

async fn handle_update(model: &mut Model, update: Update) -> Result<()> {
    let Update::InlineQuery(query) = update else {
        return Ok(());
    };

    let quote = model.generate(query.text()).await?;
    let article = Article::new("MK quote", quote.clone()).description(quote);
    query
        .answer([article])
        .cache_time(0)
        .send()
        .await
        .context("Failed to answer inline query")
}

async fn run(args: Args, config: Config) -> Result<()> {
    if let Some(parent) = args.session.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .context("Failed to create Telegram session directory")?;
    }

    let mut model = Model::spawn(&args.adapter).await?;
    let session = Arc::new(
        SqliteSession::open(&args.session)
            .await
            .context("Failed to open Telegram session")?,
    );
    let SenderPool {
        runner,
        handle,
        updates,
    } = SenderPool::new(session, config.api_id);
    let client = Client::new(handle);
    let _runner = tokio::spawn(runner.run());

    if !client
        .is_authorized()
        .await
        .context("Failed to read Telegram authorization state")?
    {
        client
            .bot_sign_in(&config.bot_token, &config.api_hash)
            .await
            .context("Telegram bot sign-in failed")?;
    }

    let me = client
        .get_me()
        .await
        .map_err(|error| anyhow!("Failed to get Telegram bot identity: {error}"))?;
    println!(
        "get_me: id={} username={:?} first_name={:?} last_name={:?}",
        me.id(),
        me.username(),
        me.first_name(),
        me.last_name()
    );

    let mut updates = client
        .stream_updates(updates, Default::default())
        .await
        .map_err(|error| anyhow!("Failed to start Telegram update stream: {error}"))?;
    println!("Ready for Telegram inline queries");

    loop {
        tokio::select! {
            signal = tokio::signal::ctrl_c() => {
                signal.context("Failed to listen for shutdown signal")?;
                break;
            }
            update = updates.next() => {
                let update = update.context("Failed to receive Telegram update")?;
                if let Err(error) = handle_update(&mut model, update).await {
                    eprintln!("Failed to handle Telegram update: {error:#}");
                }
            }
        }
    }

    Ok(())
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    run(Args::parse(), Config::from_env()?).await
}
