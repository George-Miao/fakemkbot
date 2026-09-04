use std::{path::Path, process::Stdio};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines},
    process::{Child, ChildStdin, ChildStdout, Command},
};

#[derive(Serialize)]
struct Request<'a> {
    prompt: &'a str,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Reply {
    Ready,

    #[serde(rename = "result")]
    Generated {
        text: String,
    },

    Error {
        message: String,
    },
}

pub struct Model {
    _child: Child,
    stdin: ChildStdin,
    stdout: Lines<BufReader<ChildStdout>>,
}

impl Model {
    pub async fn spawn(adapter: &Path) -> Result<Self> {
        let mut command = Command::new("uv");
        command
            .args(["run", "--no-sync", "fakemkbot-worker", "--adapter"])
            .arg(adapter)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true);

        let mut child = command.spawn().context("Failed to start model worker")?;
        let stdin = child
            .stdin
            .take()
            .context("Model worker has no standard input")?;
        let stdout = child
            .stdout
            .take()
            .context("Model worker has no standard output")?;
        let mut model = Self {
            _child: child,
            stdin,
            stdout: BufReader::new(stdout).lines(),
        };

        match model.read_reply().await? {
            Reply::Ready => Ok(model),
            reply => Err(anyhow!(
                "Model worker sent an invalid startup reply: {reply:?}"
            )),
        }
    }

    pub async fn generate(&mut self, prompt: &str) -> Result<String> {
        let mut request =
            serde_json::to_vec(&Request { prompt }).context("Failed to encode model request")?;
        request.push(b'\n');
        self.stdin
            .write_all(&request)
            .await
            .context("Failed to write model request")?;
        self.stdin
            .flush()
            .await
            .context("Failed to flush model request")?;

        match self.read_reply().await? {
            Reply::Generated { text } if !text.trim().is_empty() => Ok(text),
            Reply::Generated { .. } => Err(anyhow!("Model worker generated empty text")),
            Reply::Error { message } => Err(anyhow!("Model worker failed: {message}")),
            Reply::Ready => Err(anyhow!("Model worker sent an unexpected ready reply")),
        }
    }

    async fn read_reply(&mut self) -> Result<Reply> {
        let line = self
            .stdout
            .next_line()
            .await
            .context("Failed to read model worker reply")?
            .context("Model worker stopped")?;
        decode_reply(&line)
    }
}

fn decode_reply(line: &str) -> Result<Reply> {
    serde_json::from_str(line).context("Model worker sent invalid JSON")
}

#[cfg(test)]
mod tests {
    use super::{Reply, decode_reply};

    #[test]
    fn decodes_unicode_result() {
        assert_eq!(
            decode_reply(r#"{"type":"result","text":"太有创意了"}"#).unwrap(),
            Reply::Generated {
                text: "太有创意了".into(),
            }
        );
    }

    #[test]
    fn rejects_unknown_reply_type() {
        assert!(decode_reply(r#"{"type":"other"}"#).is_err());
    }
}
