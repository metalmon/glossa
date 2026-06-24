pub mod prompt;
pub mod mock;
pub mod cli;
pub mod openai;

use crate::dataset::Question;
use std::path::Path;

pub trait AgentBackend {
    fn needs_corpus(&self) -> bool;
    fn answer(&self, work: &Path, q: &Question) -> anyhow::Result<String>;
}

/// Substitute `{prompt}` and `{mcp_config}` tokens in a CLI arg template.
pub fn substitute(template: &[String], prompt: &str, mcp_config: &str) -> Vec<String> {
    template
        .iter()
        .map(|a| match a.as_str() {
            "{prompt}" => prompt.to_string(),
            "{mcp_config}" => mcp_config.to_string(),
            other => other.to_string(),
        })
        .collect()
}

/// Run a child process to completion with a timeout, capturing stdout. Kills + errors on timeout.
pub fn run_with_timeout(
    mut cmd: std::process::Command,
    timeout: std::time::Duration,
) -> anyhow::Result<String> {
    use std::io::Read;
    use std::process::Stdio;
    use std::sync::mpsc;

    cmd.stdout(Stdio::piped()).stderr(Stdio::null());
    let mut child = cmd.spawn().map_err(|e| anyhow::anyhow!("spawn failed: {e}"))?;
    let mut stdout = child.stdout.take().expect("piped stdout");
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut s = String::new();
        let _ = stdout.read_to_string(&mut s);
        let _ = tx.send(s);
    });
    let start = std::time::Instant::now();
    loop {
        if child.try_wait()?.is_some() {
            break;
        }
        if start.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            anyhow::bail!("timeout after {:?}", timeout);
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    Ok(rx.recv_timeout(std::time::Duration::from_secs(2)).unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn substitute_replaces_tokens() {
        let t = vec!["-p".to_string(), "{prompt}".to_string(), "--cfg".to_string(), "{mcp_config}".to_string()];
        assert_eq!(substitute(&t, "Q?", "/c.json"), vec!["-p", "Q?", "--cfg", "/c.json"]);
    }

    #[test]
    fn run_with_timeout_captures_quick_output() {
        #[cfg(windows)]
        let c = { let mut c = std::process::Command::new("cmd"); c.args(["/C", "echo hi"]); c };
        #[cfg(not(windows))]
        let c = { let mut c = std::process::Command::new("sh"); c.args(["-c", "echo hi"]); c };
        let out = run_with_timeout(c, std::time::Duration::from_secs(10)).unwrap();
        assert!(out.contains("hi"));
    }
}
