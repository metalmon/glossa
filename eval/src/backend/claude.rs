use super::{prompt, AgentBackend};
use crate::dataset::Question;
use anyhow::Context;
use std::path::Path;
use std::process::Command;

/// Drives `claude -p` as a headless agent that is itself the glossa MCP client.
pub struct ClaudeBackend {
    pub kb_bin: String,
    pub profile: String,
    pub no_graph: bool,
}

impl AgentBackend for ClaudeBackend {
    fn needs_corpus(&self) -> bool {
        true
    }
    fn answer(&self, work: &Path, q: &Question) -> anyhow::Result<String> {
        let mut args = vec!["mcp".to_string(), "--profile".to_string(), self.profile.clone(), "--trace".to_string()];
        if self.no_graph {
            args.push("--no-graph".to_string());
        }
        args.push(work.display().to_string());
        let cfg = serde_json::json!({ "mcpServers": { "glossa": { "command": self.kb_bin, "args": args } } });
        let cfg_path = work.join(".claude-mcp.json");
        std::fs::write(&cfg_path, serde_json::to_string(&cfg)?)?;

        // NOTE: claude CLI flags are best-effort; verify against the installed version.
        let out = Command::new("claude")
            .arg("-p")
            .arg(prompt::build_prompt(q))
            .arg("--mcp-config")
            .arg(&cfg_path)
            .arg("--permission-mode")
            .arg("bypassPermissions")
            .output()
            .context("spawn claude -p")?;
        Ok(prompt::parse_answer(&String::from_utf8_lossy(&out.stdout)))
    }
}
