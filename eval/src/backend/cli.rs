use super::{prompt, run_with_timeout, substitute, AgentBackend};
use crate::dataset::Question;
use std::path::Path;
use std::process::Command;
use std::time::Duration;

/// Generic CLI-agent backend: runs an arbitrary command that is itself an MCP client.
/// `args` is a template; `{prompt}` and `{mcp_config}` tokens are substituted. If the template
/// contains `{mcp_config}`, a glossa MCP config (work dir, trace on) is written and its path
/// substituted; otherwise the operator is expected to have configured the agent's MCP externally.
pub struct CliBackend {
    pub command: String,
    pub args: Vec<String>,
    pub kb_bin: String,
    pub profile: String,
    pub no_graph: bool,
    pub timeout: Duration,
}

impl CliBackend {
    /// Default args template for `claude -p` acting as the glossa MCP client.
    pub fn claude_preset() -> Vec<String> {
        vec![
            "-p".to_string(), "{prompt}".to_string(),
            "--mcp-config".to_string(), "{mcp_config}".to_string(),
            "--permission-mode".to_string(), "bypassPermissions".to_string(),
        ]
    }
}

impl AgentBackend for CliBackend {
    fn needs_corpus(&self) -> bool {
        true
    }
    fn answer(&self, work: &Path, q: &Question) -> anyhow::Result<String> {
        let mcp_config = if self.args.iter().any(|a| a == "{mcp_config}") {
            let mut margs = vec!["mcp".to_string(), "--profile".to_string(), self.profile.clone(), "--trace".to_string()];
            if self.no_graph {
                margs.push("--no-graph".to_string());
            }
            margs.push(work.display().to_string());
            let cfg = serde_json::json!({ "mcpServers": { "glossa": { "command": self.kb_bin, "args": margs } } });
            let cfg_path = work.join(".eval-mcp.json");
            std::fs::write(&cfg_path, serde_json::to_string(&cfg)?)?;
            cfg_path.display().to_string()
        } else {
            String::new()
        };
        let p = prompt::build_prompt(q);
        let final_args = substitute(&self.args, &p, &mcp_config);
        let mut cmd = Command::new(&self.command);
        cmd.args(&final_args);
        let out = run_with_timeout(cmd, self.timeout)?;
        Ok(prompt::parse_answer(&out))
    }
}
