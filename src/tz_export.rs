//! Generate TensorZero tool config from the live MCP tool definitions.
//!
//! `dump(config_dir)` writes `tools/<name>.json` for every tool exposed by
//! `GlossaServer` in Reader profile, then splices the `[tools.*]` blocks and
//! the `tools = [...]` function-level list into the marked regions of
//! `tensorzero.toml`.  The markers must already exist in the toml file; if they
//! are absent the function returns a helpful error.

use anyhow::Context;
use std::path::Path;

// ── helpers ──────────────────────────────────────────────────────────────────

/// TOML-quote a single-line string: escape `\` and `"`, then wrap in `"…"`.
/// Descriptions produced by the `#[schemars(description = "…")]` macro are
/// always single-line, so we never need a TOML multi-line literal.
fn toml_quote(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

/// Replace the content between the first line that *starts with* `start_prefix`
/// and the next line that *starts with* `end_prefix` with `replacement`.
/// Both marker lines are preserved; only the text between them changes.
fn splice(
    content: &str,
    start_prefix: &str,
    end_prefix: &str,
    replacement: &str,
) -> anyhow::Result<String> {
    let lines: Vec<&str> = content.lines().collect();
    let mut start_idx = None;
    let mut end_idx = None;

    for (i, line) in lines.iter().enumerate() {
        if start_idx.is_none() && line.starts_with(start_prefix) {
            start_idx = Some(i);
        } else if start_idx.is_some() && end_idx.is_none() && line.starts_with(end_prefix) {
            end_idx = Some(i);
        }
    }

    let start = start_idx.ok_or_else(|| {
        anyhow::anyhow!(
            "Missing marker '{}' in tensorzero.toml — add it once per the task brief.",
            start_prefix
        )
    })?;
    let end = end_idx.ok_or_else(|| {
        anyhow::anyhow!(
            "Missing closing marker '{}' in tensorzero.toml — add it once per the task brief.",
            end_prefix
        )
    })?;

    let mut result = String::with_capacity(content.len() + replacement.len());
    // Include the start marker line itself.
    for line in &lines[..=start] {
        result.push_str(line);
        result.push('\n');
    }
    // The new content between the markers.
    result.push_str(replacement);
    // The end marker line and everything after it.
    for line in &lines[end..] {
        result.push_str(line);
        result.push('\n');
    }
    Ok(result)
}

// ── public API ────────────────────────────────────────────────────────────────

/// Generate TZ tool config from the live MCP definitions.
///
/// * Writes `<config_dir>/tools/<name>.json` for each Reader-profile tool.
/// * Splices the `[tools.*]` TOML blocks between the `GENERATED TOOLS` markers.
/// * Splices the `tools = [...]` line between the `GENERATED TOOL LIST` markers.
///
/// Returns the number of tools written.
pub fn dump(config_dir: &Path) -> anyhow::Result<usize> {
    // 1. Ensure the tools/ sub-directory exists.
    let tools_dir = config_dir.join("tools");
    std::fs::create_dir_all(&tools_dir)
        .with_context(|| format!("create dir {}", tools_dir.display()))?;

    // 2. Instantiate with Reader profile to get the 6 query tools.
    let srv = crate::mcp::GlossaServer::new(
        std::path::PathBuf::from("."),
        crate::mcp::Profile::Reader,
        false,
        false,
    );
    let mut tools = srv.tool_specs();

    // 3. Sort by name for stable, deterministic output.
    tools.sort_by(|a, b| a.name.cmp(&b.name));

    // 4. Write each tool's JSON schema.
    for tool in &tools {
        let json = serde_json::to_string_pretty(&*tool.input_schema)?;
        let out = tools_dir.join(format!("{}.json", tool.name));
        std::fs::write(&out, format!("{}\n", json))
            .with_context(|| format!("write {}", out.display()))?;
    }

    // 5. Build the [tools.*] TOML blocks.
    let mut blocks = String::new();
    for (i, tool) in tools.iter().enumerate() {
        let desc = tool.description.as_deref().unwrap_or("");
        blocks.push_str(&format!(
            "[tools.{}]\ndescription = {}\nparameters = \"tools/{}.json\"\n",
            tool.name,
            toml_quote(desc),
            tool.name,
        ));
        if i + 1 < tools.len() {
            blocks.push('\n');
        }
    }
    // Ensure replacement always ends with exactly one newline so the closing
    // marker starts on a fresh line.
    if !blocks.ends_with('\n') {
        blocks.push('\n');
    }

    // 6. Build the tool-list line.
    let names: Vec<String> = tools.iter().map(|t| format!("\"{}\"", t.name)).collect();
    let tool_list = format!("tools = [{}]\n", names.join(", "));

    // 7. Splice both regions into tensorzero.toml.
    let toml_path = config_dir.join("tensorzero.toml");
    let content = std::fs::read_to_string(&toml_path)
        .with_context(|| format!("read {}", toml_path.display()))?;
    let content = splice(
        &content,
        "# >>> GENERATED TOOLS",
        "# <<< GENERATED TOOLS",
        &blocks,
    )?;
    let content = splice(
        &content,
        "# >>> GENERATED TOOL LIST",
        "# <<< GENERATED TOOL LIST",
        &tool_list,
    )?;
    std::fs::write(&toml_path, &content)
        .with_context(|| format!("write {}", toml_path.display()))?;

    Ok(tools.len())
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_dump_tz_tools() {
        let tmp = TempDir::new().unwrap();
        let config_dir = tmp.path();

        // Minimal tensorzero.toml with all four markers and surrounding text
        // that must survive unchanged.
        let toml_content = "[functions.x]\ntype = \"chat\"\n\n\
# >>> GENERATED TOOLS (kb dump-tz-tools) \u{2014} do not edit by hand\n\
[tools.old]\n\
description = \"old\"\n\
parameters = \"tools/old.json\"\n\
# <<< GENERATED TOOLS\n\
\n\
[functions.answer_hotpot]\n\
type = \"chat\"\n\
# >>> GENERATED TOOL LIST\n\
tools = [\"old\"]\n\
# <<< GENERATED TOOL LIST\n\
\n\
[metrics.em]\n\
type = \"boolean\"\n";

        std::fs::write(config_dir.join("tensorzero.toml"), toml_content).unwrap();
        std::fs::create_dir_all(config_dir.join("tools")).unwrap();

        let n = dump(config_dir).unwrap();
        // Reader profile has 6 tools: glob, glossary, grep, neighbors, read, search
        assert!(n >= 1, "expected at least 1 tool, got {}", n);

        // (a) tools/search.json exists and parses as JSON with a "properties" key
        let search_json_path = config_dir.join("tools").join("search.json");
        assert!(search_json_path.exists(), "tools/search.json not found");
        let search_json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&search_json_path).unwrap()).unwrap();
        assert!(
            search_json.get("properties").is_some(),
            "search.json missing 'properties'"
        );

        // (b) GENERATED-TOOLS region contains [tools.search] + correct parameters path
        let toml_out = std::fs::read_to_string(config_dir.join("tensorzero.toml")).unwrap();
        assert!(toml_out.contains("[tools.search]"), "missing [tools.search]");
        assert!(
            toml_out.contains("parameters = \"tools/search.json\""),
            "missing parameters for search"
        );

        // (c) surrounding non-marked text is unchanged
        assert!(
            toml_out.contains("[functions.x]"),
            "surrounding text [functions.x] missing"
        );
        assert!(
            toml_out.contains("[metrics.em]"),
            "surrounding text [metrics.em] missing"
        );

        // (d) tool-list region contains "neighbors"
        assert!(
            toml_out.contains("\"neighbors\""),
            "tool list missing 'neighbors'"
        );

        // (e) all four markers are preserved
        assert!(
            toml_out.contains("# >>> GENERATED TOOLS"),
            "start tools marker missing"
        );
        assert!(
            toml_out.contains("# <<< GENERATED TOOLS"),
            "end tools marker missing"
        );
        assert!(
            toml_out.contains("# >>> GENERATED TOOL LIST"),
            "list start marker missing"
        );
        assert!(
            toml_out.contains("# <<< GENERATED TOOL LIST"),
            "list end marker missing"
        );
    }
}
