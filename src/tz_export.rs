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
    let nl = if content.contains("\r\n") { "\r\n" } else { "\n" };
    let lines: Vec<&str> = content.lines().collect();
    let mut start_idx = None;
    let mut end_idx = None;

    for (i, line) in lines.iter().enumerate() {
        // Only treat a line as a marker when it starts with `#` (markers are TOML comments).
        // This prevents content lines like `description = "# >>> GENERATED ..."` from
        // being mis-matched on a second chained splice call.
        if start_idx.is_none() && line.trim_start().starts_with('#') && line.starts_with(start_prefix) {
            start_idx = Some(i);
        } else if start_idx.is_some() && end_idx.is_none() && line.trim_start().starts_with('#') && line.starts_with(end_prefix) {
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

    // Normalise the replacement to use the dominant line ending of the source file.
    let replacement_nl: std::borrow::Cow<str> = if nl == "\r\n" {
        std::borrow::Cow::Owned(replacement.replace("\r\n", "\n").replace('\n', "\r\n"))
    } else {
        std::borrow::Cow::Borrowed(replacement)
    };

    let mut result = String::with_capacity(content.len() + replacement_nl.len());
    // Include the start marker line itself.
    for line in &lines[..=start] {
        result.push_str(line);
        result.push_str(nl);
    }
    // The new content between the markers.
    result.push_str(&replacement_nl);
    // The end marker line and everything after it.
    for line in &lines[end..] {
        result.push_str(line);
        result.push_str(nl);
    }
    Ok(result)
}

// ── public API ────────────────────────────────────────────────────────────────

/// Generate TZ tool config from the live MCP definitions.
///
/// * Writes `<config_dir>/tools/<name>.json` for every Full-profile tool (includes
///   write tools: `graph_upsert`, `graph_delete`, `index`, `reindex`, `resolve`, `purge`).
/// * Splices the `[tools.*]` TOML blocks (Full set) between the `GENERATED TOOLS` markers.
/// * Splices the `tools = [...]` line (Reader set — the answer_hotpot function) between
///   the `GENERATED TOOL LIST` markers.
/// * Splices the `tools = [...]` line (Editor set — the enrich function, a read+edit agent)
///   between the `GENERATED ENRICH TOOL LIST` markers.
///
/// Returns the number of tool JSON files written.
pub fn dump(config_dir: &Path) -> anyhow::Result<usize> {
    // 1. Ensure the tools/ sub-directory exists.
    let tools_dir = config_dir.join("tools");
    std::fs::create_dir_all(&tools_dir)
        .with_context(|| format!("create dir {}", tools_dir.display()))?;

    // 2a. Full profile — DEF set: json files + [tools.*] blocks for every tool.
    let full_srv = crate::mcp::GlossaServer::new(
        std::path::PathBuf::from("."),
        crate::mcp::Profile::Full,
        false,
        false,
    );
    let mut full_tools = full_srv.tool_specs();
    full_tools.sort_by(|a, b| a.name.cmp(&b.name));

    // 2b. Reader profile — LIST set: names for the answer-function tools = [...] line.
    let reader_srv = crate::mcp::GlossaServer::new(
        std::path::PathBuf::from("."),
        crate::mcp::Profile::Reader,
        false,
        false,
    );
    let mut reader_tools = reader_srv.tool_specs();
    reader_tools.sort_by(|a, b| a.name.cmp(&b.name));

    // 2c. Editor profile — LIST set for the ENRICH function. The enricher reads AND edits the
    // graph, so its tools are exactly the Editor profile (kept 1:1 with MCP, not hand-maintained).
    let editor_srv = crate::mcp::GlossaServer::new(
        std::path::PathBuf::from("."),
        crate::mcp::Profile::Editor,
        false,
        false,
    );
    let mut editor_tools = editor_srv.tool_specs();
    editor_tools.sort_by(|a, b| a.name.cmp(&b.name));

    // 3. Write each tool's JSON schema (from the full set).
    for tool in &full_tools {
        let mut schema: serde_json::Value = serde_json::to_value(&*tool.input_schema)?;
        // No-arg tools (Parameters<Empty>) schema-derive to a bare `{"type":"object"}`. LM Studio's
        // OpenAI-tools validator (zod) REQUIRES `function.parameters.properties` to be present, and
        // rejects the whole request (400) when it is undefined. Ensure every schema carries it.
        if let Some(obj) = schema.as_object_mut() {
            obj.entry("properties").or_insert_with(|| serde_json::json!({}));
        }
        let json = serde_json::to_string_pretty(&schema)?;
        let out = tools_dir.join(format!("{}.json", tool.name));
        std::fs::write(&out, format!("{}\n", json))
            .with_context(|| format!("write {}", out.display()))?;
    }

    // Runtime control tool `done` — NOT an MCP tool: the enrich loop intercepts it to end the
    // episode (explicit completion signal), so it never reaches glossa. Write its schema here so
    // the gateway can advertise it to the model alongside the real tools.
    let done_json = tools_dir.join("done.json");
    std::fs::write(
        &done_json,
        "{\n  \"type\": \"object\",\n  \"properties\": {\n    \"note\": {\n      \"type\": \"string\",\n      \"description\": \"One short line: the chain you wrote, or why the case needed nothing.\"\n    }\n  }\n}\n",
    )
    .with_context(|| format!("write {}", done_json.display()))?;

    // 4. Build the [tools.*] TOML blocks (full set).
    let mut blocks = String::new();
    for (i, tool) in full_tools.iter().enumerate() {
        let desc = tool.description.as_deref().unwrap_or("");
        blocks.push_str(&format!(
            "[tools.{}]\ndescription = {}\nparameters = \"tools/{}.json\"\n",
            tool.name,
            toml_quote(desc),
            tool.name,
        ));
        if i + 1 < full_tools.len() {
            blocks.push('\n');
        }
    }
    // Ensure replacement always ends with exactly one newline so the closing
    // marker starts on a fresh line.
    if !blocks.ends_with('\n') {
        blocks.push('\n');
    }
    // Declare the runtime `done` tool so the gateway advertises it (referenced by the enrich list).
    let done_desc = "Signal that the reasoning graph for this case is complete — the chain is written with a MENTIONS anchor, or `glossary` showed it already exists. Calling `done` ends the episode. Give a one-line `note`.";
    blocks.push_str(&format!(
        "\n[tools.done]\ndescription = {}\nparameters = \"tools/done.json\"\n",
        toml_quote(done_desc),
    ));

    // 5. Build the tool-list line (reader set — the answer_hotpot function).
    let names: Vec<String> = reader_tools.iter().map(|t| format!("\"{}\"", t.name)).collect();
    let tool_list = format!("tools = [{}]\n", names.join(", "));

    // 5b. Build the tool-list line (editor set — the enrich function). Append the runtime `done`
    // tool: enrich-only (the answer/reader list above does NOT get it — its completion is a text
    // answer, while the enricher signals completion explicitly).
    let mut enrich_names: Vec<String> = editor_tools.iter().map(|t| format!("\"{}\"", t.name)).collect();
    enrich_names.push("\"done\"".to_string());
    let enrich_list = format!("tools = [{}]\n", enrich_names.join(", "));

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
    let content = splice(
        &content,
        "# >>> GENERATED ENRICH TOOL LIST",
        "# <<< GENERATED ENRICH TOOL LIST",
        &enrich_list,
    )?;
    std::fs::write(&toml_path, &content)
        .with_context(|| format!("write {}", toml_path.display()))?;

    Ok(full_tools.len())
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
[functions.enrich]\n\
type = \"chat\"\n\
# >>> GENERATED ENRICH TOOL LIST\n\
tools = [\"old\"]\n\
# <<< GENERATED ENRICH TOOL LIST\n\
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

        // (a2) a NO-ARG tool (index → Parameters<Empty>) must also carry `properties` — LM Studio
        // rejects a tool schema without it. Guarded at the source (Empty's schema) AND here.
        let index_json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(config_dir.join("tools").join("index.json")).unwrap())
                .unwrap();
        assert!(
            index_json.get("properties").map(|p| p.is_object()).unwrap_or(false),
            "index.json (no-arg tool) missing object 'properties'"
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

        // (f) the enrich list is generated from the Editor profile — it carries an editor-only
        // write tool (graph_upsert, absent from the Reader/answer_hotpot list) and keeps its markers.
        assert!(
            toml_out.contains("\"graph_upsert\""),
            "enrich list missing editor-only 'graph_upsert'"
        );
        assert!(
            toml_out.contains("# >>> GENERATED ENRICH TOOL LIST")
                && toml_out.contains("# <<< GENERATED ENRICH TOOL LIST"),
            "enrich list markers missing"
        );

        // (g) the runtime `done` tool is declared and is ENRICH-ONLY: present in the enrich list,
        // absent from the answer (reader) list.
        assert!(toml_out.contains("[tools.done]"), "missing runtime [tools.done] block");
        let enrich_region = toml_out
            .split("# >>> GENERATED ENRICH TOOL LIST").nth(1).unwrap()
            .split("# <<< GENERATED ENRICH TOOL LIST").next().unwrap();
        assert!(enrich_region.contains("\"done\""), "enrich list missing runtime 'done'");
        let answer_region = toml_out
            .split("# >>> GENERATED TOOL LIST").nth(1).unwrap()
            .split("# <<< GENERATED TOOL LIST").next().unwrap();
        assert!(!answer_region.contains("\"done\""), "answer list must NOT carry 'done'");
    }

    #[test]
    fn splice_marker_in_content_does_not_corrupt_second_call() {
        // A replacement block whose text contains the literal marker phrase must not
        // be mis-matched as a marker on a subsequent splice call.
        let content = "\
# >>> GENERATED TOOLS (kb dump-tz-tools) — do not edit by hand\n\
[tools.old]\n\
description = \"old\"\n\
# <<< GENERATED TOOLS\n";
        // First splice: replace content between the markers.
        // The new text intentionally embeds the start-marker phrase inside a description.
        let replacement = "description = \"# >>> GENERATED TOOLS fake\"\n";
        let after_first = splice(
            content,
            "# >>> GENERATED TOOLS",
            "# <<< GENERATED TOOLS",
            replacement,
        ).unwrap();
        assert!(after_first.contains("fake"), "replacement not applied");
        // Second splice on the already-spliced result must still find the real markers.
        let replacement2 = "description = \"new\"\n";
        let after_second = splice(
            &after_first,
            "# >>> GENERATED TOOLS",
            "# <<< GENERATED TOOLS",
            replacement2,
        ).unwrap();
        assert!(after_second.contains("description = \"new\""), "second splice lost");
        // The content line with the embedded marker phrase must be gone now.
        assert!(!after_second.contains("fake"), "embedded marker phrase leaked into output");
    }
}
