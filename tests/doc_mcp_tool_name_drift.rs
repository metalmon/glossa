//! Docs-vs-registry drift guard for MCP tool names.
//!
//! `glossa::tools::registry::catalog()` is the single source of truth for every MCP route (see
//! `tests/tool_registry_parity.rs`, which pins it at exactly 26). `docs/mcp.md`'s `## Tools`
//! table is meant to list the same 26 names — this test is the mechanical half of keeping that
//! true: if the table ever names a tool that doesn't exist (a typo, a rename the doc missed), CI
//! fails instead of the doc quietly lying for the next seven releases.
//!
//! Scope is deliberately narrow: this checks ONLY that one table's first column, not every bare
//! word in `docs/mcp.md` or elsewhere that happens to match a tool name. A tool name like
//! `index`, `read`, `note`, or `del` is also an ordinary English word, so matching free prose
//! against the catalog would false-positive constantly on unrelated uses; a markdown table row's
//! first cell, under the `## Tools` heading, is the one place a bare identifier unambiguously
//! claims to be "the tool this row documents".

use glossa::doc_lint::extract_table_first_column;
use glossa::tools::registry::catalog;
use std::collections::HashSet;
use std::path::Path;

#[test]
fn mcp_tools_table_does_not_claim_a_tool_that_does_not_exist() {
    let known: HashSet<&str> = catalog().iter().map(|m| m.name).collect();
    assert!(
        known.len() >= 20,
        "suspiciously few tools in the registry catalog: {}",
        known.len()
    );

    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("docs/mcp.md");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));

    let claims = extract_table_first_column(&text, "## Tools");
    assert!(
        claims.len() >= 20,
        "suspiciously few tool names extracted from docs/mcp.md's ## Tools table ({}) — \
         is extract_table_first_column still matching that heading?",
        claims.len()
    );

    let violations: Vec<String> = claims
        .iter()
        .filter(|(_, name)| !known.contains(name.as_str()))
        .map(|(line, name)| {
            format!(
                "{}:{line}: `## Tools` table claims a tool `{name}` that is not in \
                 glossa::tools::registry::catalog() — fix the doc, or add the tool to the registry",
                path.display()
            )
        })
        .collect();
    assert!(violations.is_empty(), "\n{}", violations.join("\n"));
}
