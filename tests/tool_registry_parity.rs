//! Cross-surface tool-name parity guard.
//!
//! `glossa::tools::registry::registry()` is the single source of truth for the 7
//! agent-facing reasoning tools (search, read, glossary, reach, grep, glob, graph_query).
//! Every surface that hands tools to a model must expose a superset of that registry:
//!
//!   - MCP Reader profile   — the live `GlossaServer` tool_specs() an MCP client sees.
//!   - TZ reader dump       — the tool list `kb dump-tz-tools` splices into the
//!     `answer_hotpot` function's `tools = [...]` line.
//!   - OpenAI/kbx agent loop — the function list the kb-eval crate's OpenAI-format
//!     tool builder sends to the model.
//!
//! The MCP Reader profile and the TZ reader dump are the SAME source: both come from
//! `GlossaServer::new(.., Profile::Reader, ..).tool_specs()` (see
//! `src/tz_export.rs::reader_tool_names()`), so a single glossa-side helper covers both
//! assertions below. The relationship is subset, not equality — the Reader MCP profile
//! intentionally carries a few extra non-reasoning tools the answer-loop never calls
//! (`get_source_file`, `get_ontology`, `ls`); see the doc comment on
//! `reader_profile_tool_names_match_registry` in `src/tz_export.rs` for why.
//!
//! The OpenAI side is asserted separately in the kb-eval crate
//! (`openai_tools_match_registry_graph_on`) — noted here for completeness, not re-run.

#[test]
fn all_glossa_surfaces_expose_at_least_the_registry_tools() {
    let reg: Vec<String> = {
        use glossa::tools::registry::{resolve_tools, FeatureSet, Tier, ToolContext};
        let ctx = ToolContext {
            profile: Tier::Reader,
            graph_on: true,
            verify_available: true,
            no_source_file: false,
            no_image: false,
            features: FeatureSet::default(),
        };
        resolve_tools(&ctx)
            .iter()
            .map(|t| t.name.to_string())
            .collect()
    };
    assert!(!reg.is_empty(), "resolve_tools returned no tools");

    // TZ side: the tool-name list `kb dump-tz-tools` would splice into the reader
    // (`answer_hotpot`) function's `tools = [...]` line.
    let tz: Vec<String> = glossa::tz_export::reader_tool_names();
    for name in &reg {
        // `verify` fails closed when the corpus is uncalibrated: `GlossaServer` (built here on
        // `"."`, which has no calibrated `[verify]` ontology) drops it from the live tool_specs(),
        // so it is absent from `reader_tool_names()` even though it IS in `registry()`. Skip it in
        // the parity assertion — mirrors the identical `if name == "verify" { continue; }` skip in
        // the in-crate `reader_profile_tool_names_match_registry` test (src/tz_export.rs).
        if name == "verify" {
            continue;
        }
        assert!(
            tz.contains(name),
            "registry tool '{name}' missing from the TZ reader dump (tz has: {tz:?})"
        );
    }

    // MCP side: the Reader-profile `tool_specs()` an MCP client would see. This is the
    // exact same call `reader_tool_names()` makes internally, so reuse it rather than
    // re-deriving `GlossaServer`/`Profile::Reader`/`ServerFlags` construction here.
    let mcp: Vec<String> = glossa::tz_export::reader_tool_names();
    for name in &reg {
        // Same fail-closed skip as above: uncalibrated `verify` is not in the live Reader profile.
        if name == "verify" {
            continue;
        }
        assert!(
            mcp.contains(name),
            "registry tool '{name}' missing from the MCP Reader profile (mcp has: {mcp:?})"
        );
    }
}
