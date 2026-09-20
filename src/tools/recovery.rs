//! Reader-recovery responses shared by the MCP server (`mcp::apply_signals`) and the eval agent
//! loop, so a GOVERNED reasoning reader sees the SAME feedback on a repeat/streak in both surfaces.
//!
//! Detection (what counts as a repeat/streak/plateau) lives in
//! [`crate::tools::retrieval_progress::ReaderSignals`]; this module is the RESPONSE side:
//!   - **repeat** → [`next_best_action`]: fan the fixated term across the complementary tools and
//!     return their non-empty results — concrete alternatives instead of the same dead body.
//!   - **streak** → [`unproductive_steer`]: a short neutral-ish steer that the last few varied
//!     calls surfaced nothing new.
//!   - (**plateau** stays a neutral observation emitted by `ReaderSignals` itself.)
//!
//! Relocated from `kb-eval`'s `backend::glossa_tools` so both callers share ONE implementation
//! (previously the eval loop owned these and the MCP server only had bare markers — the two
//! surfaces disagreed on what a stuck reader saw). The eval side now delegates here.

use crate::graph::store::GraphStore;
use crate::index::store::DocIndex;
use crate::tools::ChainSpec;
use crate::trace::TraceLog;
use serde_json::Value;

/// Steer fed back when the reader has strung together `STREAK_K` calls that each surfaced nothing
/// NEW — even though the calls themselves varied (different tool/args). This is the over-search
/// spiral an exact-repeat dedup does not catch: many different probes landing on already-seen
/// ground. Fires once per streak (the caller resets its counter after using this).
pub fn unproductive_steer(_name: &str) -> String {
    "(no new information) Your last few calls surfaced nothing new — more searching the same way \
     won't help. Commit your best SPECIFIC answer from what you've already found, or change \
     approach fundamentally (a different entity or relation) — do not just rephrase the same query."
        .to_string()
}

/// Fallback when a repeated call has no clean fan-out term (an id/path/SQL repeat) or nothing
/// complementary comes back.
pub fn repeat_nudge(name: &str) -> String {
    format!(
        "(skipped) You already called `{name}` with these exact arguments — its result is above and \
         rerunning returns the same thing. Try a DIFFERENT tool, or change the arguments/query."
    )
}

/// The free-text intent a repeated call fixated on. Only the two text lookups give a clean term to
/// fan out; ids/paths/SQL give none, so those callers fall back to [`repeat_nudge`]. Parses the raw
/// tool args (the eval loop's `Value` shape); the MCP server passes the typed term directly to
/// [`next_best_action`] instead.
pub fn repeated_term(name: &str, args: &Value) -> Option<String> {
    let key = match name {
        "glossary" => "name",
        "search" => "query",
        _ => return None,
    };
    let t = args.get(key)?.as_str()?.trim();
    (!t.is_empty()).then(|| t.to_string())
}

/// A tool body that carries no usable result — a miss. Cheap heuristic (real relevance scoring is a
/// systemic follow-up's job).
fn looks_empty(body: &str) -> bool {
    let b = body.trim().to_lowercase();
    b.len() < 8
        || b.contains("no matches")
        || b.contains("not found")
        || b.contains("no results")
        || b.contains("(none")
        || b.contains("unavailable")
}

/// Next-best-action on a stuck (repeated) call: the model re-issued `name` with the same `term`, so
/// re-running is a dead end. Fan `term` across the COMPLEMENTARY tools (the ones it did NOT just
/// call) and return their non-empty results fused — concrete alternatives instead of the same dead
/// result. Falls back to [`repeat_nudge`] when nothing complementary comes back. Graph-backed
/// candidates (glossary/sql) are skipped when no graph is present.
pub fn next_best_action(
    name: &str,
    term: &str,
    idx: &DocIndex,
    graph: Option<&GraphStore>,
    spec: &ChainSpec,
    trace: &TraceLog,
) -> String {
    let mut out = format!(
        "(skipped) You already called `{name}` twice with the same arguments — that is a dead end. \
         Here is what the OTHER tools return for \"{term}\"; use one of these or change your query:\n"
    );
    let mut any = false;
    let fold = |tool: &str, body: &str, any: &mut bool, out: &mut String| {
        let body = body.trim();
        if body.is_empty() || looks_empty(body) {
            return;
        }
        *any = true;
        let snip: String = body.chars().take(600).collect();
        out.push_str(&format!("\n[{tool}]\n{snip}\n"));
    };
    if name != "search" {
        let (body, _) = crate::tools::search(idx, term, 12, None, None, trace, None);
        fold("search", &body, &mut any, &mut out);
    }
    if let Some(g) = graph {
        if name != "glossary" {
            let body = crate::tools::glossary(idx, g, term, spec, trace, None, None, None);
            fold("glossary", &body, &mut any, &mut out);
        }
        if name != "sql" {
            let t = term.replace('\'', " ");
            let q = format!(
                "SELECT src_label, edge_type, dst_label FROM edges_labeled \
                 WHERE src_label LIKE '%{t}%' OR dst_label LIKE '%{t}%' LIMIT 12"
            );
            let body = crate::tools::sql(idx, g, &q, trace);
            fold("sql", &body, &mut any, &mut out);
        }
    }
    if any {
        out
    } else {
        repeat_nudge(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn repeated_term_extracts_text_intent_only() {
        assert_eq!(
            repeated_term("glossary", &json!({"name":"Acme"})).as_deref(),
            Some("Acme")
        );
        assert_eq!(
            repeated_term("search", &json!({"query":"blue widget"})).as_deref(),
            Some("blue widget")
        );
        // ids/paths/SQL give no clean fan-out term.
        assert_eq!(repeated_term("read", &json!({"path":"a.md#1"})), None);
        assert_eq!(repeated_term("sql", &json!({"sql":"SELECT 1"})), None);
        // empty term is treated as no term.
        assert_eq!(repeated_term("search", &json!({"query":"  "})), None);
    }

    #[test]
    fn steer_and_nudge_mention_the_situation_without_being_a_plateau_directive() {
        assert!(unproductive_steer("search").contains("nothing new"));
        assert!(repeat_nudge("search").contains("already called"));
    }
}
