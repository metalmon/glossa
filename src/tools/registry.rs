//! Single source of truth for agent tool DECLARATIONS (name, description, JSON schema,
//! graph-gated flag). MCP (src/mcp.rs), and later the OpenAI/TZ surfaces, build their
//! tool listings from `registry()` instead of hand-duplicating them, so the three
//! surfaces cannot drift apart. Descriptions are extracted here verbatim from the
//! current MCP `#[tool(description = …)]` attributes; schemas come from the SAME arg
//! structs `src/mcp.rs` already deserializes into (`schemars::schema_for!`), normalized
//! to the OpenAI-function core `{ "type": "object", "properties": {…}, "required": […] }`.

use crate::mcp::{
    GlobArgs, GlossaryArgs, GrepArgs, NeighborsArgs, ReachArgs, ReadArgs, RelatedArgs, SearchArgs,
};

pub const DESC_SEARCH: &str = "Full-text search over the knowledge base — natural-language keywords (morphology-aware, BM25-ranked), NOT a regex. Returns ranked hits, one per line as `[#n] path · label · snippet`. Open a hit with `read(path, n)` using that `[#n]` number. Scope with optional glob/file_type filters; for an exact token or code use `grep` instead. Hits are ranked best-first — the top few usually contain the answer, so read those rather than running many searches.";

pub const DESC_READ: &str = "Read material by reference. Usually a document chunk: pass the `path` and chunk number `n` (the `[#n]` from a search/grep result; for PDFs the page). It returns the chunk's WHOLE text — for a large chunk that is a lot, and a table in its middle is easy to under-read; when you only need a value or its table, `grep` that value with `context` and read just the window instead. If a PDF table page is hard to read as text, call read again with `page_image: true` to return a 200 DPI JPEG instead (requires the server started with --vision). Returns the full text plus prev/next chunk numbers; if `n` is out of range the reply states the valid range. You may ALSO pass a graph NODE id (e.g. a Resolution id from a `glossary` line) as `path` — then it returns that node plus every evidence chunk it and its 1-hop chain MENTION, each labelled with where it came from.";

pub const DESC_GLOSSARY: &str = "Resolve a concept (a symptom, error, component or task in a few words) to graph nodes. A reasoning node prints its `id [type] label` followed by its full chain — cause → resolution — each with a `read path #n` anchor, so ONE call gives you the likely fix. The line may also show `· comm N · pr …` — the problem cluster id. After a hit, call `related(<that node id>)` to list alternate and related cases before searching again. Structural Section/Document nodes show their `path #n` anchor. Empty result = nothing matches yet. Morphology-aware over labels/aliases. Also call it before creating a node, to REUSE an existing one.";

pub const DESC_RELATED: &str = "Broaden a `glossary` hit — list OTHER solved cases linked to the same node. Call AFTER `glossary` when the cause→resolution chain is close but not quite right, you want alternates, or before running another search. Pass the reasoning-node `node` id copied from the glossary line (the token before `[Symptom]`/`[Cause]`/`[Resolution]`, e.g. `sym:...`), or a chunk `path` + `n`. Each line is prefixed and has a `read path #n` anchor: `SIMILAR` — paraphrase cases that share evidence; `COMMUNITY` — other nodes in the same problem cluster (same `comm N` as the glossary suffix), top by centrality. Empty → try another glossary term or fall back to search/grep. For the node's OWN chain, use `glossary` — not related.";

pub const DESC_NEIGHBORS: &str = "List a node's DIRECT structural edges (its actual typed relationships — e.g. what it REFERENCES, what CONSTRAINS it), one hop, each with the real edge direction (-> outgoing, <- incoming) and a `read path #n` anchor. Pass a `node` id (from `glossary`) or a chunk `path`+`n`. Filter with `edge_types` (relation names) and `direction` (out/in/both). This is FACTUAL graph structure — for fuzzy 'similar cases' use `related`; for how two nodes connect (possibly across documents) use `reach`. Empty => no such edges.";

pub const DESC_REACH: &str = "Cross-document reasoning bridge — the ONE traversal tool, two directions. Omit `to` for DISCOVERY: walk `relation` forward from `from`, crossing document boundaries on shared mentions (the bridge, on by default), and return every node reached as a candidate answer — use this to resolve a relational multi-hop instead of inferring it from prose. Pass `to` for VERIFY: does a grounded path from `from` to that specific candidate exist (a self-check on an answer you already produced)? `relation` fuzzy-matches an ontology edge type (omit = all chaining relations, undirected). Each hop prints its real edge direction (--REL--> / <--REL--) with a `read path #n` anchor, or `↝ bridged on \"<term>\"` where the reasoning crossed a document — never a silent jump. Give `from`/`to` as node ids (from `glossary`) or as `from_path`+`from_n` / `to_path`+`to_n` chunk refs. `max_depth` defaults to 6 (max 12); `bridge` defaults to true (false = graph-only, in-document connectivity — this reproduces the old `path` tool). For a node's own direct edges use `neighbors`.";

pub const DESC_GREP: &str = "Find an exact string in the text — a code, identifier, parameter name, or a value (e.g. `maxTsdr`, `M6`, `250`). ripgrep regex supported; smart-case. Use it whenever you know a precise token to locate (beats keyword `search`; for fuzzy/conceptual lookup use `search`). TO READ A TABLE, grep one of its values with `context` set to ~20-40: the reply then carries that many lines around each hit — a focused window onto the table — so you get the whole column in one call without reading the entire chunk. Returns matching lines as `path:#n: line`; a context line uses `-` instead of `:`. Reach for `read(path, n)` only when you actually need a whole chunk, not to locate a value. Other flags mirror ripgrep: -i/-F/-w, -o only-matching, -n line-number, -c count, -m max-count, -U multiline.";

pub const DESC_GLOB: &str = "List knowledge-base documents whose path matches a ripgrep `-g` glob (e.g. `*` or `**/*` for all documents, or `*<name-fragment>*` to find a file by name). Returns one `path  (N chunks)` per line — use it to discover what documents exist or find a file by name, then `read(path, n)` or scope a `search`/`grep` to it. N is the document's last page/section number; every page 1..N is addressable (blank pages return empty text).";

/// A single agent tool declaration: name, model-facing description, JSON-Schema for its
/// arguments (OpenAI-function core shape), and whether it requires the reasoning graph.
pub struct ToolDescriptor {
    pub name: &'static str,
    pub description: &'static str,
    pub params_schema: serde_json::Value,
    pub graph_gated: bool,
}

/// Normalize a `schemars::schema_for!` result to the OpenAI-function core schema:
/// `{ "type": "object", "properties": {…}, "required": […] }` — strips the schemars
/// `$schema`/`title`/`$defs` root-schema wrapper that tool-calling APIs don't expect.
fn normalize_schema(v: serde_json::Value) -> serde_json::Value {
    let properties = v
        .get("properties")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    let required = v
        .get("required")
        .cloned()
        .unwrap_or_else(|| serde_json::json!([]));
    serde_json::json!({
        "type": "object",
        "properties": properties,
        "required": required,
    })
}

fn schema_of<T: schemars::JsonSchema>() -> serde_json::Value {
    let schema = schemars::schema_for!(T);
    normalize_schema(serde_json::to_value(schema).expect("schema serializes to JSON"))
}

/// The canonical set of agent tool descriptors — retrieval tools first (ungated), then
/// the graph tools (gated on a reasoning graph existing for the corpus).
pub fn registry() -> Vec<ToolDescriptor> {
    vec![
        ToolDescriptor {
            name: "search",
            description: DESC_SEARCH,
            params_schema: schema_of::<SearchArgs>(),
            graph_gated: false,
        },
        ToolDescriptor {
            name: "read",
            description: DESC_READ,
            params_schema: schema_of::<ReadArgs>(),
            graph_gated: false,
        },
        ToolDescriptor {
            name: "grep",
            description: DESC_GREP,
            params_schema: schema_of::<GrepArgs>(),
            graph_gated: false,
        },
        ToolDescriptor {
            name: "glob",
            description: DESC_GLOB,
            params_schema: schema_of::<GlobArgs>(),
            graph_gated: false,
        },
        ToolDescriptor {
            name: "glossary",
            description: DESC_GLOSSARY,
            params_schema: schema_of::<GlossaryArgs>(),
            graph_gated: true,
        },
        ToolDescriptor {
            name: "related",
            description: DESC_RELATED,
            params_schema: schema_of::<RelatedArgs>(),
            graph_gated: true,
        },
        ToolDescriptor {
            name: "neighbors",
            description: DESC_NEIGHBORS,
            params_schema: schema_of::<NeighborsArgs>(),
            graph_gated: true,
        },
        ToolDescriptor {
            name: "reach",
            description: DESC_REACH,
            params_schema: schema_of::<ReachArgs>(),
            graph_gated: true,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_lists_core_tools_with_schemas() {
        let r = registry();
        let names: Vec<_> = r.iter().map(|d| d.name).collect();
        for t in [
            "search",
            "read",
            "grep",
            "glob",
            "glossary",
            "related",
            "neighbors",
            "reach",
        ] {
            assert!(names.contains(&t), "registry missing {t}");
        }
        // graph tools gated; retrieval tools not
        let g = |n| r.iter().find(|d| d.name == n).unwrap();
        assert!(g("glossary").graph_gated && g("related").graph_gated);
        assert!(!g("search").graph_gated && !g("read").graph_gated);
        // schema is a valid object with properties for a known arg
        let s = &g("search").params_schema;
        assert_eq!(s["type"], "object");
        assert!(s["properties"]["query"].is_object(), "search.query schema present");
    }
}
