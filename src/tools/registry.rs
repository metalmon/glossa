//! Single source of truth for agent tool DECLARATIONS (name, description, JSON schema,
//! graph-gated flag). MCP (src/mcp.rs), and later the OpenAI/TZ surfaces, build their
//! tool listings from `registry()` instead of hand-duplicating them, so the three
//! surfaces cannot drift apart. Descriptions are extracted here verbatim from the
//! current MCP `#[tool(description = …)]` attributes; schemas come from the SAME arg
//! structs `src/mcp.rs` already deserializes into (`schemars::schema_for!`), normalized
//! to the OpenAI-function core `{ "type": "object", "properties": {…}, "required": […] }`.

use crate::mcp::{
    Empty, GlobArgs, GlossaryArgs, GraphQueryArgs, GrepArgs, ReachArgs, ReadArgs, SearchArgs,
    SourceFileArgs, VerifyArgs,
};

pub const DESC_SEARCH: &str = "Full-text search over the knowledge base — natural-language keywords (morphology-aware, BM25-ranked), NOT a regex. Returns ranked hits, one per line as `path#n · label · snippet`. Open a hit with `read(path#n)` — copy that leading token exactly as shown; the same token is what a node's `source_path` takes to ground it. Scope with optional glob/file_type filters; for an exact token or code use `grep` instead. Hits are ranked best-first — the top few usually contain the answer, so read those rather than running many searches.";

pub const DESC_READ: &str = "Read material by reference. Usually a document chunk: pass the copy-ready `path#n` token exactly as a search/grep/read result showed it (or `path` plus chunk number `n` separately; for PDFs `n` is the page). It returns the chunk's WHOLE text — for a large chunk that is a lot, and a table in its middle is easy to under-read; when you only need a value or its table, `grep` that value with `context` and read just the window instead. If a PDF table page is hard to read as text, call read again with `page_image: true` to return a 200 DPI JPEG instead (requires the server started with --vision). Returns the full text plus prev/next chunk references, also `path#n`; if `n` is out of range the reply states the valid range. That same `path#n` token is what a node's `source_path` takes to ground it — an ungrounded query-side node should OMIT `source_path` entirely. You may ALSO pass a graph NODE id (e.g. a Resolution id from a `glossary` line) as `path` — then it returns that node plus every evidence chunk it and its 1-hop chain MENTION, each labelled with where it came from.";

pub const DESC_GLOSSARY: &str = "Resolve a concept (a symptom, error, component or task in a few words) to graph nodes. A reasoning node prints its `id [type] label` followed by its full chain — cause → resolution — each with a `read(path#n)` anchor, so ONE call gives you the likely fix. The line may also show `· comm N · pr …` — the problem cluster id. After a hit, call `related(<that node id>)` to list alternate and related cases before searching again. Structural Section/Document nodes show their `path#n` anchor — that same token grounds a node's `source_path` (omit `source_path` entirely for an ungrounded query-side node). Empty result = nothing matches yet. Morphology-aware over labels/aliases. Also call it before creating a node, to REUSE an existing one.";

pub const DESC_REACH: &str = "Cross-document reasoning bridge — the ONE traversal tool, two directions. Omit `to` for DISCOVERY: walk `relation` forward from `from`, crossing document boundaries on shared mentions (the bridge, on by default), and return every node reached as a candidate answer — use this to resolve a relational multi-hop instead of inferring it from prose. Pass `to` for VERIFY: does a grounded path from `from` to that specific candidate exist (a self-check on an answer you already produced)? `relation` fuzzy-matches an ontology edge type (omit = all chaining relations, undirected). Each hop prints its real edge direction (--REL--> / <--REL--) with a `read(path#n)` anchor, or `↝ bridged on \"<term>\"` where the reasoning crossed a document — never a silent jump; that same `path#n` token grounds a node's `source_path` (omit it for an ungrounded query-side node). Give `from`/`to` as node ids (from `glossary`) or as `from_path`+`from_n` / `to_path`+`to_n` chunk refs. `max_depth` defaults to 6 (max 12); `bridge` defaults to true (false = graph-only, in-document connectivity — this reproduces the old `path` tool). For a node's own direct edges use `neighbors`.";

pub const DESC_GREP: &str = "Find an exact string in the text — a code, identifier, parameter name, or a value (e.g. `maxTsdr`, `M6`, `250`). ripgrep regex supported; smart-case. Use it whenever you know a precise token to locate (beats keyword `search`; for fuzzy/conceptual lookup use `search`). TO READ A TABLE, grep one of its values with `context` set to ~20-40: the reply then carries that many lines around each hit — a focused window onto the table — so you get the whole column in one call without reading the entire chunk. Returns matching lines as `path#n: line`; a context line uses `-` instead of `:`. Reach for `read(path#n)` only when you actually need a whole chunk, not to locate a value; that same `path#n` token is what a node's `source_path` takes to ground it. Other flags mirror ripgrep: -i/-F/-w, -o only-matching, -n line-number, -c count, -m max-count, -U multiline.";

pub const DESC_GLOB: &str = "List knowledge-base documents whose path matches a ripgrep `-g` glob (e.g. `*` or `**/*` for all documents, or `*<name-fragment>*` to find a file by name). Returns one `path  (N chunks)` per line — use it to discover what documents exist or find a file by name, then `read(path, n)` or scope a `search`/`grep` to it. N is the document's last page/section number; every page 1..N is addressable (blank pages return empty text).";

pub const DESC_VERIFY: &str = "Check whether an answer is grounded in the cited chunks; returns serve/abstain. Pass the final answer and the chunk paths it rests on.";

pub const DESC_SQL: &str = "Run a read-only SQL SELECT over the reasoning graph to compute/aggregate/rank/filter/traverse-by-join over facts and edges; an empty query returns the schema. Tables: nodes(id, node_type, label), edges(efrom, edge_type, eto), node_validity(node_id, valid_from, ...), edges_labeled(src_label, edge_type, dst_label, efrom, eto). This is SQLite (read-only SELECT). LIKE is case-insensitive incl. Cyrillic; ILIKE is accepted and treated as LIKE; no trailing ';' needed.";

pub const DESC_GET_SOURCE_FILE: &str = "Deliver the ORIGINAL source file behind a citation to the user for source attribution — NOT for reading its text (use `read` for content). Pass the document `path` from a search/grep result and, for a PDF, the cited page `n`. Returns the file as an embedded resource the client can preview or download, plus a one-line note of what was delivered. A large PDF is delivered as just the cited page (still a real, text-bearing PDF); an oversize non-PDF, or an oversize ref with no page, returns guidance to cite a specific PDF page. Read-only; available in every profile. A DOCX is delivered as PDF by default (source format renders inconsistently across clients); pass `raw: true` to get the original .docx.";

pub const DESC_GET_ONTOLOGY: &str = "Return the knowledge-base ontology as JSON: parameters, constraints, relations, and graph-building patterns. Call first to learn valid node/edge shapes before graph_upsert.";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tier {
    Reader,
    Editor,
    Full,
}

impl Tier {
    /// Does a server running `self` profile expose a tool declared at `tool_tier`?
    fn allows(self, tool_tier: Tier) -> bool {
        matches!(
            (self, tool_tier),
            (Tier::Full, _)
                | (Tier::Editor, Tier::Reader | Tier::Editor)
                | (Tier::Reader, Tier::Reader)
        )
    }
}

/// A gate a tool must pass for the active context. Gates AND together.
pub enum Gate {
    Graph,
    Verify,
    SourceFile,
    Feature(&'static str),
}

#[derive(Clone, Copy)]
pub enum ShapeFlag {
    NoImage,
}

/// Flag-driven CORE-schema shaping applied after a tool is available.
pub enum Shape {
    DropProps(&'static [&'static str]),
}

#[derive(Clone, Copy, Default)]
pub struct FeatureSet {
    pub notebook: bool,
    pub constraint: bool,
}
impl FeatureSet {
    fn has(&self, name: &str) -> bool {
        match name {
            "notebook" => self.notebook,
            "constraint" => self.constraint,
            _ => false,
        }
    }
}

/// The launch/corpus inputs that decide the advertised surface. Built identically
/// by the MCP server and by the eval run that represents a given deployment.
pub struct ToolContext {
    pub profile: Tier,
    pub graph_on: bool,
    pub verify_available: bool,
    pub no_source_file: bool,
    pub no_image: bool,
    pub features: FeatureSet,
}

pub struct ToolMeta {
    pub name: &'static str,
    pub tier: Tier,
    pub gates: &'static [Gate],
    pub shapes: &'static [(ShapeFlag, Shape)],
    /// Some for Reader-tier tools eval renders; None for MCP-only (schema in the macro).
    pub schema: Option<serde_json::Value>,
    pub desc: Option<&'static str>,
}

/// The single catalog of EVERY MCP route with its gate metadata. Replaces the
/// per-site constants in `src/mcp.rs` and the `graph_gated`/`verify_gated` bools here.
pub fn catalog() -> Vec<ToolMeta> {
    // helper to keep entries terse
    fn agent(
        name: &'static str,
        tier: Tier,
        gates: &'static [Gate],
        shapes: &'static [(ShapeFlag, Shape)],
        schema: serde_json::Value,
        desc: &'static str,
    ) -> ToolMeta {
        ToolMeta {
            name,
            tier,
            gates,
            shapes,
            schema: Some(schema),
            desc: Some(desc),
        }
    }
    fn mcp_only(name: &'static str, tier: Tier, gates: &'static [Gate]) -> ToolMeta {
        ToolMeta {
            name,
            tier,
            gates,
            shapes: &[],
            schema: None,
            desc: None,
        }
    }
    const READ_SHAPE: &[(ShapeFlag, Shape)] =
        &[(ShapeFlag::NoImage, Shape::DropProps(&["page_image", "include_images"]))];
    vec![
        // Reader-tier, agent-facing (eval renders these)
        agent("search", Tier::Reader, &[], &[], schema_of::<SearchArgs>(), DESC_SEARCH),
        agent("read", Tier::Reader, &[], READ_SHAPE, schema_of::<ReadArgs>(), DESC_READ),
        agent("grep", Tier::Reader, &[], &[], schema_of::<GrepArgs>(), DESC_GREP),
        agent("glob", Tier::Reader, &[], &[], schema_of::<GlobArgs>(), DESC_GLOB),
        agent("glossary", Tier::Reader, &[Gate::Graph], &[], schema_of::<GlossaryArgs>(), DESC_GLOSSARY),
        agent("reach", Tier::Reader, &[Gate::Graph], &[], schema_of::<ReachArgs>(), DESC_REACH),
        agent("sql", Tier::Reader, &[Gate::Graph], &[], schema_of::<GraphQueryArgs>(), DESC_SQL),
        agent("verify", Tier::Reader, &[Gate::Verify], &[], schema_of::<VerifyArgs>(), DESC_VERIFY),
        agent("get_source_file", Tier::Reader, &[Gate::SourceFile], &[], schema_of::<SourceFileArgs>(), DESC_GET_SOURCE_FILE),
        agent("get_ontology", Tier::Reader, &[], &[], schema_of::<Empty>(), DESC_GET_ONTOLOGY),
        // MCP-only Reader (notebook-read)
        mcp_only("ls", Tier::Reader, &[Gate::Feature("notebook")]),
        // MCP-only Editor
        mcp_only("note", Tier::Editor, &[Gate::Feature("notebook")]),
        mcp_only("del", Tier::Editor, &[Gate::Feature("notebook")]),
        mcp_only("index", Tier::Editor, &[Gate::Graph]),
        mcp_only("resolve", Tier::Editor, &[Gate::Graph]),
        mcp_only("neighbors", Tier::Editor, &[Gate::Graph]),
        mcp_only("related", Tier::Editor, &[Gate::Graph]),
        mcp_only("graph_upsert", Tier::Editor, &[Gate::Graph]),
        mcp_only("graph_delete", Tier::Editor, &[Gate::Graph]),
        mcp_only("graph_update", Tier::Editor, &[Gate::Graph]),
        mcp_only("graph_generalize", Tier::Editor, &[Gate::Graph]),
        mcp_only("graph_doctor", Tier::Editor, &[Gate::Graph]),
        mcp_only("graph_stats", Tier::Editor, &[]),
        mcp_only("constraint_solve", Tier::Editor, &[Gate::Feature("constraint")]),
        mcp_only("graph_build", Tier::Editor, &[Gate::Feature("constraint")]),
        // MCP-only Full
        mcp_only("purge", Tier::Full, &[Gate::Graph]),
    ]
}

/// A single agent tool declaration: name, model-facing description, JSON-Schema for its
/// arguments (OpenAI-function core shape), and whether it requires the reasoning graph.
pub struct ToolDescriptor {
    pub name: &'static str,
    pub description: &'static str,
    pub params_schema: serde_json::Value,
    pub graph_gated: bool,
    /// Withheld from the advertised tool schema when the answer-grounding gate is
    /// disabled/uncalibrated for the corpus (`glossa::gate::VerifyConfig`) — serving parity with
    /// the live MCP server's fail-closed advertisement. Set only on `verify`.
    pub verify_gated: bool,
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
            verify_gated: false,
        },
        ToolDescriptor {
            name: "read",
            description: DESC_READ,
            params_schema: schema_of::<ReadArgs>(),
            graph_gated: false,
            verify_gated: false,
        },
        ToolDescriptor {
            name: "grep",
            description: DESC_GREP,
            params_schema: schema_of::<GrepArgs>(),
            graph_gated: false,
            verify_gated: false,
        },
        ToolDescriptor {
            name: "glob",
            description: DESC_GLOB,
            params_schema: schema_of::<GlobArgs>(),
            graph_gated: false,
            verify_gated: false,
        },
        ToolDescriptor {
            name: "verify",
            description: DESC_VERIFY,
            params_schema: schema_of::<VerifyArgs>(),
            graph_gated: false,
            verify_gated: true,
        },
        ToolDescriptor {
            name: "glossary",
            description: DESC_GLOSSARY,
            params_schema: schema_of::<GlossaryArgs>(),
            graph_gated: true,
            verify_gated: false,
        },
        ToolDescriptor {
            name: "reach",
            description: DESC_REACH,
            params_schema: schema_of::<ReachArgs>(),
            graph_gated: true,
            verify_gated: false,
        },
        ToolDescriptor {
            name: "sql",
            description: DESC_SQL,
            params_schema: schema_of::<GraphQueryArgs>(),
            graph_gated: true,
            verify_gated: false,
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
        for t in ["search", "read", "grep", "glob", "verify", "glossary", "reach", "sql"] {
            assert!(names.contains(&t), "registry missing {t}");
        }
        // withheld from the Reader profile: measured clutter (related/neighbors)
        for t in ["related", "neighbors"] {
            assert!(!names.contains(&t), "registry must NOT contain {t}");
        }
        assert_eq!(
            names.len(),
            8,
            "registry must contain exactly the Reader profile's 8 tools"
        );
        // graph tools gated; retrieval + the model-free grounding gate are not
        let g = |n| r.iter().find(|d| d.name == n).unwrap();
        assert!(g("glossary").graph_gated && g("reach").graph_gated && g("sql").graph_gated);
        assert!(
            !g("search").graph_gated
                && !g("read").graph_gated
                && !g("grep").graph_gated
                && !g("glob").graph_gated
                && !g("verify").graph_gated
        );
        // verify alone is gated on the answer-grounding gate being enabled/calibrated
        assert!(g("verify").verify_gated, "verify must be verify_gated");
        for t in ["search", "read", "grep", "glob", "glossary", "reach", "sql"] {
            assert!(!g(t).verify_gated, "{t} must NOT be verify_gated");
        }
        // schema is a valid object with properties for a known arg
        let s = &g("search").params_schema;
        assert_eq!(s["type"], "object");
        assert!(
            s["properties"]["query"].is_object(),
            "search.query schema present"
        );
    }

    #[test]
    fn catalog_has_all_26_routes_with_expected_tiers() {
        use std::collections::BTreeSet;
        let names: BTreeSet<&str> = catalog().iter().map(|m| m.name).collect();
        let expected: BTreeSet<&str> = [
            "search","read","grep","glob","glossary","reach","sql","verify",
            "get_source_file","get_ontology","ls","note","del","index","resolve",
            "neighbors","related","graph_upsert","graph_delete","graph_update",
            "graph_generalize","graph_doctor","graph_stats","constraint_solve",
            "graph_build","purge",
        ].into_iter().collect();
        assert_eq!(names, expected, "catalog must list exactly the 26 MCP routes");

        let by = |n: &str| catalog().into_iter().find(|m| m.name == n).unwrap();
        // Reader-tier agent tools carry Some(schema)+Some(desc); MCP-only carry None.
        assert!(by("search").schema.is_some() && by("search").desc.is_some());
        assert!(by("get_ontology").schema.is_some(), "get_ontology is Reader-tier, eval renders it");
        assert!(by("purge").schema.is_none(), "purge is MCP-only Full-tier");
        assert!(matches!(by("purge").tier, Tier::Full));
        assert!(matches!(by("sql").tier, Tier::Reader));
    }
}
