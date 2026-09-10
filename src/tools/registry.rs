//! Single source of truth for agent tool DECLARATIONS (name, description, JSON schema,
//! gate metadata). MCP (src/mcp.rs) and the eval OpenAI/Anthropic/Responses/TZ surfaces
//! build their tool listings from `catalog()`/`resolve_tools` instead of hand-duplicating
//! them, so the surfaces cannot drift apart. Descriptions are extracted here verbatim from the
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

/// A single resolved agent-facing tool: name, description, and its CORE schema after
/// per-context shaping (e.g. image fields dropped when `no_image` is set).
pub struct ResolvedTool {
    pub name: &'static str,
    pub desc: &'static str,
    pub core_schema: serde_json::Value,
}

fn gate_ok(g: &Gate, ctx: &ToolContext) -> bool {
    match g {
        Gate::Graph => ctx.graph_on,
        Gate::Verify => ctx.verify_available,
        Gate::SourceFile => !ctx.no_source_file,
        Gate::Feature(f) => ctx.features.has(f),
    }
}

fn is_available(m: &ToolMeta, ctx: &ToolContext) -> bool {
    ctx.profile.allows(m.tier) && m.gates.iter().all(|g| gate_ok(g, ctx))
}

fn shape_active(flag: ShapeFlag, ctx: &ToolContext) -> bool {
    match flag {
        ShapeFlag::NoImage => ctx.no_image,
    }
}

fn apply_shape(schema: &mut serde_json::Value, shape: &Shape) {
    match shape {
        Shape::DropProps(props) => {
            if let Some(obj) = schema.get_mut("properties").and_then(|p| p.as_object_mut()) {
                for p in *props {
                    obj.remove(*p);
                }
            }
            if let Some(req) = schema.get_mut("required").and_then(|r| r.as_array_mut()) {
                req.retain(|v| v.as_str().map(|s| !props.contains(&s)).unwrap_or(true));
            }
        }
    }
}

/// Available Reader-tier (schema-bearing) tools, with per-context schema shaping applied.
pub fn resolve_tools(ctx: &ToolContext) -> Vec<ResolvedTool> {
    catalog()
        .into_iter()
        .filter(|m| is_available(m, ctx) && m.schema.is_some())
        .map(|m| {
            let mut schema = m.schema.clone().unwrap();
            for (flag, shape) in m.shapes {
                if shape_active(*flag, ctx) {
                    apply_shape(&mut schema, shape);
                }
            }
            ResolvedTool {
                name: m.name,
                desc: m.desc.unwrap(),
                core_schema: schema,
            }
        })
        .collect()
}

/// All available tool NAMES for the context (every tier) — the MCP router's keep-set.
pub fn available_names(ctx: &ToolContext) -> std::collections::HashSet<&'static str> {
    catalog()
        .into_iter()
        .filter(|m| is_available(m, ctx))
        .map(|m| m.name)
        .collect()
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

/// Test-only accessor: normalize a live rmcp route `input_schema` the same way `schema_of`
/// normalizes a `schemars` schema, so the MCP parity test can compare a route's advertised
/// schema against a catalog `core_schema` on equal footing.
pub fn normalize_for_test(v: &serde_json::Value) -> serde_json::Value {
    normalize_schema(v.clone())
}

fn schema_of<T: schemars::JsonSchema>() -> serde_json::Value {
    let schema = schemars::schema_for!(T);
    normalize_schema(serde_json::to_value(schema).expect("schema serializes to JSON"))
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn reader_ctx() -> ToolContext {
        ToolContext {
            profile: Tier::Reader,
            graph_on: true,
            verify_available: true,
            no_source_file: false,
            no_image: false,
            features: FeatureSet::default(),
        }
    }

    #[test]
    fn resolve_reader_full_gates_open() {
        use std::collections::BTreeSet;
        let names: BTreeSet<&str> = resolve_tools(&reader_ctx()).iter().map(|t| t.name).collect();
        let expected: BTreeSet<&str> = [
            "search", "read", "grep", "glob", "glossary", "reach", "sql", "verify",
            "get_source_file", "get_ontology",
        ]
        .into_iter()
        .collect();
        assert_eq!(names, expected);
    }

    #[test]
    fn verify_hidden_when_unavailable() {
        let mut ctx = reader_ctx();
        ctx.verify_available = false;
        assert!(!resolve_tools(&ctx).iter().any(|t| t.name == "verify"));
        assert!(!available_names(&ctx).contains("verify"));
    }

    #[test]
    fn sql_hidden_when_no_graph() {
        let mut ctx = reader_ctx();
        ctx.graph_on = false;
        let names: Vec<&str> = resolve_tools(&ctx).iter().map(|t| t.name).collect();
        assert!(!names.contains(&"sql"), "D1: sql is graph-gated");
        assert!(!names.contains(&"glossary") && !names.contains(&"reach"));
        assert!(names.contains(&"search") && names.contains(&"get_source_file"));
    }

    #[test]
    fn no_image_strips_read_page_fields() {
        let mut ctx = reader_ctx();
        ctx.no_image = true;
        let read = resolve_tools(&ctx).into_iter().find(|t| t.name == "read").unwrap();
        let props = read.core_schema.get("properties").unwrap().as_object().unwrap();
        assert!(!props.contains_key("page_image") && !props.contains_key("include_images"));
        // and default (no_image=false) keeps them
        let keep = resolve_tools(&reader_ctx()).into_iter().find(|t| t.name == "read").unwrap();
        let kprops = keep.core_schema.get("properties").unwrap().as_object().unwrap();
        assert!(kprops.contains_key("page_image"));
    }

    #[test]
    fn editor_profile_adds_editor_tools_in_available_names() {
        let mut ctx = reader_ctx();
        ctx.profile = Tier::Editor;
        let a = available_names(&ctx);
        assert!(a.contains("graph_upsert") && a.contains("resolve"));
        assert!(!a.contains("purge"), "purge is Full-tier");
        let r = available_names(&reader_ctx());
        assert!(!r.contains("graph_upsert"), "Reader profile excludes editor tools");
    }

    #[test]
    fn source_file_gate_and_feature_gate() {
        let mut ctx = reader_ctx();
        ctx.no_source_file = true;
        assert!(!available_names(&ctx).contains("get_source_file"));
        let mut ctx2 = reader_ctx();
        ctx2.profile = Tier::Editor; // notebook off by default
        assert!(!available_names(&ctx2).contains("note"));
        ctx2.features.notebook = true;
        assert!(available_names(&ctx2).contains("note"));
    }
}
