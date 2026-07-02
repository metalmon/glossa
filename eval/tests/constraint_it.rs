use std::path::Path;

use glossa::graph::store::{Edge, GraphStore, Node, Provenance};

const ONTOLOGY_TOML: &str = include_str!("../ontology-constraint.toml");

fn setup_corpus(dir: &Path) -> GraphStore {
    let glossa_dir = dir.join(".glossa");
    std::fs::create_dir_all(&glossa_dir).unwrap();
    std::fs::write(glossa_dir.join("ontology.toml"), ONTOLOGY_TOML).unwrap();
    GraphStore::open(dir).unwrap()
}

fn mkprov(src: &str) -> Provenance {
    Provenance {
        source_path: src.into(),
        range: None,
        file_sig: None,
        origin: "agent".into(),
        confidence: 1.0,
        created_at: 0,
    }
}

fn insert_node(g: &GraphStore, id: &str, node_type: &str, label: &str, src: &str) {
    g.put_node(&Node {
        id: id.into(),
        node_type: node_type.into(),
        label: label.into(),
        aliases: vec![],
        prov: mkprov(src),
    })
    .unwrap();
}

fn insert_edge(g: &GraphStore, from: &str, edge_type: &str, to: &str, src: &str) {
    g.put_edge(&Edge {
        from: from.into(),
        edge_type: edge_type.into(),
        to: to.into(),
        prov: mkprov(src),
    })
    .unwrap();
}

const GOST_SRC: &str = "gost-31369-2023.pdf";

fn build_gost_graph(g: &GraphStore) {
    insert_node(g, "fld:diam", "Field", "Диаметр", GOST_SRC);
    insert_node(g, "lit:dmin", "Literal", "10", GOST_SRC);
    insert_node(g, "lit:dmax", "Literal", "200", GOST_SRC);
    insert_node(g, "rng:diam", "Range", "Диаметр range", GOST_SRC);
    insert_edge(g, "fld:diam", "CONSTRAINED_BY", "rng:diam", GOST_SRC);
    insert_edge(g, "rng:diam", "HAS_MIN", "lit:dmin", GOST_SRC);
    insert_edge(g, "rng:diam", "HAS_MAX", "lit:dmax", GOST_SRC);

    insert_node(g, "fld:type", "Field", "Тип", GOST_SRC);
    insert_node(g, "dom:types", "Domain", "Valid types", GOST_SRC);
    insert_node(g, "lit:ta", "Literal", "A", GOST_SRC);
    insert_node(g, "lit:tb", "Literal", "B", GOST_SRC);
    insert_node(g, "lit:tc", "Literal", "C", GOST_SRC);
    insert_node(g, "enum:type", "Enum", "Тип enum", GOST_SRC);
    insert_edge(g, "fld:type", "CONSTRAINED_BY", "enum:type", GOST_SRC);
    insert_edge(g, "enum:type", "HAS_DOMAIN", "dom:types", GOST_SRC);
    insert_edge(g, "dom:types", "HAS_LITERAL", "lit:ta", GOST_SRC);
    insert_edge(g, "dom:types", "HAS_LITERAL", "lit:tb", GOST_SRC);
    insert_edge(g, "dom:types", "HAS_LITERAL", "lit:tc", GOST_SRC);

    insert_node(g, "fld:code", "Field", "Код", GOST_SRC);
    insert_node(g, "lit:pat", "Literal", "^[0-9]{4}-[A-Z]+$", GOST_SRC);
    insert_node(g, "rx:code", "Regex", "Код regex", GOST_SRC);
    insert_edge(g, "fld:code", "CONSTRAINED_BY", "rx:code", GOST_SRC);
    insert_edge(g, "rx:code", "HAS_PATTERN", "lit:pat", GOST_SRC);

    insert_node(g, "fld:prot", "Field", "Защита", GOST_SRC);
    insert_node(g, "req:prot", "Required", "Защита required", GOST_SRC);
    insert_edge(g, "fld:prot", "CONSTRAINED_BY", "req:prot", GOST_SRC);
}

fn solve(
    dir: &Path,
    g: &GraphStore,
    source_path: &str,
    mode: &str,
    assignment: Vec<(&str, serde_json::Value)>,
) -> serde_json::Value {
    let ont = glossa::graph::ontology::Ontology::load_or_default(dir);
    let problem = glossa::constraint_adapter::load_problem(g, &ont, source_path).unwrap();

    let solve_mode = match mode {
        "validate" => glossa_constraint::SolveMode::Validate,
        "infer" => glossa_constraint::SolveMode::Infer,
        "check" => glossa_constraint::SolveMode::Check,
        _ => panic!("unknown mode"),
    };

    let assignment: Vec<(String, serde_json::Value)> = assignment
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();

    let result = glossa_constraint::solver::solve(&problem, solve_mode, &assignment);
    serde_json::to_value(&result).unwrap()
}

#[test]
fn validate_passes_valid_assignments() {
    let dir = tempfile::tempdir().unwrap();
    let g = setup_corpus(dir.path());
    build_gost_graph(&g);

    let result = solve(
        dir.path(), &g, GOST_SRC, "validate",
        vec![
            ("Диаметр", serde_json::json!(150.0)),
            ("Тип", serde_json::json!("A")),
            ("Код", serde_json::json!("1234-ABCD")),
            ("Защита", serde_json::json!("есть")),
        ],
    );

    let violations = result.get("violations").and_then(|v| v.as_array());
    assert!(violations.map_or(true, |v| v.is_empty()), "expected no violations, got: {violations:?}");
}

#[test]
fn validate_rejects_out_of_range() {
    let dir = tempfile::tempdir().unwrap();
    let g = setup_corpus(dir.path());
    build_gost_graph(&g);

    let result = solve(dir.path(), &g, GOST_SRC, "validate",
        vec![("Диаметр", serde_json::json!(999.0))]);

    let violations = result["violations"].as_array().unwrap();
    assert!(!violations.is_empty(), "expected violations for out-of-range value");
    let msg = serde_json::to_string(&result).unwrap();
    assert!(msg.contains("Диаметр"), "violation should mention field name: {msg}");
}

#[test]
fn validate_rejects_invalid_enum() {
    let dir = tempfile::tempdir().unwrap();
    let g = setup_corpus(dir.path());
    build_gost_graph(&g);

    let result = solve(dir.path(), &g, GOST_SRC, "validate",
        vec![("Тип", serde_json::json!("X"))]);

    let violations = result["violations"].as_array().unwrap();
    assert!(!violations.is_empty(), "expected violations for invalid enum value");
}

#[test]
fn validate_rejects_bad_regex() {
    let dir = tempfile::tempdir().unwrap();
    let g = setup_corpus(dir.path());
    build_gost_graph(&g);

    let result = solve(dir.path(), &g, GOST_SRC, "validate",
        vec![("Код", serde_json::json!("bad"))]);

    let violations = result["violations"].as_array().unwrap();
    assert!(!violations.is_empty(), "expected violations for regex mismatch");
}

#[test]
fn infer_returns_domains() {
    let dir = tempfile::tempdir().unwrap();
    let g = setup_corpus(dir.path());
    build_gost_graph(&g);

    let result = solve(dir.path(), &g, GOST_SRC, "infer", vec![]);

    let domains = result["domains"].as_array().unwrap();
    let map: std::collections::HashMap<&str, &serde_json::Value> = domains
        .iter()
        .map(|d| (d["field"].as_str().unwrap(), d))
        .collect();
    assert!(map.contains_key("Диаметр"));
    assert!(map.contains_key("Тип"));
    assert!(map.contains_key("Код"));
    assert!(map.contains_key("Защита"));

    let diam = &map["Диаметр"]["domain"]["value"];
    assert_eq!(map["Диаметр"]["domain"]["type"], "interval");
    assert_eq!(diam["min"], 10.0);
    assert_eq!(diam["max"], 200.0);

    let typ = &map["Тип"]["domain"]["value"];
    assert_eq!(map["Тип"]["domain"]["type"], "set");
    let vals = typ["values"].as_array().unwrap();
    let strs: Vec<&str> = vals.iter().map(|v| v.as_str().unwrap()).collect();
    assert_eq!(strs, vec!["A", "B", "C"]);
}

#[test]
fn check_reports_consistency() {
    let dir = tempfile::tempdir().unwrap();
    let g = setup_corpus(dir.path());
    build_gost_graph(&g);

    let result = solve(dir.path(), &g, GOST_SRC, "check", vec![]);

    let issues = result.get("issues").and_then(|v| v.as_array());
    assert!(issues.map_or(true, |v| v.is_empty()), "expected no issues for valid graph: {issues:?}");
}

#[test]
fn check_detects_empty_enum() {
    let dir = tempfile::tempdir().unwrap();
    let g = setup_corpus(dir.path());

    insert_node(&g, "fld:empty", "Field", "Пусто", GOST_SRC);
    insert_node(&g, "dom:empty", "Domain", "Empty domain", GOST_SRC);
    insert_node(&g, "enum:empty", "Enum", "Empty enum", GOST_SRC);
    insert_edge(&g, "fld:empty", "CONSTRAINED_BY", "enum:empty", GOST_SRC);
    insert_edge(&g, "enum:empty", "HAS_DOMAIN", "dom:empty", GOST_SRC);

    let result = solve(dir.path(), &g, GOST_SRC, "check", vec![]);
    let issues = result["issues"].as_array().unwrap();
    assert!(!issues.is_empty(), "expected issues for empty enum domain");
}

#[test]
fn validate_single_field() {
    let dir = tempfile::tempdir().unwrap();
    let g = setup_corpus(dir.path());
    build_gost_graph(&g);

    let result = solve(dir.path(), &g, GOST_SRC, "validate",
        vec![("Диаметр", serde_json::json!(50.0)), ("Защита", serde_json::json!("есть"))]);
    let violations = result.get("violations").and_then(|v| v.as_array());
    assert!(violations.map_or(true, |v| v.is_empty()), "50 ∈ [10,200] should pass: {violations:?}");
}

#[test]
fn validate_missing_required_field() {
    let dir = tempfile::tempdir().unwrap();
    let g = setup_corpus(dir.path());
    build_gost_graph(&g);

    let result = solve(dir.path(), &g, GOST_SRC, "validate",
        vec![("Диаметр", serde_json::json!(50.0))]);
    let violations = result["violations"].as_array().unwrap();
    assert!(!violations.is_empty(), "Защита is Required but missing");
}

#[test]
fn gost_7805_bolt_range_and_enum() {
    let dir = tempfile::tempdir().unwrap();
    let g = setup_corpus(dir.path());

    insert_node(&g, "fld:diam", "Field", "d_диаметр", GOST_SRC);
    insert_node(&g, "lit:dmin", "Literal", "2", GOST_SRC);
    insert_node(&g, "lit:dmax", "Literal", "36", GOST_SRC);
    insert_node(&g, "rng:diam", "Range", "диаметр резьбы", GOST_SRC);
    insert_edge(&g, "fld:diam", "CONSTRAINED_BY", "rng:diam", GOST_SRC);
    insert_edge(&g, "rng:diam", "HAS_MIN", "lit:dmin", GOST_SRC);
    insert_edge(&g, "rng:diam", "HAS_MAX", "lit:dmax", GOST_SRC);

    // Valid bolt diameters per GOST 7805-70
    insert_node(&g, "dom:diam_vals", "Domain", "valid diameters", GOST_SRC);
    insert_node(&g, "enum:diam_vals", "Enum", "диаметр enum", GOST_SRC);
    insert_edge(&g, "fld:diam", "CONSTRAINED_BY", "enum:diam_vals", GOST_SRC);
    insert_edge(&g, "enum:diam_vals", "HAS_DOMAIN", "dom:diam_vals", GOST_SRC);
    for d in ["2", "2.5", "3", "4", "5", "6", "8", "10", "12", "14", "16", "18", "20", "24", "27", "30", "36"] {
        let id = format!("lit:d{d}");
        insert_node(&g, &id, "Literal", d, GOST_SRC);
        insert_edge(&g, "dom:diam_vals", "HAS_LITERAL", &id, GOST_SRC);
    }

    // Valid in-range: d=12
    let result = solve(dir.path(), &g, GOST_SRC, "validate",
        vec![("d_диаметр", serde_json::json!(12.0))]);
    let violations = result.get("violations").and_then(|v| v.as_array());
    assert!(violations.map_or(true, |v| v.is_empty()),
        "d=12 should be valid, got: {violations:?}");

    // Out of range: d=999
    let result = solve(dir.path(), &g, GOST_SRC, "validate",
        vec![("d_диаметр", serde_json::json!(999.0))]);
    let violations = result.get("violations").and_then(|v| v.as_array());
    assert!(violations.map_or(false, |v| !v.is_empty()),
        "d=999 should violate Range");

    // Not in enum: d=7 (valid range but not a standard bolt diameter)
    let result = solve(dir.path(), &g, GOST_SRC, "validate",
        vec![("d_диаметр", serde_json::json!(7.0))]);
    let violations = result.get("violations").and_then(|v| v.as_array());
    assert!(violations.map_or(false, |v| !v.is_empty()),
        "d=7 is not a standard bolt diameter");
}

#[test]
fn gost_7805_tolerance_enum() {
    let dir = tempfile::tempdir().unwrap();
    let g = setup_corpus(dir.path());

    insert_node(&g, "fld:tol", "Field", "допуск", GOST_SRC);
    insert_node(&g, "dom:tols", "Domain", "tolerance classes", GOST_SRC);
    insert_node(&g, "enum:tol", "Enum", "допуск enum", GOST_SRC);
    insert_edge(&g, "fld:tol", "CONSTRAINED_BY", "enum:tol", GOST_SRC);
    insert_edge(&g, "enum:tol", "HAS_DOMAIN", "dom:tols", GOST_SRC);
    for label in ["6e", "6g", "6h", "8g"] {
        let id = format!("lit:{}", label);
        insert_node(&g, &id, "Literal", label, GOST_SRC);
        insert_edge(&g, "dom:tols", "HAS_LITERAL", &id, GOST_SRC);
    }

    // Valid tolerance
    let result = solve(dir.path(), &g, GOST_SRC, "validate",
        vec![("допуск", serde_json::json!("6g"))]);
    let violations = result.get("violations").and_then(|v| v.as_array());
    assert!(violations.map_or(true, |v| v.is_empty()),
        "6g is a valid tolerance");

    // Invalid tolerance
    let result = solve(dir.path(), &g, GOST_SRC, "validate",
        vec![("допуск", serde_json::json!("7h"))]);
    let violations = result.get("violations").and_then(|v| v.as_array());
    assert!(violations.map_or(false, |v| !v.is_empty()),
        "7h is not a standard tolerance");
}

#[test]
fn ontology_file_is_parsable() {
    let ont = glossa::graph::ontology::Ontology::parse(ONTOLOGY_TOML).unwrap();

    let ctypes = ont.constraint_types();
    assert!(ctypes.contains_key("Range"));
    assert!(ctypes.contains_key("Regex"));
    assert!(ctypes.contains_key("Enum"));
    assert!(ctypes.contains_key("Formula"));
    assert!(ctypes.contains_key("Required"));
    assert!(ctypes.contains_key("Forbidden"));
    assert!(ctypes.contains_key("Conditional"));
    assert_eq!(ctypes.len(), 7);

    let ok: Vec<String> = vec!["Field".into(), "Literal".into(), "Domain".into()];
    for name in &ok {
        assert!(ont.entity_types().contains(name.as_str()), "missing entity {name}");
    }

    let rels = ont.raw_relations();
    assert!(rels.contains_key("CONSTRAINED_BY"));
    assert!(rels.contains_key("HAS_MIN"));
    assert!(rels.contains_key("HAS_MAX"));
    assert!(rels.contains_key("HAS_PATTERN"));
    assert!(rels.contains_key("HAS_DOMAIN"));
    assert!(rels.contains_key("HAS_LITERAL"));
    assert!(rels.contains_key("HAS_EXPRESSION"));

    assert!(ont.strict(), "constraint ontology should be strict");
}

/// The agent path end-to-end: label-based graph_upsert through the shared ops
/// layer with the constraint ontology. Constraint-type nodes (Enum) and
/// label-resolved edges must all land in one batch — this is exactly what the
/// eval agent sends.
#[test]
fn agent_style_upsert_builds_enum_constraint() {
    let dir = tempfile::tempdir().unwrap();
    let g = setup_corpus(dir.path());
    let idx = glossa::index::store::DocIndex::open_or_create(dir.path()).unwrap();
    idx.write_chunks(&[glossa::model::Chunk {
        doc_path: "gost.docx".into(),
        location: "S1".into(),
        file_type: "docx".into(),
        text: "Наружный диаметр: 125, 150".into(),
    }])
    .unwrap();
    let ont = glossa::graph::ontology::Ontology::parse(ONTOLOGY_TOML).unwrap();

    let unode = |t: &str, l: &str| glossa::graph::ops::UpsertNode {
        node_type: t.into(),
        label: l.into(),
        source_path: "gost.docx".into(),
        aliases: vec![],
    };
    let uedge = |f: &str, et: &str, to: &str| glossa::graph::ops::UpsertEdge {
        from: f.into(),
        edge_type: et.into(),
        to: to.into(),
        source_path: "gost.docx".into(),
    };

    let out = glossa::graph::ops::graph_upsert(
        &idx,
        &g,
        &ont,
        vec![
            unode("Field", "Наружный диаметр"),
            unode("Enum", "Наружный диаметр enum"),
            unode("Domain", "Наружный диаметр domain"),
            unode("Literal", "125"),
            unode("Literal", "150"),
        ],
        vec![
            uedge("Наружный диаметр", "CONSTRAINED_BY", "Наружный диаметр enum"),
            uedge("Наружный диаметр enum", "HAS_DOMAIN", "Наружный диаметр domain"),
            uedge("Наружный диаметр domain", "HAS_LITERAL", "125"),
            uedge("Наружный диаметр domain", "HAS_LITERAL", "150"),
        ],
        1,
    );
    assert!(!out.rejected, "{}", out.message);
    assert_eq!(g.all_nodes().unwrap().len(), 5, "{}", out.message);
    assert_eq!(g.all_edges().unwrap().len(), 4, "{}", out.message);
}
