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

fn insert_node_aliases(g: &GraphStore, id: &str, node_type: &str, label: &str, aliases: &[&str], src: &str) {
    g.put_node(&Node {
        id: id.into(),
        node_type: node_type.into(),
        label: label.into(),
        aliases: aliases.iter().map(|s| s.to_string()).collect(),
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
    insert_node_aliases(g, "enum:type", "Enum", "Тип enum", &["A", "B", "C"], GOST_SRC);
    insert_edge(g, "fld:type", "CONSTRAINED_BY", "enum:type", GOST_SRC);

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
    let problem = glossa::constraint_adapter::load_problem(g, &ont, Some(source_path)).unwrap();

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
    insert_node_aliases(&g, "enum:empty", "Enum", "Empty enum", &[], GOST_SRC);
    insert_edge(&g, "fld:empty", "CONSTRAINED_BY", "enum:empty", GOST_SRC);

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

    // Valid diameters as Enum aliases (discrete standard values)
    insert_node_aliases(
        &g,
        "enum:diam_vals",
        "Enum",
        "диаметр enum",
        &["2", "2.5", "3", "4", "5", "6", "8", "10", "12", "14", "16", "18", "20", "24", "27", "30", "36"],
        GOST_SRC,
    );
    insert_edge(&g, "fld:diam", "CONSTRAINED_BY", "enum:diam_vals", GOST_SRC);

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
    insert_node_aliases(&g, "enum:tol", "Enum", "допуск enum", &["6e", "6g", "6h", "8g"], GOST_SRC);
    insert_edge(&g, "fld:tol", "CONSTRAINED_BY", "enum:tol", GOST_SRC);

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

    let ok: Vec<String> = vec!["Field".into(), "Literal".into()];
    for name in &ok {
        assert!(ont.entity_types().contains(name.as_str()), "missing entity {name}");
        assert!(ont.description(name).is_some(), "missing description for entity {name}");
    }

    let rels = ont.raw_relations();
    for key in [
        "CONSTRAINED_BY", "HAS_MIN", "HAS_MAX", "HAS_PATTERN", "HAS_EXPRESSION",
        "IF_FIELD", "IF_VALUE", "HAS_CONSTRAINT",
    ] {
        assert!(rels.contains_key(key), "missing relation {key}");
        assert!(ont.description(key).is_some(), "missing description for relation {key}");
    }

    for (name, _) in ont.constraint_types() {
        assert!(ont.description(name).is_some(), "missing description for constraint {name}");
    }

    assert!(!ont.patterns().is_empty(), "patterns should be declared");
    let export = glossa::graph::ontology_export::export_json(&ont);
    for p in export["patterns"].as_array().unwrap() {
        assert!(p.get("example").is_none(), "patterns must not export example");
    }

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

    let unode = |t: &str, l: &str, aliases: Vec<&str>| glossa::graph::ops::UpsertNode {
        node_type: t.into(),
        label: l.into(),
        source_path: "gost.docx".into(),
        aliases: aliases.iter().map(|s| s.to_string()).collect(),
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
            unode("Field", "Наружный диаметр", vec![]),
            unode("Enum", "Наружный диаметр enum", vec!["125", "150"]),
        ],
        vec![uedge("Наружный диаметр", "CONSTRAINED_BY", "Наружный диаметр enum")],
        1,
    );
    assert!(!out.rejected, "{}", out.message);
    assert_eq!(g.all_nodes().unwrap().len(), 2, "{}", out.message);
    assert_eq!(g.all_edges().unwrap().len(), 1, "{}", out.message);
}

fn build_conditional_grit_graph(g: &GraphStore) {
    insert_node(g, "fld:grit_type", "Field", "тип_зернистости", GOST_SRC);
    insert_node_aliases(g, "enum:grit_type", "Enum", "тип_зернистости enum", &["F", "M"], GOST_SRC);
    insert_edge(g, "fld:grit_type", "CONSTRAINED_BY", "enum:grit_type", GOST_SRC);

    insert_node(g, "fld:grit", "Field", "зернистость", GOST_SRC);

    insert_node(g, "cond:grit_f", "Conditional", "зернистость when F", GOST_SRC);
    insert_node(g, "lit:field", "Literal", "тип_зернистости", GOST_SRC);
    insert_node(g, "lit:val_f", "Literal", "F", GOST_SRC);
    insert_edge(g, "cond:grit_f", "IF_FIELD", "lit:field", GOST_SRC);
    insert_edge(g, "cond:grit_f", "IF_VALUE", "lit:val_f", GOST_SRC);
    insert_node_aliases(g, "enum:grit_f", "Enum", "зернистость F", &["F16", "F24", "F36"], GOST_SRC);
    insert_edge(g, "cond:grit_f", "HAS_CONSTRAINT", "enum:grit_f", GOST_SRC);
    insert_edge(g, "fld:grit", "CONSTRAINED_BY", "cond:grit_f", GOST_SRC);

    insert_node(g, "cond:grit_m", "Conditional", "зернистость when M", GOST_SRC);
    insert_node(g, "lit:val_m", "Literal", "M", GOST_SRC);
    insert_edge(g, "cond:grit_m", "IF_FIELD", "lit:field", GOST_SRC);
    insert_edge(g, "cond:grit_m", "IF_VALUE", "lit:val_m", GOST_SRC);
    insert_node_aliases(g, "enum:grit_m", "Enum", "зернистость M", &["M40", "M60", "M80"], GOST_SRC);
    insert_edge(g, "cond:grit_m", "HAS_CONSTRAINT", "enum:grit_m", GOST_SRC);
    insert_edge(g, "fld:grit", "CONSTRAINED_BY", "cond:grit_m", GOST_SRC);
}

#[test]
fn conditional_enum_switches_domain() {
    let dir = tempfile::tempdir().unwrap();
    let g = setup_corpus(dir.path());
    build_conditional_grit_graph(&g);

    let ok = solve(
        dir.path(),
        &g,
        GOST_SRC,
        "validate",
        vec![
            ("тип_зернистости", serde_json::json!("F")),
            ("зернистость", serde_json::json!("F24")),
        ],
    );
    let violations = ok.get("violations").and_then(|v| v.as_array());
    assert!(violations.map_or(true, |v| v.is_empty()), "F + F24 should pass: {violations:?}");

    let bad = solve(
        dir.path(),
        &g,
        GOST_SRC,
        "validate",
        vec![
            ("тип_зернистости", serde_json::json!("F")),
            ("зернистость", serde_json::json!("M60")),
        ],
    );
    let violations = bad.get("violations").and_then(|v| v.as_array());
    assert!(violations.map_or(false, |v| !v.is_empty()), "F + M60 should violate");

    let infer = solve(
        dir.path(),
        &g,
        GOST_SRC,
        "infer",
        vec![("тип_зернистости", serde_json::json!("F"))],
    );
    let domains = infer["domains"].as_array().unwrap();
    let grit = domains
        .iter()
        .find(|d| d["field"] == "зернистость")
        .expect("зернистость domain");
    let vals = grit["domain"]["value"]["values"].as_array().unwrap();
    let strs: Vec<&str> = vals.iter().map(|v| v.as_str().unwrap()).collect();
    assert_eq!(strs, vec!["F16", "F24", "F36"]);
}

#[test]
fn conditional_enum_adapter_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let g = setup_corpus(dir.path());
    build_conditional_grit_graph(&g);
    let ont = glossa::graph::ontology::Ontology::parse(ONTOLOGY_TOML).unwrap();

    let problem = glossa::constraint_adapter::load_problem(&g, &ont, Some(GOST_SRC)).unwrap();
    assert_eq!(problem.fields.len(), 2);
    let grit = problem.fields.iter().find(|f| f.name == "зернистость").unwrap();
    assert_eq!(grit.constraints.len(), 2, "two Conditional branches on one Field");

    let result = glossa_constraint::solver::solve(
        &problem,
        glossa_constraint::SolveMode::Validate,
        &[
            ("тип_зернистости".into(), serde_json::json!("M")),
            ("зернистость".into(), serde_json::json!("M60")),
        ],
    );
    assert!(result.valid, "M + M60 should validate: {:?}", result.violations);
    let bad = glossa_constraint::solver::solve(
        &problem,
        glossa_constraint::SolveMode::Validate,
        &[
            ("тип_зернистости".into(), serde_json::json!("M")),
            ("зернистость".into(), serde_json::json!("F16")),
        ],
    );
    assert!(!bad.valid, "M + F16 should fail");
}
