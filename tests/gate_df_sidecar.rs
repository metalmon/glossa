use glossa::gate::df::DfTable;

#[test]
fn index_builds_df_sidecar() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    std::fs::write(root.join("a.md"), "modbus configuration pp.19.00.00.00").unwrap();
    std::fs::write(root.join("b.md"), "general modbus notes").unwrap();
    glossa::index::store::index_dir(root, true).unwrap();
    let df = DfTable::load(&DfTable::sidecar_path(&root.join(".glossa"))).expect("sidecar written");
    assert!(df.n_chunks >= 2);
    assert_eq!(df.df("pp.19.00.00.00"), 1);
    assert_eq!(df.df("modbus"), 2);
}
