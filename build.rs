use std::{env, fs, path::Path};

fn main() {
    let manifest = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let dir = Path::new(&manifest).join("templates").join("ontologies");
    println!("cargo:rerun-if-changed={}", dir.display());

    let mut entries: Vec<(String, String)> = Vec::new();
    if dir.exists() {
        for e in fs::read_dir(&dir).expect("read templates dir") {
            let p = e.expect("dir entry").path();
            if p.extension().and_then(|x| x.to_str()) == Some("toml") {
                let name = p.file_stem().unwrap().to_str().unwrap().to_string();
                let abs = p.to_str().unwrap().replace('\\', "/");
                println!("cargo:rerun-if-changed={abs}");
                entries.push((name, abs));
            }
        }
    }
    entries.sort();

    let mut out = String::from("pub static TEMPLATES: &[(&str, &str)] = &[\n");
    for (name, abs) in &entries {
        out.push_str(&format!("    ({name:?}, include_str!({abs:?})),\n"));
    }
    out.push_str("];\n");

    let dest = Path::new(&env::var("OUT_DIR").unwrap()).join("ontology_templates.rs");
    fs::write(dest, out).expect("write generated templates");
}
