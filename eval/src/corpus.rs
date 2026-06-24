use crate::dataset::{sanitize_title, Question};
use anyhow::{bail, Context};
use std::path::Path;
use std::process::Command;

pub fn write_corpus(work: &Path, q: &Question) -> anyhow::Result<()> {
    if work.exists() {
        for ent in std::fs::read_dir(work)? {
            let p = ent?.path();
            if p.extension().and_then(|e| e.to_str()) == Some("md") {
                let _ = std::fs::remove_file(p);
            }
        }
        let _ = std::fs::remove_dir_all(work.join(".glossa"));
    } else {
        std::fs::create_dir_all(work)?;
    }
    for para in &q.paragraphs {
        let file = work.join(format!("{}.md", sanitize_title(&para.title)));
        let mut body = format!("# {}\n", para.title);
        for s in &para.sentences {
            body.push_str(s);
            body.push('\n');
        }
        std::fs::write(&file, body).with_context(|| format!("write {file:?}"))?;
    }
    Ok(())
}

pub fn index(work: &Path, kb_bin: &str) -> anyhow::Result<()> {
    let status = Command::new(kb_bin)
        .arg("index")
        .arg(work)
        .status()
        .with_context(|| format!("spawn {kb_bin} index"))?;
    if !status.success() {
        bail!("kb index failed for {work:?}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Paragraph;

    fn q() -> Question {
        Question {
            id: "q1".into(), question: "?".into(), answer: "a".into(),
            paragraphs: vec![Paragraph { title: "Bob Page".into(), sentences: vec!["b1.".into(), "b2.".into()] }],
            supporting_titles: vec!["Bob Page".into()],
        }
    }

    #[test]
    fn write_corpus_writes_md_and_clears_prior() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("stale.md"), b"old").unwrap();
        write_corpus(dir.path(), &q()).unwrap();
        assert!(!dir.path().join("stale.md").exists());
        let body = std::fs::read_to_string(dir.path().join("Bob_Page.md")).unwrap();
        assert!(body.contains("# Bob Page") && body.contains("b1.") && body.contains("b2."));
    }
}
