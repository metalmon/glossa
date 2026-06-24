use anyhow::{Context, Result};
use bzip2::read::BzDecoder;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use tar::Archive;

pub struct PrepStats {
    pub shards: usize,
    pub articles: usize,
}

#[derive(serde::Deserialize)]
struct WikiArticle {
    title: String,
    text: serde_json::Value, // list of sentences, or list of paragraphs (list of lists)
}

/// Convert the HotpotQA abstracts `tar.bz2` into one markdown file per inner `.bz2` shard.
/// Each article becomes a `# <title>` section followed by its intro text, so glossa's markdown
/// extractor yields one chunk per article with `location == title`.
pub fn prep_fullwiki(archive: &Path, out: &Path, max_shards: Option<usize>) -> Result<PrepStats> {
    let file = File::open(archive).with_context(|| format!("open {archive:?}"))?;
    let mut ar = Archive::new(BzDecoder::new(BufReader::new(file))); // outer bz2 -> tar
    fs::create_dir_all(out)?;
    let mut stats = PrepStats { shards: 0, articles: 0 };

    for entry in ar.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.to_path_buf();
        if path.extension().and_then(|e| e.to_str()) != Some("bz2") {
            continue; // skip directories / non-shard entries
        }
        if let Some(max) = max_shards {
            if stats.shards >= max {
                break;
            }
        }
        let stem = sanitize_shard_name(&path);
        let md_path = out.join(format!("{stem}.md"));
        let mut w = std::io::BufWriter::new(File::create(&md_path)?);
        let reader = BufReader::new(BzDecoder::new(&mut entry)); // inner bz2 -> JSON lines
        let mut shard_articles = 0usize;
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let art: WikiArticle = match serde_json::from_str(&line) {
                Ok(a) => a,
                Err(_) => continue, // skip malformed line, keep going
            };
            let title = art.title.replace(['\n', '\r'], " ");
            let intro = flatten_text(&art.text);
            if title.trim().is_empty() || intro.trim().is_empty() {
                continue;
            }
            writeln!(w, "# {title}")?;
            writeln!(w, "{intro}")?;
            shard_articles += 1;
        }
        w.flush()?;
        stats.shards += 1;
        stats.articles += shard_articles;
    }
    Ok(stats)
}

/// `enwiki/AA/wiki_00.bz2` -> `AA_wiki_00`.
fn sanitize_shard_name(path: &Path) -> String {
    let comps: Vec<String> = path
        .components()
        .filter_map(|c| c.as_os_str().to_str().map(|s| s.to_string()))
        .collect();
    let n = comps.len();
    let dir = if n >= 2 { comps[n - 2].as_str() } else { "x" };
    let file = comps.last().map(|s| s.strip_suffix(".bz2").unwrap_or(s)).unwrap_or("shard");
    format!("{dir}_{file}")
}

/// HotpotQA `text` is a list of sentence strings, or a list of paragraphs (each a list of
/// sentences). Flatten to one string.
fn flatten_text(v: &serde_json::Value) -> String {
    let mut out = String::new();
    if let Some(arr) = v.as_array() {
        for item in arr {
            match item {
                serde_json::Value::String(s) => {
                    out.push_str(s);
                    out.push(' ');
                }
                serde_json::Value::Array(inner) => {
                    for s in inner {
                        if let Some(s) = s.as_str() {
                            out.push_str(s);
                            out.push(' ');
                        }
                    }
                }
                _ => {}
            }
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use bzip2::write::BzEncoder;
    use bzip2::Compression;
    use std::io::Write;

    fn bz(data: &[u8]) -> Vec<u8> {
        let mut e = BzEncoder::new(Vec::new(), Compression::default());
        e.write_all(data).unwrap();
        e.finish().unwrap()
    }

    #[test]
    fn converts_nested_archive_to_md_sections() {
        let dir = tempfile::tempdir().unwrap();
        // inner shard: two JSON-line articles (one flat text, one nested paragraphs)
        let jsonl = concat!(
            r#"{"title":"Alpha","text":["A1.","A2."]}"#, "\n",
            r#"{"title":"Beta","text":[["B1.","B2."]]}"#, "\n",
        );
        let inner_bz = bz(jsonl.as_bytes());
        // tar containing enwiki/AA/wiki_00.bz2
        let mut tar_buf = Vec::new();
        {
            let mut tb = tar::Builder::new(&mut tar_buf);
            let mut hdr = tar::Header::new_gnu();
            hdr.set_size(inner_bz.len() as u64);
            hdr.set_mode(0o644);
            hdr.set_cksum();
            tb.append_data(&mut hdr, "enwiki/AA/wiki_00.bz2", &inner_bz[..]).unwrap();
            tb.finish().unwrap();
        }
        let archive = dir.path().join("abstracts.tar.bz2");
        std::fs::write(&archive, bz(&tar_buf)).unwrap();

        let out = dir.path().join("corpus");
        let stats = prep_fullwiki(&archive, &out, None).unwrap();
        assert_eq!(stats.shards, 1);
        assert_eq!(stats.articles, 2);

        let md = std::fs::read_to_string(out.join("AA_wiki_00.md")).unwrap();
        assert!(md.contains("# Alpha") && md.contains("A1. A2."));
        assert!(md.contains("# Beta") && md.contains("B1. B2."));
    }

    #[test]
    fn max_shards_limits_output() {
        let dir = tempfile::tempdir().unwrap();
        let inner_bz = bz(br#"{"title":"X","text":["x."]}"#);
        let mut tar_buf = Vec::new();
        {
            let mut tb = tar::Builder::new(&mut tar_buf);
            for name in ["enwiki/AA/wiki_00.bz2", "enwiki/AA/wiki_01.bz2"] {
                let mut hdr = tar::Header::new_gnu();
                hdr.set_size(inner_bz.len() as u64);
                hdr.set_mode(0o644);
                hdr.set_cksum();
                tb.append_data(&mut hdr, name, &inner_bz[..]).unwrap();
            }
            tb.finish().unwrap();
        }
        let archive = dir.path().join("a.tar.bz2");
        std::fs::write(&archive, bz(&tar_buf)).unwrap();
        let out = dir.path().join("c");
        let stats = prep_fullwiki(&archive, &out, Some(1)).unwrap();
        assert_eq!(stats.shards, 1, "max_shards=1 stops after one shard");
    }

    #[test]
    fn prep_output_indexes_with_title_as_location() {
        use glossa::extract::markdown::MarkdownExtractor;
        use glossa::extract::Extractor;

        let dir = tempfile::tempdir().unwrap();
        let inner_bz = bz(br#"{"title":"Anarchism","text":["Anarchism is a political philosophy."]}"#);
        let mut tar_buf = Vec::new();
        {
            let mut tb = tar::Builder::new(&mut tar_buf);
            let mut hdr = tar::Header::new_gnu();
            hdr.set_size(inner_bz.len() as u64);
            hdr.set_mode(0o644);
            hdr.set_cksum();
            tb.append_data(&mut hdr, "enwiki/AA/wiki_00.bz2", &inner_bz[..]).unwrap();
            tb.finish().unwrap();
        }
        let archive = dir.path().join("a.tar.bz2");
        std::fs::write(&archive, bz(&tar_buf)).unwrap();
        let out = dir.path().join("c");
        prep_fullwiki(&archive, &out, None).unwrap();

        let md_path = out.join("AA_wiki_00.md");
        let bytes = std::fs::read(&md_path).unwrap();
        let chunks = MarkdownExtractor.extract(&md_path, &bytes).unwrap();
        assert!(
            chunks.iter().any(|c| c.location == "Anarchism"),
            "prep output must index with location == title (Recall@k depends on it); got {:?}",
            chunks.iter().map(|c| &c.location).collect::<Vec<_>>()
        );
    }
}
