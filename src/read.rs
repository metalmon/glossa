use crate::walk::extractors;
use anyhow::Context;
use std::path::Path;

/// Read a document's text, optionally narrowed to a `location` (heading/sheet/page),
/// matched as a case-insensitive substring of the chunk's `location`.
pub fn read_region(path: &Path, location: Option<&str>) -> anyhow::Result<String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let mut chunks = Vec::new();
    for ex in extractors() {
        if ex.file_types().contains(&ext.as_str()) {
            chunks = ex.extract(path, &bytes)?;
            break;
        }
    }
    let selected: Vec<&str> = match location {
        Some(loc) => {
            let needle = loc.to_lowercase();
            let matched: Vec<&str> = chunks
                .iter()
                .filter(|c| c.location.to_lowercase().contains(&needle))
                .map(|c| c.text.as_str())
                .collect();
            if matched.is_empty() {
                chunks.iter().map(|c| c.text.as_str()).collect()
            } else {
                matched
            }
        }
        None => chunks.iter().map(|c| c.text.as_str()).collect(),
    };
    Ok(selected.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_whole_then_narrows_by_location() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("d.md");
        std::fs::write(&p, b"# Intro\nalpha\n## Body\nbeta\n").unwrap();

        let whole = read_region(&p, None).unwrap();
        assert!(whole.contains("alpha") && whole.contains("beta"));

        let body = read_region(&p, Some("body")).unwrap();
        assert!(body.contains("beta") && !body.contains("alpha"));
    }

    #[test]
    fn unmatched_location_falls_back_to_whole_doc() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("d.md");
        std::fs::write(&p, b"# Intro\nalpha\n").unwrap();
        let out = read_region(&p, Some("nonexistent")).unwrap();
        assert!(out.contains("alpha"));
    }
}

// ── Task 3: extract_images ────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct DocImage {
    pub mime: String,
    pub bytes: Vec<u8>,
}

fn mime_for(name: &str) -> Option<&'static str> {
    let lower = name.to_lowercase();
    if lower.ends_with(".png") {
        Some("image/png")
    } else if lower.ends_with(".jpg") || lower.ends_with(".jpeg") {
        Some("image/jpeg")
    } else if lower.ends_with(".gif") {
        Some("image/gif")
    } else if lower.ends_with(".bmp") {
        Some("image/bmp")
    } else if lower.ends_with(".webp") {
        Some("image/webp")
    } else {
        None
    }
}

pub fn extract_images(path: &Path, max: usize) -> anyhow::Result<Vec<DocImage>> {
    let bytes = std::fs::read(path)?;
    let reader = std::io::Cursor::new(bytes);
    let mut archive = match zip::ZipArchive::new(reader) {
        Ok(a) => a,
        Err(_) => return Ok(Vec::new()), // not a zip → no images
    };
    let media_names: Vec<String> = archive
        .file_names()
        .filter(|n| {
            n.starts_with("word/media/")
                || n.starts_with("xl/media/")
                || n.starts_with("ppt/media/")
        })
        .map(|s| s.to_string())
        .collect();

    let mut out = Vec::new();
    for name in media_names {
        if out.len() >= max {
            break;
        }
        let Some(mime) = mime_for(&name) else { continue };
        use std::io::Read;
        let mut entry = archive.by_name(&name)?;
        let mut buf = Vec::new();
        entry.read_to_end(&mut buf)?;
        out.push(DocImage { mime: mime.into(), bytes: buf });
    }
    Ok(out)
}

#[cfg(test)]
mod image_tests {
    use super::*;
    use std::io::Write;
    use zip::write::SimpleFileOptions;

    fn make_docx_with_png(path: &Path, png: &[u8]) {
        let f = std::fs::File::create(path).unwrap();
        let mut zw = zip::ZipWriter::new(f);
        let opts = SimpleFileOptions::default();
        zw.start_file("word/document.xml", opts).unwrap();
        zw.write_all(b"<w:document/>").unwrap();
        zw.start_file("word/media/image1.png", opts).unwrap();
        zw.write_all(png).unwrap();
        zw.finish().unwrap();
    }

    #[test]
    fn extracts_png_media_from_office_zip() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("withimg.docx");
        let png = b"\x89PNG\r\n\x1a\n-fake-png-bytes";
        make_docx_with_png(&p, png);

        let imgs = extract_images(&p, 10).unwrap();
        assert_eq!(imgs.len(), 1);
        assert_eq!(imgs[0].mime, "image/png");
        assert_eq!(imgs[0].bytes, png);

        // max cap respected
        assert_eq!(extract_images(&p, 0).unwrap().len(), 0);
    }

    #[test]
    fn non_zip_returns_no_images() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("plain.md");
        std::fs::write(&p, b"# H\nhi\n").unwrap();
        assert!(extract_images(&p, 10).unwrap().is_empty());
    }
}
