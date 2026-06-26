use std::path::Path;

/// Read a document's text, optionally narrowed to a `location` (heading/sheet/page),
/// matched as a case-insensitive substring of the chunk's `location`.
pub fn read_region(path: &Path, location: Option<&str>) -> anyhow::Result<String> {
    let mut chunks = Vec::new();
    crate::extract::extract_file(path, &mut |c| chunks.push(c))?;
    let selected: Vec<&str> = match location {
        Some(loc) => {
            let needle = loc.to_lowercase();
            // Exact location match first (so "p.1" does not also match "p.10"); then substring.
            let exact: Vec<&str> = chunks
                .iter()
                .filter(|c| c.location.to_lowercase() == needle)
                .map(|c| c.text.as_str())
                .collect();
            let matched: Vec<&str> = if !exact.is_empty() {
                exact
            } else {
                chunks
                    .iter()
                    .filter(|c| c.location.to_lowercase().contains(&needle))
                    .map(|c| c.text.as_str())
                    .collect()
            };
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

    #[test]
    fn exact_location_match_wins_over_substring_prefix() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("pages.md");
        // Two headings whose names are substring-prefixes: "p.1" is a prefix of "p.10".
        std::fs::write(&p, b"# p.1\nalpha\n# p.10\nbeta\n").unwrap();
        let one = read_region(&p, Some("p.1")).unwrap();
        assert!(one.contains("alpha"), "exact p.1 must include page-1 text");
        assert!(!one.contains("beta"), "exact p.1 must NOT include p.10 text");
    }

    #[test]
    fn reads_plain_txt_via_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("notes.txt");
        std::fs::write(&p, b"first line\nsecond line\n").unwrap();
        let out = read_region(&p, None).unwrap();
        assert!(out.contains("first line") && out.contains("second line"));
    }

    #[test]
    fn reads_csv_via_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("t.csv");
        std::fs::write(&p, b"name,age\nbob,5\n").unwrap();
        let out = read_region(&p, None).unwrap();
        assert!(out.contains("name,age") && out.contains("bob,5"));
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
    } else if lower.ends_with(".tif") || lower.ends_with(".tiff") {
        Some("image/tiff")
    } else {
        None
    }
}

pub fn extract_images(path: &Path, page: u64, max: usize) -> anyhow::Result<Vec<DocImage>> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
    match ext.as_str() {
        "png" | "jpg" | "jpeg" | "gif" | "webp" | "bmp" | "tif" | "tiff" => {
            if max == 0 {
                return Ok(Vec::new());
            }
            let bytes = std::fs::read(path)?;
            let mime = mime_for(&format!("x.{ext}")).unwrap_or("application/octet-stream");
            Ok(vec![DocImage { mime: mime.into(), bytes }])
        }
        "pdf" => extract_pdf_page_images(path, page, max),
        _ => extract_zip_media(path, max),
    }
}

fn extract_pdf_page_images(path: &Path, page: u64, max: usize) -> anyhow::Result<Vec<DocImage>> {
    use oxidize_pdf::operations::{extract_images_from_pages, ExtractImagesOptions};
    if max == 0 {
        return Ok(Vec::new());
    }
    // Unique per call (pid + monotonic counter) so concurrent reads — tool calls now run in
    // parallel threads — never share a temp dir and clobber each other's files.
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let uniq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp = std::env::temp_dir()
        .join(format!("glossa-img-{}-{}", std::process::id(), uniq));
    let _ = std::fs::create_dir_all(&tmp);
    let result = (|| -> anyhow::Result<Vec<DocImage>> {
        let opts = ExtractImagesOptions {
            output_dir: tmp.clone(),
            create_dir: true,
            ..Default::default()
        };
        // page is 1-based (chunk number); oxidize-pdf uses 0-based page indices
        let page_0 = page.saturating_sub(1) as usize;
        let images = extract_images_from_pages(path, &[page_0], opts)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut out = Vec::new();
        for img in images.into_iter().take(max) {
            let mime = match img.format {
                oxidize_pdf::graphics::ImageFormat::Jpeg => "image/jpeg",
                oxidize_pdf::graphics::ImageFormat::Png => "image/png",
                oxidize_pdf::graphics::ImageFormat::Tiff => "image/tiff",
                _ => continue, // Raw / undecodable → skip
            };
            if let Ok(b) = std::fs::read(&img.file_path) {
                out.push(DocImage { mime: mime.into(), bytes: b });
            }
        }
        Ok(out)
    })();
    let _ = std::fs::remove_dir_all(&tmp);
    // A text-only page, malformed PDF, or decode failure → no images, not an error.
    Ok(result.unwrap_or_default())
}

fn extract_zip_media(path: &Path, max: usize) -> anyhow::Result<Vec<DocImage>> {
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

        let imgs = extract_images(&p, 1, 10).unwrap();
        assert_eq!(imgs.len(), 1);
        assert_eq!(imgs[0].mime, "image/png");
        assert_eq!(imgs[0].bytes, png);

        // max cap respected
        assert_eq!(extract_images(&p, 1, 0).unwrap().len(), 0);
    }

    #[test]
    fn non_zip_returns_no_images() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("plain.md");
        std::fs::write(&p, b"# H\nhi\n").unwrap();
        assert!(extract_images(&p, 1, 10).unwrap().is_empty());
    }

    #[test]
    fn loose_image_returns_its_own_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("pic.png");
        let data = b"\x89PNG\r\n\x1a\nDATA";
        std::fs::write(&p, data).unwrap();
        let imgs = extract_images(&p, 1, 4).unwrap();
        assert_eq!(imgs.len(), 1);
        assert_eq!(imgs[0].mime, "image/png");
        assert_eq!(imgs[0].bytes, data);
    }
}
