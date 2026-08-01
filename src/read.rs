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
        assert!(
            !one.contains("beta"),
            "exact p.1 must NOT include p.10 text"
        );
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

    #[test]
    fn exact_empty_pdf_page_does_not_fallback_to_whole_doc() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("three-page-blank-middle.pdf");
        std::fs::write(
            &p,
            include_bytes!("../tests/fixtures/three-page-blank-middle.pdf"),
        )
        .unwrap();
        let whole = read_region(&p, None).unwrap();
        let p2 = read_region(&p, Some("p.2")).unwrap();
        assert!(
            whole.contains("page one") && whole.contains("page three"),
            "whole: {whole}"
        );
        assert!(
            !p2.contains("page one") && !p2.contains("page three"),
            "p.2 must not fall back: {p2:?}"
        );
        assert!(p2.trim().is_empty(), "blank page body: {p2:?}");
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
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    match ext.as_str() {
        "png" | "jpg" | "jpeg" | "gif" | "webp" | "bmp" | "tif" | "tiff" => {
            if max == 0 {
                return Ok(Vec::new());
            }
            let bytes = std::fs::read(path)?;
            let mime = mime_for(&format!("x.{ext}")).unwrap_or("application/octet-stream");
            Ok(vec![DocImage {
                mime: mime.into(),
                bytes,
            }])
        }
        "pdf" => extract_pdf_page_images(path, page, max),
        "htm" | "html" => extract_html_images(path, max),
        _ => extract_zip_media(path, max),
    }
}

fn extract_pdf_page_images(path: &Path, page: u64, max: usize) -> anyhow::Result<Vec<DocImage>> {
    use pdf_oxide::extractors::ImageData;
    use pdf_oxide::PdfDocument;

    if max == 0 {
        return Ok(Vec::new());
    }
    let Some(page_index) = page.checked_sub(1).and_then(|p| usize::try_from(p).ok()) else {
        return Ok(Vec::new());
    };
    let Ok(document) = PdfDocument::open(path) else {
        return Ok(Vec::new());
    };
    let Ok(images) = document.extract_images(page_index) else {
        return Ok(Vec::new());
    };

    let mut out = Vec::new();
    for image in images {
        if out.len() >= max {
            break;
        }
        let (mime, bytes) = match image.data() {
            ImageData::Jpeg(bytes) => ("image/jpeg", bytes.clone()),
            ImageData::Raw { .. } => {
                let Ok(bytes) = image.to_png_bytes() else {
                    continue;
                };
                ("image/png", bytes)
            }
        };
        out.push(DocImage {
            mime: mime.into(),
            bytes,
        });
    }
    Ok(out)
}

/// Render one PDF page (1-based) to an image.
///
/// A page that is a single full-bleed embedded scan is returned byte-for-byte
/// from the source PDF (original JPEG, no re-encode). Everything else is
/// rasterized to JPEG @ 200 DPI. Empty on failure.
pub fn render_pdf_page(path: &Path, page: u64) -> anyhow::Result<Option<DocImage>> {
    use pdf_oxide::rendering::{render_page, ImageFormat, RenderOptions};
    use pdf_oxide::PdfDocument;

    let Some(page_index) = page.checked_sub(1).and_then(|p| usize::try_from(p).ok()) else {
        return Ok(None);
    };
    let Ok(document) = PdfDocument::open(path) else {
        return Ok(None);
    };
    if let Some(image) = full_bleed_scan(&document, page_index) {
        return Ok(Some(image));
    }
    let mut options = RenderOptions::with_dpi(200);
    options.format = ImageFormat::Jpeg;
    let Ok(image) = render_page(&document, page_index, &options) else {
        return Ok(None);
    };
    Ok(Some(DocImage {
        mime: "image/jpeg".into(),
        bytes: image.data,
    }))
}

/// If the page is exactly one image filling the media box, return it as-is.
fn full_bleed_scan(document: &pdf_oxide::PdfDocument, page_index: usize) -> Option<DocImage> {
    use pdf_oxide::extractors::ImageData;

    let images = document.extract_images(page_index).ok()?;
    if images.len() != 1 {
        return None;
    }
    let image = &images[0];
    if image.rotation_degrees() != 0 {
        return None;
    }
    let bbox = image.bbox()?;
    let media = document.get_page_media_box(page_index).ok()?;
    let (x0, y0, x1, y1) = media;
    // Tolerate rounding to the point: bbox must cover the media box.
    const EPS: f32 = 1.0;
    let covers = bbox.x <= x0 + EPS
        && bbox.y <= y0 + EPS
        && bbox.x + bbox.width >= x1 - EPS
        && bbox.y + bbox.height >= y1 - EPS;
    if !covers {
        return None;
    }
    let ImageData::Jpeg(bytes) = image.data() else {
        return None;
    };
    Some(DocImage {
        mime: "image/jpeg".into(),
        bytes: bytes.to_vec(),
    })
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
        let Some(mime) = mime_for(&name) else {
            continue;
        };
        use std::io::Read;
        let mut entry = match archive.by_name(&name) {
            Ok(e) => e,
            Err(_) => continue,
        };
        let mut buf = Vec::new();
        if entry.read_to_end(&mut buf).is_err() {
            continue;
        }
        out.push(DocImage {
            mime: mime.into(),
            bytes: buf,
        });
    }
    Ok(out)
}

/// Extract images referenced by `<img>` tags in an HTML file.
///
/// Only local file paths are resolved; remote URLs (http/https/ftp) and
/// inline data URIs are silently skipped. Relative paths are resolved
/// relative to the HTML file's parent directory.
fn extract_html_images(path: &Path, max: usize) -> anyhow::Result<Vec<DocImage>> {
    if max == 0 {
        return Ok(Vec::new());
    }
    let html = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(_) => return Ok(Vec::new()),
    };
    let dir = path.parent().unwrap_or(Path::new("."));
    let mut out = Vec::new();
    // Match <img ... src="..."> — captures the src attribute value.
    for cap in regex_captures_iter(r#"<img[^>]+src="([^"]+)""#, &html) {
        if out.len() >= max {
            break;
        }
        let src = &cap[1];
        // Skip remote URLs and inline data URIs.
        if src.starts_with("http://")
            || src.starts_with("https://")
            || src.starts_with("ftp://")
            || src.starts_with("data:")
        {
            continue;
        }
        let img_path = if std::path::Path::new(src).is_absolute() {
            std::path::PathBuf::from(src)
        } else {
            dir.join(src)
        };
        let Some(mime) = mime_for(img_path.to_str().unwrap_or("")) else {
            continue;
        };
        let Ok(bytes) = std::fs::read(&img_path) else {
            continue;
        };
        out.push(DocImage {
            mime: mime.into(),
            bytes,
        });
    }
    Ok(out)
}

fn regex_captures_iter<'a>(pat: &'a str, text: &'a str) -> Vec<regex::Captures<'a>> {
    let re = regex::Regex::new(pat).unwrap();
    re.captures_iter(text).collect()
}

#[cfg(test)]
mod image_tests {
    use super::*;
    use base64::Engine;
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

    fn sample_pdf() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("sample.pdf")
    }

    #[test]
    fn render_pdf_page_returns_jpeg_magic() {
        let image = render_pdf_page(&sample_pdf(), 1)
            .unwrap()
            .expect("sample PDF page 1 should render");
        assert_eq!(image.mime, "image/jpeg");
        assert!(image.bytes.starts_with(b"\xff\xd8"));
        assert!(image.bytes.len() > 100);
    }

    #[test]
    fn render_pdf_page_missing_page_is_none() {
        assert_eq!(render_pdf_page(&sample_pdf(), 99).unwrap(), None);
    }

    fn fixture(name: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join(name)
    }

    #[test]
    fn full_bleed_scan_passes_original_jpeg_through() {
        let p = fixture("order365.pdf");
        let rendered = render_pdf_page(&p, 1)
            .unwrap()
            .expect("scan page 1 should render");
        assert_eq!(rendered.mime, "image/jpeg");
        assert!(rendered.bytes.starts_with(b"\xff\xd8"));
        // The page is one full-page JPEG scan; the original bytes must be
        // returned verbatim (not re-rasterized/re-encoded).
        let extracted = extract_images(&p, 1, 1).unwrap();
        assert_eq!(extracted.len(), 1);
        assert_eq!(rendered.bytes, extracted[0].bytes);
    }

    #[test]
    fn full_bleed_scan_page_fits_under_mcp_line_limit() {
        // Regression: 200 DPI PNG renders of these scan pages exceeded the
        // 4 MiB MCP stdio line limit and hung page_image calls.
        let cap_bytes = 4 * 1024 * 1024;
        for name in ["order365.pdf", "order455.pdf", "pril1.pdf"] {
            let p = fixture(name);
            let rendered = render_pdf_page(&p, 1)
                .unwrap()
                .expect("scan page 1 should render");
            let b64 = base64::engine::general_purpose::STANDARD.encode(&rendered.bytes);
            assert!(
                b64.len() < cap_bytes,
                "{name}: base64 page image {len} bytes exceeds {cap_bytes}",
                len = b64.len()
            );
        }
    }

    #[test]
    fn extract_images_from_pdf_soft_fails() {
        assert!(extract_images(&sample_pdf(), 1, 4).is_ok());
    }

    #[test]
    fn extract_html_images_returns_local_png() {
        let dir = tempfile::tempdir().unwrap();
        let png = b"\x89PNG\r\n\x1a\nfake-png";
        std::fs::write(dir.path().join("photo.png"), png).unwrap();
        std::fs::write(
            dir.path().join("page.htm"),
            r#"<html><body><img src="photo.png"></body></html>"#,
        )
        .unwrap();
        let imgs = extract_images(&dir.path().join("page.htm"), 1, 10).unwrap();
        assert_eq!(imgs.len(), 1);
        assert_eq!(imgs[0].mime, "image/png");
        assert_eq!(imgs[0].bytes, png);
    }

    #[test]
    fn extract_html_images_skips_remote_urls() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("page.html"),
            r#"<img src="https://example.com/a.png"><img src="local.png">"#,
        )
        .unwrap();
        std::fs::write(dir.path().join("local.png"), b"LOCAL").unwrap();
        let imgs = extract_images(&dir.path().join("page.html"), 1, 10).unwrap();
        assert_eq!(imgs.len(), 1);
        assert_eq!(imgs[0].bytes, b"LOCAL");
    }

    #[test]
    fn extract_html_images_skips_data_uri() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("p.htm"),
            r#"<img src="data:image/png;base64,abc">"#,
        )
        .unwrap();
        assert!(extract_images(&dir.path().join("p.htm"), 1, 10)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn extract_html_images_max_zero_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.png"), b"PNG").unwrap();
        std::fs::write(dir.path().join("p.htm"), r#"<img src="a.png">"#).unwrap();
        assert!(extract_images(&dir.path().join("p.htm"), 1, 0)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn extract_html_images_missing_file_skipped() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("p.htm"),
            r#"<img src="nope.png"><img src="also-nope.jpg">"#,
        )
        .unwrap();
        assert!(extract_images(&dir.path().join("p.htm"), 1, 10)
            .unwrap()
            .is_empty());
    }
}
