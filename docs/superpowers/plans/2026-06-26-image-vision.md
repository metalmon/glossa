# Image / Vision Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Index loose image files (discoverable by name) and have `read` return the actual image bytes — loose images and scanned-PDF page images — so the vision model can read them. No OCR.

**Architecture:** A new `ImageExtractor` registers image extensions and emits one filename-derived chunk per image (like the no-text-PDF fallback). `read.rs::extract_images` becomes extension-aware: image file → its own bytes; PDF → page-N images via oxidize-pdf's `ImageExtractor`; Office zip → existing media. The MCP and eval surfaces already forward the returned `DocImage`s to the vision model.

**Tech Stack:** Rust, `oxidize-pdf` 2.16.6 (`operations::ImageExtractor`), `zip` (existing), the `image` crate only if needed (pure-Rust).

## Global Constraints

- **Pure-Rust, C-free**: `cargo tree -p glossa -i cc` MUST stay empty. Do NOT enable oxidize-pdf's `external-images` feature. Add no C deps.
- Indexing must NOT abort on a bad image — always emit the filename chunk; image *decoding* is read-time only and failures degrade to "no images".
- File-First: images indexed at their real path; `read(path, n)` addressing unchanged.
- TDD. Frequent commits.

---

### Task 1: `ImageExtractor` — index loose image files by name

**Files:**
- Create: `src/extract/image.rs`
- Modify: `src/extract.rs` (add `pub mod image;`), `src/walk.rs` (register in `extractors()`)

**Interfaces:**
- Produces: `struct ImageExtractor` implementing `crate::extract::Extractor` (`file_types`, `extract`).
- Consumes: `crate::model::Chunk`, the `Extractor` trait.

- [ ] **Step 1: Write `src/extract/image.rs`**

```rust
use crate::extract::Extractor;
use crate::model::Chunk;
use std::path::Path;

pub struct ImageExtractor;

/// Build a searchable label from the parent folder name + file stem, with separators turned into
/// spaces (so an image is findable by name/topic via search/glob). No pixels are read at index time.
fn label_for(path: &Path) -> String {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    let parent = path.parent().and_then(|p| p.file_name()).and_then(|s| s.to_str()).unwrap_or("");
    let raw = format!("{parent} {stem}");
    raw.chars().map(|c| if c == '_' || c == '-' || c == '.' { ' ' } else { c }).collect::<String>()
        .split_whitespace().collect::<Vec<_>>().join(" ")
}

impl Extractor for ImageExtractor {
    fn file_types(&self) -> &'static [&'static str] {
        &["png", "jpg", "jpeg", "gif", "webp", "bmp", "tif", "tiff"]
    }

    fn extract(&self, path: &Path, _bytes: &[u8]) -> anyhow::Result<Vec<Chunk>> {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("img").to_lowercase();
        Ok(vec![Chunk {
            doc_path: path.to_path_buf(),
            location: "(image)".into(),
            file_type: ext,
            text: label_for(path),
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexes_image_by_folder_and_stem() {
        let chunks = ImageExtractor
            .extract(Path::new("kb/Схемы/profibus_сегмент-2.png"), b"\x89PNG")
            .unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].file_type, "png");
        assert_eq!(chunks[0].location, "(image)");
        assert!(chunks[0].text.contains("Схемы") && chunks[0].text.contains("profibus") && chunks[0].text.contains("сегмент"));
    }
}
```

- [ ] **Step 2: Register the module + extractor**

In `src/extract.rs` add `pub mod image;` alongside the other `pub mod` declarations.
In `src/walk.rs`: add the import `use crate::extract::image::ImageExtractor;` and `Box::new(ImageExtractor),` to the `vec![...]` in `extractors()`.

- [ ] **Step 3: Run tests**

Run: `cargo test -p glossa extract::image` and `cargo test -p glossa walk`
Expected: PASS.

- [ ] **Step 4: Integration — an indexed image is discoverable**

Add to `src/index/store.rs` `incremental_tests`:
```rust
#[test]
fn index_dir_indexes_loose_images() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("Схемы")).unwrap();
    std::fs::write(dir.path().join("Схемы").join("profibus.png"), b"\x89PNG\r\n").unwrap();
    index_dir(dir.path(), true).unwrap();
    let idx = DocIndex::open_or_create(dir.path()).unwrap();
    assert!(idx.search("profibus", 10).unwrap().iter().any(|h| h.path.ends_with("profibus.png")),
        "loose image is searchable by name");
}
```
Run: `cargo test -p glossa index::store`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/extract/image.rs src/extract.rs src/walk.rs src/index/store.rs
git commit -m "feat(extract): index loose image files by name (png/jpg/…)"
```

---

### Task 2: `read` returns image bytes (loose images + scanned-PDF pages)

**Files:**
- Modify: `src/read.rs` (`extract_images` becomes extension-aware + gains a `page` arg)
- Modify: `src/tools.rs` (pass `n` as the page to `extract_images`)
- Modify any other `extract_images` call sites (search the tree)

**Interfaces:**
- Produces: `extract_images(path: &Path, page: u64, max: usize) -> anyhow::Result<Vec<DocImage>>`.
- Consumes: `oxidize_pdf::operations::ImageExtractor` (+ `ExtractImagesOptions`, `ExtractedImage`, `ImageFormat`) — **verify exact paths/signatures against the crate source `~/.cargo/registry/src/*/oxidize-pdf-2.16.6/src/operations/extract_images.rs` before writing**.

- [ ] **Step 1: Find all `extract_images(` call sites** so the signature change compiles everywhere.

Run: `rg -n "extract_images\(" src/` — expect `src/tools.rs` (the read path) and `src/read.rs` tests. Note them.

- [ ] **Step 2: Rewrite `extract_images` to dispatch by extension** (`src/read.rs`)

Add a `page: u64` parameter and dispatch:
```rust
pub fn extract_images(path: &Path, page: u64, max: usize) -> anyhow::Result<Vec<DocImage>> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
    match ext.as_str() {
        "png" | "jpg" | "jpeg" | "gif" | "webp" | "bmp" | "tif" | "tiff" => {
            // The file IS the image. Return its bytes (subject to `max`).
            if max == 0 { return Ok(Vec::new()); }
            let bytes = std::fs::read(path)?;
            let mime = mime_for(&format!("x.{ext}")).unwrap_or("application/octet-stream");
            Ok(vec![DocImage { mime: mime.into(), bytes }])
        }
        "pdf" => extract_pdf_page_images(path, page, max),
        _ => extract_zip_media(path, max), // existing Office/zip behavior
    }
}
```
Rename the current zip body into `fn extract_zip_media(path: &Path, max: usize) -> anyhow::Result<Vec<DocImage>>` (the existing logic verbatim). Keep `mime_for`.

- [ ] **Step 3: Implement `extract_pdf_page_images`** (`src/read.rs`)

Use oxidize-pdf's `ImageExtractor` into a unique temp dir, read produced files back, map format→mime, cleanup. VERIFY the API against the crate; the shape is roughly:
```rust
fn extract_pdf_page_images(path: &Path, page: u64, max: usize) -> anyhow::Result<Vec<DocImage>> {
    use oxidize_pdf::operations::{ImageExtractor, ExtractImagesOptions};
    use oxidize_pdf::parser::{PdfReader, PdfDocument, ParseOptions};
    if max == 0 { return Ok(Vec::new()); }
    // unique temp dir (no Date/rand available in lib code here — use the page + a process/file hint)
    let tmp = std::env::temp_dir().join(format!("glossa-img-{}-{}", std::process::id(), page));
    let _ = std::fs::create_dir_all(&tmp);
    let result = (|| -> anyhow::Result<Vec<DocImage>> {
        let bytes = std::fs::read(path)?;
        let reader = PdfReader::new_with_options(std::io::Cursor::new(bytes), ParseOptions::lenient())?;
        let doc = PdfDocument::new(reader);
        let opts = ExtractImagesOptions { output_dir: tmp.clone(), create_dir: true, ..Default::default() };
        let mut out = Vec::new();
        // extract images for the 1-based `page` (CONFIRM 0- vs 1-based against the crate; the
        // text extractor treats partition pages as 0-based, page label = page+1).
        let extractor = ImageExtractor::new(doc, opts);
        let images = extractor.extract_from_page((page.saturating_sub(1)) as usize)?; // verify base
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
    // A PDF that has no extractable images (pure-text page) or a decode failure → no images, not an error.
    Ok(result.unwrap_or_default())
}
```
NOTE: the exact constructor (`ImageExtractor::new` vs a convenience fn), the page method name, the page index base, and the `graphics::ImageFormat` path are **to be confirmed against the crate source**; adjust to whatever compiles. Wrap PDF parsing so a malformed PDF yields `Ok(vec![])`, never a panic/error that aborts a read.

- [ ] **Step 4: Update `src/tools.rs` read to pass the page**

Change the call (currently `crate::read::extract_images(std::path::Path::new(path), 8)`) to:
```rust
let images = crate::read::extract_images(std::path::Path::new(path), n, 4).unwrap_or_default();
```
(`n` is the resolved chunk number; cap at 4 images to bound context.)

- [ ] **Step 5: Update read.rs tests + add image-return tests**

Fix the existing `extract_images` tests to the new arity (pass a `page`, e.g. `extract_images(&p, 1, 10)`); the docx-media test stays valid (zip path). Add:
```rust
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
```

- [ ] **Step 6: Run tests + C-free gate**

Run: `cargo test -p glossa read::` , `cargo test -p glossa tools::` , `cargo tree -p glossa -i cc` (empty), then full `cargo test -p glossa`.
Expected: all PASS; existing `extracts_text_from_pdf_fixture` still green (text PDF page → no images, text intact).

- [ ] **Step 7: Commit**

```bash
git add src/read.rs src/tools.rs
git commit -m "feat(read): return image bytes for loose images and scanned-PDF pages (vision)"
```

## Self-Review

**Coverage:** loose-image indexing (T1), loose-image bytes on read (T2), PDF page images (T2), Office media unchanged (T2). **Types:** `extract_images(path, page: u64, max) -> Vec<DocImage>` consistent across read.rs + the tools.rs call site; `ImageExtractor`/`ExtractedImage`/`ImageFormat` flagged for crate verification. **Placeholders:** the oxidize-pdf API specifics are explicitly "verify against the crate"; everything else is concrete. **C-free:** no new deps; `external-images` stays off (gate in T2 Step 6).
