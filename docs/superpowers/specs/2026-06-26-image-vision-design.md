# Image / Vision Support — Design

**Date:** 2026-06-26
**Status:** Approved (direction confirmed)
**Branch:** `feat/image-vision`

## Goal

Make the image-heavy knowledge base readable by the vision model (qwen3.5-4b). Today loose image files are not indexed at all, and `read` returns no images for PDFs — so PNG folders and scanned/image-only PDFs are completely invisible. Fix: index image files (discoverable by name) and have `read` return the actual image bytes — loose images and scanned-PDF page images — to the vision model. No OCR (the model reads them).

## Background

- Extractor registry `walk::extractors()` lists `MarkdownExtractor, OfficeExtractor, PdfExtractor, …`; `extract_file` also stream-dispatches `html/htm/csv/tsv/text` by extension. **No extractor claims image extensions**, so `*.png/*.jpg/…` are walked but skipped — never indexed.
- `src/read.rs::extract_images(path, max)` opens the file as a ZIP and pulls `word|xl|ppt/media/*` — it returns **nothing for non-zip files** (PDFs, loose images). So `read` gives the vision model images only for Office docs.
- The delivery plumbing already exists: `glossa::tools::read` returns `ReadOut{ text, images: Vec<DocImage> }`; MCP wraps images via `Content::image`; the eval appends them as a TZ image user-message. Once `read` *produces* the images, both surfaces forward them.
- `oxidize-pdf` 2.16.6 (already a dep) exposes `operations::ImageExtractor` (`extract_from_page` / `extract_all`) → `Vec<ExtractedImage { page_number, file_path, width, height, format: {Jpeg|Png|Tiff|Raw} }>`. **File-based** (writes encoded bytes to an `output_dir`, returns paths). DCTDecode→JPEG passthrough, FlateDecode/LZW/raw→PNG (pure-Rust `flate2`); CCITTFax/JPEG2000 skipped. C-free as long as the `external-images` feature stays off.

## Design

### Part A — index loose image files (`src/extract/image.rs`, new)
A new `ImageExtractor` implementing `Extractor`:
- `file_types()` → `["png", "jpg", "jpeg", "gif", "webp", "bmp", "tif", "tiff"]`.
- `extract(path, _bytes)` → one `Chunk`: `location = "(image)"`, `file_type = <ext>`, `text =` a humanized, searchable string built from the **parent folder name + file stem** (separators `_ - .` → spaces). So an image is findable by name/topic via `search`/`glob` (e.g. `kb/Схемы/profibus_сегмент.png` → text `"Схемы profibus сегмент"`). No pixel reading at index time (no OCR).
- Register `Box::new(ImageExtractor)` in `walk::extractors()`.

This mirrors the existing "no-text PDF → index by filename" fallback: the image becomes a discoverable document with one chunk (`ord = 1`).

### Part B — `read` returns the actual image bytes (`src/read.rs`)
Make image production extension-aware. `extract_images` gains the chunk number so PDFs return the right page; signature becomes `extract_images(path: &Path, page: u64, max: usize) -> anyhow::Result<Vec<DocImage>>`:
- **Image file** (ext in the set above): return the file's own bytes as a single `DocImage { mime: mime_for_ext(ext), bytes }` (`page` ignored). Respect a byte cap (see Risks).
- **PDF**: extract **page `page`**'s images via `oxidize_pdf::operations::ImageExtractor::extract_from_page` into a fresh temp dir, read each produced file into bytes, map `ImageFormat`→mime, return up to `max`, then remove the temp dir. Skip images that fail to decode (CCITT/JPX) gracefully.
- **Zip/Office** (existing): unchanged `word|xl|ppt/media/*` behavior.
- Anything else: `Ok(Vec::new())`.

`glossa::tools::read` passes the chunk number `n` as `page` and a small `max` (e.g. 4). Both MCP and the eval already forward the returned `DocImage`s.

### Part C — no tool-contract change
`search`/`glob`/`read` signatures and output formats are unchanged; an indexed image just appears as a normal hit and `read` returns its picture. The MCP `read`'s `include_images` flag still gates image delivery.

## Constraints

- **Pure-Rust, C-free** (`cargo tree -p glossa -i cc` empty): use `oxidize-pdf`'s `ImageExtractor` WITHOUT enabling its `external-images` feature; the `image` crate (already in the tree) is pure-Rust. Add no C deps. Confirm the feature isn't transitively enabled.
- Indexing must not abort on a bad/corrupt image — emit the filename chunk regardless; image *decoding* happens only at `read` time and failures degrade to "no images".
- File-First: images are indexed at their real path; `read(path, n)` addressing unchanged.
- TDD.

## Testing

- **Image extractor**: a `.png` fixture → indexed as one chunk, `file_type="png"`, `location="(image)"`, text contains the folder + stem tokens; `glob "*.png"` and a name `search` find it.
- **read on a loose image**: returns one `DocImage` whose bytes equal the file and mime is `image/png`.
- **read on a PDF page**: a small fixture PDF with one embedded image → `read(path, 1)` returns ≥1 `DocImage` with a sensible mime; a text PDF page with no image → empty images, text still returned (no regression to `extracts_text_from_pdf_fixture`).
- **C-free gate** + full `cargo test -p glossa`.

## Out of scope (follow-ups)

- OCR / text-from-image at index time (the vision model reads images on demand instead).
- Image downscaling/compression to fit context (see Risks — start with a count cap; revisit if scanned pages are too large).
- CCITTFax G3/G4 and JPEG2000 decoding (oxidize-pdf skips these; rare).

## Risks

- **Context size**: a scanned page PNG can be large; base64 in the prompt bloats context and may exceed the vision model's limits. Mitigation v1: cap images per `read` (`max ≈ 4`) and read one page at a time (already the contract). If still too big, add pure-Rust downscaling (the `image` crate) as a follow-up.
- **Temp-dir churn**: the PDF `ImageExtractor` writes to disk per `read`. Use a unique temp dir per call and remove it after; tolerate cleanup failure.
- **Decoder coverage**: DCTDecode + FlateDecode cover the common scanned-page case; CCITT/JPX images are skipped (return no image) rather than garbage.
