use crate::model::Chunk;
use std::path::Path;

pub trait Extractor {
    /// Lower-case file extensions this extractor handles (e.g. `["md"]`).
    fn file_types(&self) -> &'static [&'static str];
    /// Extract a file's raw bytes into heading/section-scoped chunks.
    fn extract(&self, path: &Path, bytes: &[u8]) -> anyhow::Result<Vec<Chunk>>;
    /// Whether `extract` actually reads `bytes`. Extractors that key only off the PATH — images are
    /// indexed by filename, no pixels read — return `false` so `extract_file` skips reading the file
    /// body entirely. No point slurping a multi-MB (or huge scanned) image into memory to drop it.
    fn needs_bytes(&self) -> bool {
        true
    }
}

pub mod chunk;
pub mod csv_tsv;
pub mod html;
pub mod image;
pub mod links;
pub mod markdown;
pub mod odf;
pub mod odf_chart;
pub mod office;
pub mod office_chunk;
pub mod office_table;
pub mod ooxml_chart;
pub mod pdf;
pub mod rtf;
pub mod text;

/// Extract one file's chunks into `sink`. Whole-file binary/doc formats (md/office/pdf) are read
/// fully; csv/tsv/html and any other readable file stream from the path (constant memory).
pub fn extract_file(path: &Path, sink: &mut dyn FnMut(Chunk)) -> anyhow::Result<()> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    for ex in crate::walk::extractors() {
        if ex.file_types().contains(&ext.as_str()) {
            // Skip reading the file body for extractors that don't use it (images → name-only).
            let bytes = if ex.needs_bytes() {
                #[cfg(test)]
                if let Some(e) = crate::index::store::read_fault::take_read(path) {
                    return Err(anyhow::Error::from(e).context("extract_file read (injected)"));
                }
                std::fs::read(path)?
            } else {
                Vec::new()
            };
            // Does the body say RTF while the extension says something else? Checked once, for
            // both ways the owning reader can decline the file.
            let rtf_body = ext != "rtf" && rtf::looks_like_rtf(&bytes);
            let chunks = match ex.extract(path, &bytes) {
                // Parsed, and produced nothing. Worse than an error, because there is nothing to
                // see: the file enters the manifest as indexed-and-empty, carries its signature,
                // and is never looked at again. If the bytes are RTF, this is the same mislabeled
                // file arriving by the other door.
                Ok(cs) if cs.is_empty() && rtf_body => rtf::RtfExtractor.extract(path, &bytes)?,
                Ok(cs) => cs,
                // The extension lied. A `.doc` that is really RTF is the common case — a producer
                // saved RTF under Word's old extension — and the reader that owns `.doc` refuses it
                // on the magic bytes, so the document is lost with a permanent skip. The general
                // sniff-on-failure dispatch belongs to the omnivorous design; this is the one shape
                // of it already costing documents.
                //
                // `e.context(..)` rather than a formatted message: the original error stays in the
                // chain, so `is_transient` can still see an io::Error underneath and retry instead
                // of recording a permanent skip.
                Err(e) if rtf_body => rtf::RtfExtractor
                    .extract(path, &bytes)
                    .map_err(|rtf_err| e.context(format!("rtf fallback also failed: {rtf_err}")))?,
                Err(e) => return Err(e),
            };
            for c in chunks {
                sink(c);
            }
            return Ok(());
        }
    }
    match ext.as_str() {
        "csv" | "tsv" => csv_tsv::stream(path, &ext, sink),
        "html" | "htm" => html::stream(path, &ext, sink),
        other => {
            let ft = if other.is_empty() { "txt" } else { other };
            text::stream_text(path, ft, sink)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_doc_that_is_really_rtf_is_extracted_instead_of_skipped() {
        // The reported failure, end to end: a producer saves RTF under Word's old extension, the
        // reader that owns `.doc` refuses the magic bytes, and the document is lost to a permanent
        // skip. The bytes say what the file is; the extension does not get the last word.
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("mislabelled.doc");
        std::fs::write(
            &p,
            // A raw byte string: RTF is backslashes all the way down, and escaping them twice is
            // how a fixture stops resembling the file it stands for.
            br"{\rtf1\ansi{\stylesheet{\s1\outlinelevel0 heading 1;}}\pard\s1 Agreement\par\pard The body of the agreement.\par}",
        )
        .unwrap();
        let mut chunks = Vec::new();
        extract_file(&p, &mut |c| chunks.push(c)).expect("rtf bytes under a .doc name must index");
        assert!(!chunks.is_empty());
        assert!(
            chunks
                .iter()
                .any(|c| c.text.contains("body of the agreement")),
            "{:?}",
            chunks.iter().map(|c| &c.text).collect::<Vec<_>>()
        );
        // It is extracted AS rtf, so the row says what was really read.
        assert_eq!(chunks[0].file_type, "rtf");
    }

    #[test]
    fn a_doc_that_is_not_rtf_keeps_its_own_error() {
        // The fallback must not swallow the real reader's failure for a file that simply is a
        // broken `.doc`: `is_transient` walks the chain, so the cause has to survive.
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("broken.doc");
        // The OLE compound-file signature a real `.doc` starts with, and nothing usable after.
        let mut bytes = vec![0xD0u8, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1];
        bytes.extend_from_slice(b" not a document");
        std::fs::write(&p, &bytes).unwrap();
        let mut chunks = Vec::new();
        let e = extract_file(&p, &mut |c| chunks.push(c)).unwrap_err();
        assert!(
            !format!("{e:#}").contains("rtf fallback"),
            "a non-RTF file must not be reported as an RTF failure: {e:#}"
        );
    }

    #[test]
    fn extract_file_does_not_read_an_image_body() {
        // A .png path that does NOT exist on disk: if extract_file tried to read the body it would
        // error. Because ImageExtractor::needs_bytes() is false, the read is skipped and we still get
        // the name-based chunk — proving the (potentially huge) image body is never slurped.
        let mut chunks = Vec::new();
        extract_file(Path::new("no/such/Diagrams/bus_segment.png"), &mut |c| {
            chunks.push(c)
        })
        .expect("image extraction must not read (or need) the file body");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].file_type, "png");
        assert!(chunks[0].text.contains("bus") && chunks[0].text.contains("segment"));
    }
}
