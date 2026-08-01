//! DOCX → PDF conversion via the pure-Rust `rdocx` layout engine.
//!
//! Used by `get_source_file` to deliver Word documents as PDF (many clients render
//! `.docx` inconsistently). Conversion is fully in-memory; callers treat any error as
//! "return the original bytes instead", so this module never panics past its boundary.

/// Convert DOCX bytes to PDF bytes, entirely in memory.
///
/// `rdocx` parses the DOCX, lays it out (its own pagination — not Word's), and renders a
/// real, text-bearing PDF. Any parse/layout error, or a panic inside the layout engine, is
/// surfaced as `Err` so the caller can fall back to the original file.
pub fn docx_to_pdf(bytes: &[u8]) -> anyhow::Result<Vec<u8>> {
    // rdocx's layout/shaping can panic on pathological input, like other document parsers in
    // this crate (see extract/pdf.rs); contain it so a bad DOCX never aborts the request.
    let owned = bytes.to_vec();
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
        move || -> anyhow::Result<Vec<u8>> {
            let doc = rdocx::Document::from_bytes(&owned)
                .map_err(|e| anyhow::anyhow!("parse docx: {e}"))?;
            let pdf = doc
                .to_pdf()
                .map_err(|e| anyhow::anyhow!("render pdf: {e}"))?;
            Ok(pdf)
        },
    ));
    match caught {
        Ok(res) => res,
        Err(_) => anyhow::bail!("docx layout engine panicked"),
    }
}

/// Best-effort page count of an in-memory PDF, for the delivery note. Returns `None` if the
/// PDF can't be parsed (the note then omits the count rather than failing).
pub fn pdf_page_count(pdf: &[u8]) -> Option<usize> {
    let owned = pdf.to_vec();
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        pdf_oxide::PdfDocument::from_bytes(owned)
            .ok()?
            .page_count()
            .ok()
    }))
    .ok()
    .flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid DOCX in memory so the test needs no committed binary fixture.
    fn tiny_docx() -> Vec<u8> {
        let mut d = rdocx::Document::new();
        d.add_paragraph("Hello from the conversion test fixture.");
        d.add_paragraph("A second paragraph so the document has real content.");
        d.to_bytes().expect("serialize docx")
    }

    #[test]
    fn converts_valid_docx_to_pdf() {
        let docx = tiny_docx();
        let pdf = docx_to_pdf(&docx).expect("convert docx to pdf");
        assert!(pdf.starts_with(b"%PDF"), "output must be a real PDF");
        assert!(
            pdf_page_count(&pdf).unwrap_or(0) >= 1,
            "PDF must have at least one page"
        );
    }

    #[test]
    fn malformed_docx_is_err_not_panic() {
        let junk = b"this is definitely not a docx archive";
        assert!(
            docx_to_pdf(junk).is_err(),
            "garbage input must return Err, never panic"
        );
    }
}
