//! Spike: compare oxidize-pdf text vs pdf_oxide text+PNG on kb-gost pages.
//!
//! ```text
//! cargo run --release --example pdf_spike --features pdf-spike -- kb-gost tmp/pdf-spike
//! ```

use anyhow::{bail, Result};
use oxidize_pdf::parser::{ParseOptions, PdfDocument as OxDoc, PdfReader};
use oxidize_pdf::text::ExtractionOptions;
use pdf_oxide::rendering::{render_page, RenderOptions};
use pdf_oxide::PdfDocument as PoDoc;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("pdf_spike error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let corpus = PathBuf::from(args.next().unwrap_or_else(|| "kb-gost".into()));
    let out_root = PathBuf::from(args.next().unwrap_or_else(|| "tmp/pdf-spike".into()));
    if !corpus.is_dir() {
        bail!("corpus dir not found: {}", corpus.display());
    }
    fs::create_dir_all(&out_root)?;

    let mut pdfs: Vec<PathBuf> = fs::read_dir(&corpus)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("pdf"))
        })
        .collect();
    pdfs.sort();
    if pdfs.is_empty() {
        bail!("no PDFs in {}", corpus.display());
    }

    let mut summary = format!(
        "# PDF spike SUMMARY\n\nCorpus: `{}`\n\n| File | pages | oxidize chars (p1) | pdf_oxide chars (p1) | PNG p1 | notes |\n|------|------:|-------------------:|---------------------:|--------|-------|\n",
        corpus.display()
    );

    for pdf in &pdfs {
        let stem = pdf
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("doc")
            .to_string();
        let doc_out = out_root.join(&stem);
        fs::create_dir_all(&doc_out)?;
        eprintln!("==> {}", pdf.display());

        let ox_pages = extract_oxidize(pdf).unwrap_or_else(|e| {
            eprintln!("  oxidize failed: {e:#}");
            Vec::new()
        });

        let po = match PoDoc::open(pdf) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("  pdf_oxide open failed: {e}");
                summary.push_str(&format!(
                    "| `{stem}` | — | {} | open-fail | — | pdf_oxide open error |\n",
                    ox_pages.first().map(|s| s.chars().count()).unwrap_or(0)
                ));
                continue;
            }
        };

        let po_pages = po.page_count().unwrap_or(0) as usize;
        let page_count = ox_pages.len().max(po_pages).max(1);

        let mut sample: Vec<usize> = vec![0];
        if page_count > 2 {
            sample.push(page_count / 2);
        }
        if page_count > 1 {
            sample.push(page_count - 1);
        }
        sample.sort_unstable();
        sample.dedup();

        let mut note = String::new();
        let mut ox_c1 = 0usize;
        let mut po_c1 = 0usize;
        let mut png_ok = true;

        let render_opts = RenderOptions::with_dpi(150);

        for &idx0 in &sample {
            let n = idx0 + 1;
            let ox_text = ox_pages.get(idx0).cloned().unwrap_or_default();
            let po_text = match po.extract_text(idx0) {
                Ok(t) => t,
                Err(e) => {
                    note.push_str(&format!("po-text p{n}: {e}; "));
                    String::new()
                }
            };
            if idx0 == 0 {
                ox_c1 = ox_text.chars().count();
                po_c1 = po_text.chars().count();
            }
            fs::write(doc_out.join(format!("p{n}.oxidize.txt")), &ox_text)?;
            fs::write(doc_out.join(format!("p{n}.pdf_oxide.txt")), &po_text)?;

            match render_page(&po, idx0, &render_opts) {
                Ok(img) => {
                    let png_path = doc_out.join(format!("p{n}.png"));
                    if let Err(e) = img.save(&png_path) {
                        png_ok = false;
                        note.push_str(&format!("png-save p{n}: {e}; "));
                    } else {
                        let bytes = fs::metadata(&png_path).map(|m| m.len()).unwrap_or(0);
                        eprintln!(
                            "  p{n}: ox={} po={} png={bytes}B",
                            ox_text.chars().count(),
                            po_text.chars().count()
                        );
                    }
                }
                Err(e) => {
                    png_ok = false;
                    note.push_str(&format!("png p{n}: {e}; "));
                    eprintln!("  p{n}: PNG failed: {e}");
                }
            }

            let ox_nums = numerals(&ox_text);
            let po_nums = numerals(&po_text);
            let missing: Vec<_> = ox_nums
                .iter()
                .filter(|n| !po_nums.contains(n))
                .take(5)
                .cloned()
                .collect();
            if !missing.is_empty() && idx0 == 0 {
                note.push_str(&format!("nums in ox not po (sample): {missing:?}; "));
            }
        }

        let note = note.trim_end_matches("; ").to_string();
        summary.push_str(&format!(
            "| `{stem}` | {page_count} | {ox_c1} | {po_c1} | {} | {} |\n",
            if png_ok { "ok" } else { "FAIL" },
            if note.is_empty() { "—" } else { &note }
        ));
    }

    summary.push_str(
        "\n## Scoring (fill manually 1–5)\n\n| File | text oxidize | text pdf_oxide | PNG | verdict |\n|------|-------------:|---------------:|----:|--------|\n",
    );
    for pdf in &pdfs {
        let stem = pdf.file_stem().and_then(|s| s.to_str()).unwrap_or("doc");
        summary.push_str(&format!("| `{stem}` |  |  |  |  |\n"));
    }
    summary.push_str(
        "\n## Recommendation\n\n_TBD after human review of `tmp/pdf-spike/**`._\n\n- Dual (oxidize text + pdf_oxide render)\n- Migrate extract to pdf_oxide\n- External render\n",
    );

    let summary_path = out_root.join("SUMMARY.md");
    fs::write(&summary_path, summary)?;
    eprintln!("Wrote {}", summary_path.display());
    Ok(())
}

fn extract_oxidize(path: &Path) -> Result<Vec<String>> {
    let bytes = fs::read(path)?;
    let reader = PdfReader::new_with_options(std::io::Cursor::new(bytes), ParseOptions::lenient())
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let doc = OxDoc::new(reader);
    let opts = ExtractionOptions {
        preserve_layout: true,
        space_threshold: 0.3,
        newline_threshold: 10.0,
        merge_hyphenated: true,
        reconstruct_paragraphs: true,
        detect_columns: true,
        include_artifacts: false,
        ..Default::default()
    };
    let pages = doc
        .extract_text_with_options(opts)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(pages.into_iter().map(|p| p.text).collect())
}

fn numerals(s: &str) -> Vec<String> {
    let re = regex::Regex::new(r"\d+(?:[.,]\d+)?").unwrap();
    re.find_iter(s).map(|m| m.as_str().to_string()).collect()
}
