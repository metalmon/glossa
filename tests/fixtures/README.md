# Test fixtures

**Synthetic, neutral files — no real or personal data.** Generated solely to exercise the
document extractors. Each contains the ASCII marker text `glossa sample`.

- `sample.docx` — minimal valid OOXML (Content_Types + rels + a two-paragraph `word/document.xml`).
- `sample.pdf` — minimal text PDF (PDF 1.4, Helvetica, one `Tj` text object).
- `three-page-blank-middle.pdf` — three pages; page 2 is physically blank (tests blank-page indexing).
- `sample.odt` — OpenDocument text; carries **both** heading conventions (a real `<text:h>` and a
  heading rendered as a styled paragraph `Heading_20_1`, as LibreOffice/converted docs emit) plus a
  table with a column-spanned (merged) cell.
- `sample.ods` — OpenDocument spreadsheet; two sheets (`Sheet1`, `Data`) exercising
  `number-columns-repeated`: a repeated non-empty cell (must expand) and trailing repeated empty
  cells (must be clamped to the used range).
- `sample.odp` — OpenDocument presentation; two slides (`Slide1`, `Slide2`) with heading + body
  text (one carries the `glossa sample` marker), plus an embedded 1×1 PNG under `Pictures/` in the
  zip (exercises the ODF branch of `extract_zip_media`).
- `sample_chart.docx` — the minimal `sample.docx` with an added `word/charts/chart1.xml` part (a
  synthetic English bar chart: title "Sales by quarter", series "Series 1"/"Series 2", categories
  Q1–Q3, cached values). Exercises OOXML chart-data extraction. Hand-built (not Office-generated —
  real Office files carry locale boilerplate); the chart part is unreferenced, inert to the text path.
- `sample_legacy.doc` / `sample_legacy.xls` — legacy binary OLE Office (Word 97-2003 / Excel
  97-2003) each carrying one embedded 1×1 PNG picture (exercises `office_oxide::{doc,xls}::images`
  extraction wired into `read.rs`). Generated via MS Office COM with document metadata scrubbed
  (`RemoveDocumentInformation`) — no author/username. Note: native charts are NOT embedded images
  and are not extracted; `.ppt` yields no images via office_oxide (both out of scope here).

Do **not** replace these with real business/client documents — this repository is public.
To add fixtures for other formats (xlsx/pptx/doc/xls/ppt), create equally neutral synthetic
files containing the same `glossa sample` marker.

The `.odt`/`.ods` files are generated with [`odfpy`](https://pypi.org/project/odfpy/) (keeps this
repo Python-free — only the binary fixtures are committed). Regenerate via the untracked
`gen_odf_fixtures.py` generator: `py -3 -m pip install odfpy && py -3 gen_odf_fixtures.py tests/fixtures`.
