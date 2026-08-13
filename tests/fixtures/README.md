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

Do **not** replace these with real business/client documents — this repository is public.
To add fixtures for other formats (xlsx/pptx/doc/xls/ppt), create equally neutral synthetic
files containing the same `glossa sample` marker.

The `.odt`/`.ods` files are generated with [`odfpy`](https://pypi.org/project/odfpy/) (keeps this
repo Python-free — only the binary fixtures are committed). Regenerate via the untracked
`gen_odf_fixtures.py` generator: `py -3 -m pip install odfpy && py -3 gen_odf_fixtures.py tests/fixtures`.
