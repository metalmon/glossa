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
- `sample_chart_anchored.docx` — hand-built FROM `sample_chart.docx` (same body paragraphs, the
  same `word/charts/chart1.xml` verbatim), with the anchoring that one deliberately lacks: a real
  `<w:drawing>` → `<a:graphicData>` → `<c:chart r:id="rId10"/>` and the
  `word/_rels/document.xml.rels` chain that resolves it. Because the chart is actually referenced,
  `office_oxide` 0.1.17 hoists its text into the body flow as plain paragraphs — so this is the
  fixture that exercises the docx de-duplication in `src/extract/office.rs`
  (`drop_hoisted_docx_chart_text`); `sample_chart.docx`'s unreferenced part cannot reach that path
  at all. It also carries a SmartArt diagram (`<dgm:relIds r:dm="rId11"/>` →
  `word/diagrams/data1.xml`, nodes "Diagram node alpha"/"beta") whose text rides the SAME converter
  path with no chart chunk of ours to replace it — the guard that the de-duplication must not eat
  it. Pure ASCII, built by an untracked scratchpad script, never Office-generated.
- `sample_chart.ods` — the minimal `sample.ods` with an injected `Object 1/` embedded chart
  sub-document (`chart:bar`, title "Sales by quarter", a LOCAL `<table:table>` data cache: Series
  1/2, Q1–Q3). Exercises ODF chart-data extraction. Hand-built English; mirrors how LibreOffice/odfpy
  embed ODF charts (local data table), the primary ODF-chart form.
- `sample_chart.odt` / `sample_chart.odp` — the minimal `sample.odt`/`sample.odp` with the same
  injected `Object 1/` embedded chart (local `<table:table>`, "Sales by quarter") — confirms ODF
  chart extraction is container-agnostic (odt/ods/odp all via the `Object*/content.xml` scan).
- `sample_chart_ref.ods` — an ODS whose embedded chart (`Object 1/`) references SHEET CELLS
  (`chart:values-cell-range-address="Sheet1.$B$2:.$B$4"` etc.) instead of embedding a local data
  table — the Excel→ODS export form. Sheet1 holds the data (Cat / Series 1; Q1–Q3; 4.3/2.5/3.5).
  Exercises ref-only chart resolution (resolve the ranges against the sheet). English, hand-built.
- `sample.xlsb` — Excel Binary Workbook (BIFF12): one sheet `Readings` with the marker in A1 and a
  two-column English table (`inlet pressure` 42 / `outlet pressure` 97). Exercises the `.xlsb`
  branch, which `office_oxide` decodes into the same model as `.xlsx`. Generated via Excel COM with
  `RemoveDocumentInformation`, then scrubbed further than that API reaches: `docProps/core.xml` was
  rewritten with a neutral creator and fixed dates, the localized theme/heading labels in the XML
  parts were replaced with their English equivalents, and the built-in cell-style name inside the
  binary `xl/styles.bin` was patched **length-preserving** (a BIFF12 record carries its own byte
  length, so a shorter replacement would corrupt every following offset). Verified afterwards by
  scanning every part for Cyrillic in UTF-8 *and* UTF-16LE, and by reopening the result in Excel.
- `sample_chart_anchored.ods` — hand-built FROM `sample_chart.ods` (same lineage, same
  `Object 1/` local-table chart data), with one addition: a `<draw:frame><draw:object
  xlink:href="./Object 1"/></draw:frame>` placed inside `Sheet1`'s own `<table:table>`, actually
  referencing the embedded chart. This is the referenced counterpart to `sample_chart.ods`'s
  deliberately unreferenced chart part — the contrast between the two is the point of having
  both: `sample_chart.ods`'s chart part sits in the zip with nothing in the body pointing at it
  (an orphan chunk, `Chunk::labelled`), while this one is anchored, so its chart chunk's location
  resolves to the sheet it's anchored in and renders as `Sheet1 > Sales by quarter`
  (`Chunk::nested`) instead of a bare label. Exercises the hand-built-style href spelling
  (leading `./`, no trailing slash) as opposed to Office's own `"Object 1/"` spelling. Its chart
  being anchored in the FIRST of two sheets is also what lets it witness chart placement: the
  chart chunk lands between the two sheets' chunks rather than after both.
- `sample_chart_multi_sheet.ods` — hand-built ODS with three sheets (`Sheet1`, `Middle`, `Last`)
  and a chart anchored in two DIFFERENT ones, `Sheet1` (`Object 1/`, "First sheet chart") and
  `Last` (`Object 2/`, "Last sheet chart"). The only fixture with charts on more than one sheet,
  which is the case that catches a chunk-span lookup ignoring the section index: every sheet's
  table is element 0 of its own section, so ODS element indices repeat across sheets while chunk
  indices do not. The chart on the LAST sheet is deliberate — that is where "placed after its own
  sheet" and "appended at the end" would otherwise coincide. Pure ASCII, built by an untracked
  scratchpad script, never Office-generated.
- `sample_chart.xlsx` — hand-built xlsx: one sheet (`Sheet1`, carrying the `glossa sample`
  marker plus a small Q1-Q3 data table) and a chart part at `xl/charts/chart1.xml`, with the
  conventional rels chain to it (`xl/worksheets/_rels/sheet1.xml.rels` ->
  `xl/drawings/drawing1.xml` -> `xl/charts/chart1.xml`) for fidelity to a real xlsx package —
  neither extraction pass below actually reads it; both find the chart part by a flat scan of
  the zip's part names. Title "Sales by quarter", same series/categories/cached values as
  `sample_chart.docx`'s chart. Exercises both `office_oxide`'s own xlsx chart-text pass (which
  appends a synthetic "Charts" section to its IR) and our `ooxml_chart` OOXML chart-data
  extraction — and the de-duplication between them (`src/extract/office.rs`'s
  `synthetic_xlsx_chart_section_chart_count` and `xlsx_chart_part_names`).
- `sample_chart.pptx` — the repository's **only** pptx fixture, hand-built (never
  PowerPoint-generated). One slide, three things deliberately: ordinary body text, a REFERENCED
  chart (`<p:graphicFrame>` → `<a:graphicData uri=…/chart>` → `<c:chart r:id="rId2"/>` plus the
  `ppt/slides/_rels/slide1.xml.rels` chain that resolves it to `ppt/charts/chart1.xml`), and a
  SmartArt diagram (`<dgm:relIds r:dm="rId3"/>` → `ppt/diagrams/data1.xml`, nodes "Deck diagram
  node one"/"two"). Both graphics travel the SAME converter path — `office_oxide` 0.1.17 flattens
  each into `GraphicContent::Text` (`convert_pptx.rs:538`) with no way to tell them apart — which
  is why the diagram is here: it is the guard any de-duplication must not eat, since a diagram has
  no chart chunk of ours to replace it. The chart's title ("Deck chart title"), its series name
  ("Readings") and its values (77.7 / 88.8 / 99.9) appear nowhere in the slide's body text, so a
  test can count how many times the chart's data reached the index. It reaches it **twice today**
  — the pptx duplication defect that `src/extract/office.rs`'s xlsx- and docx-gated
  de-duplication does not cover. Pure ASCII, built by an untracked scratchpad script.
  One limit to know before reusing it: its three body text shapes all carry the same
  `<a:off y="365125">`, so the converter groups them ahead of both graphics and the emitted chunk
  does **not** read in slide order. That is fine for counting duplication, which is what this
  fixture exists for, but a task that wants to witness pptx chart *placement* must give those
  shapes distinct y offsets first.
- `sample_chart_legacy.rtf` — hand-built RTF (pure ASCII, 6.4 KB) carrying an OLE1-embedded
  legacy `MSGraph.Chart.8` object: `{\*\objdata <hex>}` → OLE1 native header → compound file →
  `Workbook` stream (in the *mini* stream) → BIFF globals + chart substreams, the datasheet
  written as `0x0004` LABEL / `0x0003` NUMBER records in the short BIFF2-shaped layout MSGraph
  actually emits. Grid: Qtr1–Qtr4 × Apples/Pears/Plums. Exercises `src/extract/rtf_chart.rs`.
  Hand-built, **not** Office-generated: a Word-produced version of this exact file carries
  non-ASCII text in three separate layers (datasheet cells, series-name records, the EPRINT
  metafile) plus 27 localized style names, none of which the chart path needs.
- `sample_legacy.doc` / `sample_legacy.xls` — legacy binary OLE Office (Word 97-2003 / Excel
  97-2003) each carrying one embedded 1×1 PNG picture (exercises `office_oxide::{doc,xls}::images`
  extraction wired into `read.rs`). Generated via MS Office COM with document metadata scrubbed
  (`RemoveDocumentInformation`) — no author/username. Note: native charts are NOT embedded images
  and are not extracted; `.ppt` yields no images via office_oxide (both out of scope here).

Do **not** replace these with real business/client documents — this repository is public.
To add fixtures for the formats still without one (`.ppt`, a chartless `.pptx`), create equally
neutral synthetic files containing the same `glossa sample` marker.

The `.odt`/`.ods` files are generated with [`odfpy`](https://pypi.org/project/odfpy/) (keeps this
repo Python-free — only the binary fixtures are committed). Regenerate via the untracked
`gen_odf_fixtures.py` generator: `py -3 -m pip install odfpy && py -3 gen_odf_fixtures.py tests/fixtures`.
