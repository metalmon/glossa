# Reading engineering drawings & CAD — design direction

> Status: **direction / not built.** Captures a design discussion on whether and how LLM agents
> (Qwen-family vision models) can read engineering drawings and CAD, and how glossa would support it.

## The core question

Can a model understand *what is drawn* in an engineering drawing or CAD file? The answer depends
entirely on **which representation** the data is in. It is useful to think in three layers.

### Layer 1 — Metadata (text)

Title block, technical requirements (notes), BOM/parts list, material, scale, dimension *values*.
- **Extractable as structured text.** Qwen-class models read this reliably (strong OCR, multilingual
  incl. Cyrillic).
- **The model learns identity + specs, not shape.** "This is a shaft, Ø20, L=125, steel" — from the
  *name and annotations*, not from understanding geometry.
- **Where the work is:** parse **by region** (title block → key/value, notes → list, BOM → table).
  A flat dump of the whole sheet's text is low-signal — scattered dimension numbers (`125`, `Ø20`,
  `R5`) lose meaning once divorced from the geometry, and retrieval/reasoning suffer. Sources:
  DXF entities, vector-PDF text (already extracted by glossa's PDF path), STEP AP242 PMI.

### Layer 2 — Image (visual)

A raster scan or a render of the sheet.
- Qwen reads text/values/title-block *on the image* well; **geometry from orthographic projections
  and GD&T symbology is unreliable.**
- **Where the work is:** high resolution + tiling for dense sheets; use for *reading*, not for
  geometric reasoning.

### Layer 3 — Structure (the CAD model)

B-rep (boundary representation: vertices → edges/curves → faces/surfaces + topology) plus the
feature/parametric tree (sketch → extrude → hole → fillet …). **The only layer where geometry lives
as machine-readable data.**
- From it you can programmatically derive a *semantic* description — "4 holes Ø8 on a Ø50 PCD, central
  bore Ø20 H7, two M6 threads, 100×60×20" — which a model *can* reason over, because the geometry is
  encoded as facts rather than pixels or a floating dimension bag.
- Formats: neutral **STEP (ISO 10303, AP242 carries semantic PMI/GD&T)**, IGES; native proprietary
  (`.sldprt`, `.catpart`, …) → convert to STEP; meshes (STL/OBJ) are tessellation only (no feature
  semantics).
- **Where the work is:** parsing the file is one thing; **feature recognition** (B-rep → the
  semantic feature list) is a separate, hard problem written on top. Extracting PMI/GD&T from AP242
  is tractable but still custom.

**Bottom line:** text gives *what it is and with what parameters*; *what is drawn geometrically*
needs the structure layer (heavy, custom extraction) or the image layer (for reading, not reasoning).

## Rendering — natively feasible in Rust

Unlike Office-chart rendering (blocked — the geometry is opaque without LibreOffice; see
`architecture.md` "Known extraction limitations"), **CAD/drawing rendering is not blocked**: we hold
the geometry (DXF/STEP), and Rust has mature rasterizers / 3D renderers.

- **2D drawings (DXF):** `dxf` → map entities to paths → rasterize with `tiny-skia` or via SVG
  (`usvg`/`resvg`). Light, no GPU. (DWG → convert to DXF first.)
- **3D (STEP/B-rep):** `ruststep`/`truck` parse → `truck` tessellates (CPU) → render the mesh.
- **Meshes (STL/OBJ):** `tobj`/`stl_io` → render.

No external process dependency (no LibreOffice/ETL) is required.

## Delivery: MCP Apps (SEP-1865)

The [MCP Apps extension](https://modelcontextprotocol.io/seps/1865-mcp-apps-interactive-user-interfaces-for-mcp)
(Final 2026-01-26; Anthropic + OpenAI + MCP-UI) is the mechanism to put a **visual as an interactive
artifact directly in the agent conversation**:

- Server declares a UI resource (`ui://…`, `mimeType: text/html;profile=mcp-app`); a tool references
  it via `_meta.ui.resourceUri`.
- Host renders it in a **sandboxed iframe** (`allow-scripts allow-same-origin`); data flows via
  `ui/notifications/tool-result` (`structuredContent`); the UI calls back over JSON-RPC (`tools/call`,
  `resources/read`, `ui/update-model-context`).
- **Canvas/WebGL run client-side in the iframe.**

**Key architectural consequence:** the heavy render moves to the **client**. glossa's server only
**parses + tessellates (CPU)** and ships the geometry as `structuredContent` plus a self-contained
HTML/JS viewer resource; the 2D/3D rendering (SVG/canvas or WebGL/three.js) happens in the client's
iframe. **The server never needs GPU/wgpu.**

## Decision: a glossa `cargo` feature, not a separate service

Because MCP Apps carries the artifact over the *same* MCP connection and the render is client-side,
this belongs in glossa itself — behind an **opt-in `cargo` feature** so the default build stays lean.

- **Feature contents (`cad`):** `dxf` / `ruststep` / `truck` (parse + CPU tessellation) + feature/PMI
  extraction → `structuredContent`; a **bundled, self-contained** viewer served as `ui://glossa/cad-viewer`;
  an MCP tool (`render` / `cad_view`).
- **Default build unaffected:** the parse/tessellation crates and the viewer compile only with the feature.
- **Text/metadata extraction** (Layer 1) is light enough to live near-core (another extractor producing
  searchable chunks), independent of the `cad` feature.

### Caveats

1. **Host support.** Only MCP-Apps-capable hosts render `ui://` apps. Fallback for others: a **static
   image** via the existing `read(page_image)` path (2D → CPU raster with `resvg`/`tiny-skia`; a 3D
   snapshot would need a CPU rasterizer, an optional sub-feature). Interactive where supported, static
   everywhere.
2. **CSP / offline.** External CDNs (e.g. three.js) require declaring `resourceDomains`. To keep
   glossa **offline** (its ethos), **bundle the viewer JS inline** — CSP-clean, no network.
3. **Feature recognition is custom** — no off-the-shelf crate turns B-rep into a semantic feature list.
4. STEP kernels in Rust (`truck`) are young (0.x); proprietary native formats need STEP conversion first.

## Phasing

1. **Layer 1 extractors** — DXF / STEP-PMI / vector-PDF text, region-aware chunking. Light, near-core.
2. **`cad` feature + MCP-Apps viewer** — parse + tessellate on the server, WebGL/SVG viewer client-side,
   artifact in-conversation. Static-image fallback for non-Apps hosts.
3. **Feature recognition / 3D facts** — B-rep → semantic feature list for reasoning.

## First step (before building)

Run a small benchmark on real drawings (read the title block, N dimensions, notes, BOM) and measure
accuracy — decide by evidence, not assumption.

## References

- [SEP-1865: MCP Apps](https://modelcontextprotocol.io/seps/1865-mcp-apps-interactive-user-interfaces-for-mcp)
- [MCP Apps spec (2026-01-26)](https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx)
- Rust crates: `dxf`, `ruststep`, `truck` (+ `truck-stepio`), `tiny-skia`, `usvg`/`resvg`, `tobj`, `stl_io`; `opencascade`/`opencascade-sys` (FFI) for heavier B-rep needs.
