# Source-file delivery: `get_source_file` (Glossa) + MCP-blob intake (ZeroClaw)

**Date:** 2026-07-17
**Status:** Design draft
**Depends on / extends:** ZeroClaw `feat/acp-embedded-resource-blob` (ACP embedded resource blob, inbound + outbound). This is that spec's follow-up #3 ("Glossa/MCP path that materializes KB originals for `deliver_file`"), reworked for a **distributed** topology.

## Problem

An agent should be able to hand the end user the **original source file** behind an answer (the PDF/DOCX the citation points at), so the client can preview/download it — real source attribution, not just a text quote.

The naive design ("Glossa writes the file to a path, ZeroClaw's `deliver_file` reads it") assumes the MCP server and the agent **share a filesystem**. That holds only when `kb mcp` runs as a local stdio subprocess inside the agent's workspace. It does **not** hold in the real deployment:

```
Glossa MCP (on-prem, HTTP)  ──▶  Agent / ZeroClaw (cloud)  ──▶  Client (elsewhere)
```

Three separate zones, no shared disk. A filesystem path returned by the on-prem MCP server is meaningless to the cloud agent, and the client typically cannot reach the on-prem server directly (firewall + separate auth). So the bytes **must travel in the MCP response**, and something must relay them to the client without ever putting base64 into the model's context.

## Topology and the constraint it imposes

- Glossa MCP is reached over **streamable-HTTP** (`kb mcp --transport streamable-http`, `scripts/start-mcp-http.*`).
- The only channel from on-prem to the agent is the MCP response body. → the original's **bytes cross the wire** here.
- The only channel from the agent to the client is ACP. ZeroClaw's outbound path is `deliver_file(path)`, which reads a file **from the agent's own workspace** and inlines it as ACP `resource`+`blob`.
- Therefore the file must land in the **agent's workspace** before `deliver_file` can push it — and it gets there only by the agent materializing the bytes it received over MCP.

## Data flow (target)

Bytes make two hops (on-prem → cloud → client) and appear in the model's prompt on **neither**:

```
1. get_source_file(ref)      Glossa (on-prem)  → MCP EmbeddedResource{uri,mimeType,blob} + text provenance
2. MCP-blob intake           ZeroClaw (cloud)  → decode → {workspaceDir}/uploads/<sha16>.<ext>
                                                 model sees only:  [Document: name] <abs-path>
3. deliver_file(path)         ZeroClaw (cloud)  → ACP resource+blob → client preview/download
```

Step 2 is the missing piece today (see ZeroClaw slice below).

## Goals

1. **Glossa:** a read-only MCP tool `get_source_file(ref)` that resolves a citation reference to its backing corpus file and returns the file as an MCP embedded **blob** resource plus machine-usable provenance.
2. **ZeroClaw:** catch a designated MCP tool result that carries an embedded resource blob and materialize it into `{workspaceDir}/uploads/<sha16>.<ext>`, exposing only a path marker to the model — the outbound-analog of the existing inbound `resource.blob` intake, reusing the same helper.
3. Keep base64 out of the model context on every hop.
4. Preserve source attribution: the delivered artifact is a real, citable document (original, or an explicitly-labeled page rendition).

## Non-goals

- No shared-filesystem / path-passing assumption between MCP and agent.
- No direct client → on-prem MCP fetch (firewall/auth).
- No auto-wrapping of arbitrary tool output as blobs (ZeroClaw non-goal). The intake fires only for `get_source_file`'s embedded-resource result.
- No new ACP wire capabilities beyond what `feat/acp-embedded-resource-blob` already defines.

## Glossa tool: `get_source_file`

Lives in `src/mcp.rs` + `src/tools.rs` on master. Read-only → available in **all** profiles (`reader`/`editor`/`full`).

**Input:**

| Field | Required | Notes |
|-------|----------|-------|
| `ref` | yes | A citation reference: `path` or `path#n` (chunk/page, the `[#n]` from search/grep) or a graph node id (as `read` already accepts). Resolves to the backing corpus file. |
| `max_bytes` | no | Delivery cap; default **10 MB** to match ZeroClaw's `deliver_file` limit. |

**Resolution:** reuse existing machinery — `resolve` and the `read`-by-node path already map `path#n` / node-id → the owning source chunk with provenance and page number. The tool maps that to the corpus-root-relative source file (via `root.rs`; reject any path escaping the corpus root; reject a reasoning-node ref that backs no file).

**Output:**

- One MCP `EmbeddedResource` content item, `BlobResourceContents` shape: `{ uri: <original filename>, mimeType: <guessed/known>, blob: <base64> }`. This maps 1:1 to the ACP `resource` block, so ZeroClaw's intake and `deliver_file` stay format-stable.
- One text content item — the **provenance line** the model reads: source path, page/chunk, and **exactly what was delivered** (whole file vs page rendition). The base64 is never in the text item.

> Impl note: confirm rmcp 1.8 exposes an embedded-resource (blob) `Content` variant; `page_image` already returns image content, so a resource variant is the analogous path. If unavailable, fall back to a structured JSON result carrying the same fields and have ZeroClaw's intake key off that shape.

## Behavior on files over the cap (decided)

A citation almost always points at a **specific page**, so oversize originals degrade to page scope by returning a **smaller real PDF** — never a rasterized image, so selectable text and fidelity are preserved:

1. **`original ≤ max_bytes`** → deliver the whole original. Best case: a real, fully citable document. Provenance: `delivered: whole file`.
2. **`original > max_bytes` and PDF and `ref` is page-scoped** → extract the cited page (optionally a small window, e.g. ±1 for context) into a **new PDF** and deliver that. Still a genuine, text-bearing PDF of the source page. Provenance: `delivered: page N (extracted; original exceeds cap)`.
3. **`original > max_bytes` and non-PDF** (large DOCX/XLSX) **or a whole-document ref with no page** → **no artifact**; return a structured error telling the agent to cite a specific page. Never silently send a truncated or misleading file.

Edge: if a **single** extracted page still exceeds `max_bytes` (rare — e.g. one giant embedded scan), fall back to the structured error rather than shipping something misleading. (A 200 DPI PNG render via `page_image` remains available as an explicit, separately-named last resort, but is not part of `get_source_file`'s default path — it is not the source document.)

The provenance line always states which of these happened, so the model can cite honestly (the "no silent caps" principle).

> Impl note: pdf_oxide 0.3 `DocumentEditor::extract_pages_to_bytes(&[usize]) -> Result<Vec<u8>>` (0-based page indices) yields the sliced PDF directly in memory; `extract_page_ranges_to_bytes` covers ranges. Map the cited page (our `p.N` chunk / `read`'s 1-based page) to the 0-based index. No temp files, no rasterization.

## ZeroClaw slice (cross-repo dependency): MCP-tool blob → workspace

New slice, sitting **after** `feat/acp-embedded-resource-blob` P0. P0 already handles (a) inbound `resource.blob` in the prompt and (b) `deliver_file` of an already-local file. This adds the third case: a tool **result** carrying an embedded resource blob.

- On a `ToolResult` whose content includes an `EmbeddedResource` with `blob` (gate to `get_source_file`, or a small allowlist), reuse the **inbound helper** (base64 decode, 10 MB limit, `sha256[:16]` naming, `uploads/` dir) to write `{workspaceDir}/uploads/<sha16>.<ext>`.
- Replace the blob in the model-facing result with a path marker: `[Document: <name>] <abs-path>` (image mimes → `[IMAGE:<path>]`, matching P0).
- Keep `rawOutput` to the text summary only — no base64 in context, exactly as P0 does for `deliver_file`.
- The model then calls `deliver_file(path)` (or a future fused auto-deliver) to push it to the client.

This is the outbound analog of the inbound intake and should share the same store-agnostic helper the ACP spec already plans to extract (that spec's follow-up #2).

## Error table

| Case | Response |
|------|----------|
| `ref` resolves to no corpus file (pure reasoning node, or unknown) | tool error with the valid reference forms |
| Resolved path escapes corpus root | tool error |
| Original over cap, not page-scoped/renderable | structured error: "cite a specific page" |
| (ZeroClaw) decoded blob over 10 MB | `INVALID_PARAMS` / tool error, per P0 |
| (ZeroClaw) intake write fails | tool error; no path marker |

## Testing

- **Glossa:** `ref` as `path`, `path#n`, node-id → correct backing file + page; mime guessed; provenance states delivery mode; corpus-root escape rejected; node-with-no-file rejected; over-cap PDF page ref → PNG render under cap; over-cap non-PDF → structured error.
- **ZeroClaw:** `get_source_file` result with embedded blob → file written to `uploads/<sha16>`, model result carries only the path marker, `rawOutput` has no base64; then `deliver_file(path)` emits ACP `resource`+`blob`.
- **End-to-end:** on-prem HTTP MCP → cloud agent → client receives the source with correct filename/mime.

## Dependencies and sequencing

- The Glossa tool is **unblocked** — it can be built and unit-tested against this contract now, on master.
- **End-to-end delivery is blocked on ZeroClaw**: P0 (`deliver_file` + inbound intake) must land, then the MCP-blob intake slice above. Until then Glossa's tool returns a correct blob that nothing yet relays.
- Recommended order: (1) Glossa `get_source_file` + tests; (2) ZeroClaw P0; (3) ZeroClaw MCP-blob intake slice; (4) end-to-end test on a real on-prem/cloud/client split.

## Decision log

| Topic | Choice |
|-------|--------|
| Path-passing between MCP and agent | Rejected — no shared FS in the split topology; bytes go over the MCP wire |
| Direct client → on-prem fetch | Rejected — firewall + separate auth |
| Return shape | MCP `EmbeddedResource` blob (maps 1:1 to ACP `resource`), + text provenance |
| Base64 in model context | Never — ZeroClaw materializes to `uploads/`, model sees a path marker |
| Over-cap behavior | whole file ≤ cap; else extract cited page(s) as a new PDF (text preserved, via `extract_pages_to_bytes`); else structured error. No PNG rasterization on the default path. |
| Cap | `max_bytes`, default 10 MB (matches ZeroClaw `deliver_file`) |
| Ownership | Glossa tool on master; ZeroClaw intake as a slice after P0 |
