# Agent workspace contract

Corpus (indexed, read-only) and **notebook** (agent notes under `.glossa/notes/`) are separate.

## Corpus

| Tools | Identifier |
|-------|------------|
| `grep`, `search`, `read`, `glob` | Indexed document path (`doc.pdf`) + `#n` for chunks |

## Notebook

| Phase | Tools | Args | Profile |
|-------|-------|------|---------|
| Create / replace / append | `note` | `doc`, `file`, `content`, `append` | editor, full |
| Browse | `ls` (list), `read` (content) | `path` from `ls` | reader, editor, full |
| Delete | `del` | `path` from `ls` | editor, full |

- **`doc`**: indexed path from grep/read; trailing `#n` is stripped server-side.
- **`file`**: e.g. `workbook.md` (research dossier), `limits.csp`.
- **`path`**: full notebook path from `ls`, e.g. `spec_2019.pdf/limits.csp`.

`.csp` files are limit tables (tab-separated rows, first line = column headers). `note` validates them on write: the reply echoes parsed columns and row count, and may add brief observations about grid shape (long headers, sentence-like cells); a malformed table (empty header cell, ragged row) is rejected without writing. Any other extension is a free-form note.

Storage: `<corpus>/.glossa/notes/<document>/…` where `<document>` is the full indexed path (with extension).

**Notes are not corpus documents, but they are indexed, and they do appear in `search`.** Two halves, both true:

- Living under `.glossa` keeps notes out of the corpus indexer's **walk**. They are never picked up as documents, never chunked like one, and never counted in the corpus.
- They are nonetheless written into the **same tantivy index**, by a separate path, as a single chunk tagged `file_type = "note"`. So a note is retrievable beside corpus documents: a `search` whose terms a note contains can return that note, and it can outrank the real document — a note is short and dense, which BM25 rewards.

This is intended. A note the agent wrote is meant to be findable later, and `search` is how it finds things. The practical consequence to keep in mind when reading results: a hit's label segment says `note`, so a note is identifiable on sight, and its path is `<document>/<file>` rather than a corpus path. Code that needs to exclude them can: `search_filtered` takes a `file_type` filter.

Write operations (`note`, `del`) are serialized across MCP editor processes via `.glossa/notebook.lock`.

## Cargo feature

```toml
default = ["notebook"]
notebook = []
constraint = ["dep:glossa-constraint", "notebook"]
```

Build without notebook: `cargo build --no-default-features`.
