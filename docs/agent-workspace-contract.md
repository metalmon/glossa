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

Storage: `<corpus>/.glossa/notes/<document>/…` where `<document>` is the full indexed path (with extension). Living under `.glossa` keeps notes out of the corpus indexer's walk — the agent can never index its own notes as documents.

Write operations (`note`, `del`) are serialized across MCP editor processes via `.glossa/notebook.lock`.

## Cargo feature

```toml
default = ["notebook"]
notebook = []
constraint = ["dep:glossa-constraint", "notebook"]
```

Build without notebook: `cargo build --no-default-features`.
