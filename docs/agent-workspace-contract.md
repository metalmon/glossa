# Agent workspace contract

Corpus (indexed, read-only) and **notebook** (agent notes under `.glossa/notes/`) are separate.

## Corpus

| Tools | Identifier |
|-------|------------|
| `grep`, `search`, `read`, `glob` | Indexed document path (`doc.pdf`) + `#n` for chunks |

## Notebook

| Phase | Tools | Args | Profile |
|-------|-------|------|---------|
| Create / replace | `note` | `doc`, `file`, `content` | editor, full |
| Browse | `ls`, `cat` | `path` from `ls` | reader, editor, full |
| Edit / delete | `sed`, `del` | `path` from `ls` | editor, full |

- **`doc`**: indexed path from grep/read; trailing `#n` is stripped server-side.
- **`file`**: e.g. `parameters.md`, `limits.csp`.
- **`path`**: full notebook path from `ls`, e.g. `gost_r_57978-2017.pdf/limits.csp`.

`.csp` files are limit tables (`;`-separated CSV, first line = column headers). `note`/`sed` validate them on write: the reply echoes the parsed columns and row count; a malformed table (empty header cell, ragged row) is rejected without writing. Any other extension is a free-form note.

Storage: `<corpus>/.glossa/notes/<document>/…` where `<document>` is the full indexed path (with extension). Living under `.glossa` keeps notes out of the corpus indexer's walk — the agent can never index its own notes as documents.

Write operations (`note`, `sed`, `del`) are serialized across MCP editor processes via `.glossa/notebook.lock`.

## Constraint eval

`kb-eval-constraint` uses a temp copy of `.glossa` (or `--keep-agent-dir`). Corpus reads use `--kb`; notes write to the agent copy. Default `--tables-only` scores table coverage vs `kb-val-gost`; use `--full-pipeline` for compile + CSP.

## Cargo feature

```toml
default = ["notebook"]
notebook = []
constraint = ["dep:glossa-constraint", "notebook"]
```

Build without notebook: `cargo build --no-default-features`.
