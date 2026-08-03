# kb as a CLI tool for agents

`kb` is not only an MCP server — it is a Unix-composable command-line tool your
coding agent can shell out to. Its superpower over `cat`/`grep`/`ripgrep`: it
reads **PDF and Office** (Word, Excel, PowerPoint) natively, so an agent that
can only `cat` plain text suddenly gains eyes into binary documents.

No server, no configuration, no `.glossa` for one-off reads — point it at files
and go.

## The commands

| Command | What it does | Index? |
|---------|--------------|--------|
| `kb cat <file>` | Print a file's **whole** extracted text, straight from disk. A `cat` that understands `.pdf` / `.docx` / `.xlsx` / `.pptx` / `.md`. | none |
| `kb grep <pattern> [dir]` | Ripgrep-style regex/literal search **inside** the extracted text of every document — including binary Office/PDF. | auto |
| `kb search <keywords> [dir]` | BM25-ranked keyword search over a folder (morphology-aware). Prints `#n path · snippet`. | auto |
| `kb read <target> [location]` | Read a document by path (optionally a heading / `p.N`), **or** a result number from the last `search`. Index/graph-aware. | auto |
| `kb glob <pattern> [dir]` | List documents whose **path** matches a shell glob. | auto |

### `cat` vs `read`

They look similar but have different contracts — pick by intent:

- **`kb cat <file>`** is narrow and file-only: give it a filesystem path, it
  dumps the entire extracted text. It never builds an index and never resolves
  anything but a real file. Use it to pipe a document into your agent, `grep`,
  or `head`.
- **`kb read <target>`** is the omnivorous, corpus-aware reader: a path with an
  optional `location`, a numbered hit from the last search, or (over MCP) a
  graph node id / notebook note. Use it while navigating an indexed knowledge
  base.

## Frictionless examples

```bash
# One-shot: full text of a single document (no setup, nothing left behind)
kb cat contracts/master-agreement.docx

# Search inside a folder of mixed PDFs/Word/Excel — ripgrep can't read these
kb grep "termination for convenience" contracts/
kb grep -i "\bEBITDA\b" reports/2024/         # case-insensitive, word boundary

# Rank documents by relevance, then open the top hit
kb search "data retention policy" ./policies
kb read 1                                      # opens the #1 hit from that search

# Pull just one page/section
kb read spec.pdf "p.12"
kb read handbook.docx "Onboarding"
```

Output goes to stdout with a header only when writing to a terminal; piped or
captured, it is clean text — safe to feed straight into another tool or an
agent's context window (mind the size for large documents; prefer `grep` or a
`location` to pull just what you need).

## Telling your agent to use it

Add a line like this to your agent's system prompt or project instructions so
it reaches for `kb` instead of going blind on binary documents:

> To read or search PDF/Word/Excel/PowerPoint files, use `kb`: `kb cat <file>`
> for a file's full text, `kb grep <pattern> <dir>` to search inside documents,
> `kb search <keywords> <dir>` to rank them. Plain `cat`/`grep` cannot read
> these formats.

## When to run the MCP server instead

Reach for `kb mcp` (see [mcp.md](mcp.md) and
[connect-to-agents.md](connect-to-agents.md)) when you want a **persistent,
graph-backed** corpus: ranked retrieval plus a provenance-stamped reasoning
graph the agent can query (`glossary`, `related`, `neighbors`, `path`) and extend. The CLI is the
low-friction "read these files now" path; the MCP server is the durable
"reason over this knowledge base" path. Same binary, same extractors.
