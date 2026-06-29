# ZeroClaw + glossa

[ZeroClaw](https://github.com/zeroclaw-labs/zeroclaw) is a local Rust agent runtime (channels, tools, MCP client). glossa adds a **file-first knowledge base** over your documents — complementary to ZeroClaw’s shell, browser, and channel tools.

## Prerequisites

1. Install `kb` — [install.md](../install.md)
2. Index your document folder — `kb index /path/to/my-documents`
3. ZeroClaw installed and configured (`~/.zeroclaw/config.toml`)

## Recommended: stdio MCP

ZeroClaw spawns glossa as a subprocess. This matches glossa’s default stdio transport and needs no open port.

Edit `~/.zeroclaw/config.toml`:

```toml
[mcp]
enabled = true
deferred_loading = true

[[mcp.servers]]
name = "glossa"
transport = "stdio"
command = "/usr/local/bin/kb"   # Linux/macOS — adjust to your install path
args = [
  "mcp",
  "/path/to/my-documents",
  "--profile",
  "reader",
  "--transport",
  "stdio",
]
```

**Windows** — set `command` to the full path of `kb.exe`:

```toml
command = "C:\\Program Files\\glossa\\glossa-1.0.0-x86_64-pc-windows-msvc\\kb.exe"
args = ["mcp", "C:\\Users\\you\\Documents\\my-kb", "--profile", "reader", "--transport", "stdio"]
```

Restart ZeroClaw after saving.

Upstream reference: [ZeroClaw MCP setup](https://github.com/zeroclaw-labs/zeroclaw/blob/v0.6.9/docs/setup-guides/mcp-setup.md).

## Tool names

ZeroClaw prefixes tools with the server name. glossa tools appear as:

| glossa tool | ZeroClaw name |
|-------------|---------------|
| `search` | `glossa__search` |
| `read` | `glossa__read` |
| `grep` | `glossa__grep` |
| `glob` | `glossa__glob` |
| `glossary` | `glossa__glossary` |
| `neighbors` | `glossa__neighbors` |
| `resolve` | `glossa__resolve` |

With **`reader`** profile, write tools (`graph_upsert`, `index`, …) are hidden from the tool list.

## Auto-approve read tools

ZeroClaw may prompt before each MCP tool call. To allow read-only glossa tools without prompts, add prefixes to `[autonomy]`:

```toml
[autonomy]
auto_approve = [
  "glossa__search",
  "glossa__read",
  "glossa__grep",
  "glossa__glob",
  "glossa__glossary",
  "glossa__neighbors",
  "glossa__resolve",
]
```

Use **`editor`** profile and approve write tools only if you trust the agent to modify the reasoning graph:

```toml
args = ["mcp", "/path/to/my-documents", "--profile", "editor", "--transport", "stdio"]
```

## Editor profile + graph enrichment

If ZeroClaw should maintain a support-style reasoning graph (`Symptom` → `Cause` → `Resolution`), use `editor` and optionally copy a domain ontology:

```bash
mkdir -p /path/to/my-documents/.glossa
cp eval/ontology-support.toml /path/to/my-documents/.glossa/ontology.toml
kb index /path/to/my-documents
```

See [graph-and-ontology.md](../graph-and-ontology.md).

## HTTP alternative (optional)

If glossa runs as a local HTTP service ([deploy/service.md](../deploy/service.md)), some ZeroClaw builds support remote MCP via `http` or `sse`:

```toml
[[mcp.servers]]
name = "glossa"
transport = "http"
url = "http://127.0.0.1:8080/mcp"
```

glossa speaks **streamable-http** at `/mcp`. Verify compatibility with your ZeroClaw version; **stdio is the supported path** when in doubt.

Install the service:

- Linux: [deploy/ansible/README.md](../../deploy/ansible/README.md)
- Windows: [deploy/windows/README.md](../../deploy/windows/README.md)
- macOS: [deploy/macos/README.md](../../deploy/macos/README.md)

## What glossa gives ZeroClaw

- Offline search over PDF, Office, Markdown, and text in a folder
- Exact `grep` for codes and version strings
- Optional reasoning graph (`glossary`, `neighbors`) without embeddings or cloud vector DB
- Auto-indexing when files on disk change

ZeroClaw keeps handling channels (Telegram, Discord, …), model routing, and non-KB tools.

## Related

- [connect-to-agents.md](../connect-to-agents.md) — general MCP setup
- [mcp.md](../mcp.md) — glossa tool reference
- [install.md](../install.md) — release install
