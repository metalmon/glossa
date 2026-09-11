# CLI reference

Complete flag-by-flag reference for the `kb` binary, verified against `src/main.rs`. For a
task-framed introduction to `kb` as an agent-shell tool (5 commands, with usage patterns), see
[cli-for-agents.md](cli-for-agents.md) — this page is the complete, flag-level reference for every
subcommand.

`kbx` (`build`/`reason`/`train`/`distil`/`eval`) is a **separate** developer/benchmark binary built
from source only — it is not part of the end-user surface and is not documented here. See
[eval-and-training.md](eval-and-training.md).

`kb` auto-indexes on demand: every read path (`search`, `read`, `grep`, `glob`, and every MCP tool)
calls `ensure_fresh` first, so an explicit `kb index` is optional. Running it once up front is
still recommended before heavy agent use, so the first query isn't the one paying for indexing.

## Global flags

These apply to every subcommand (clap `global = true`):

| Flag | Env | Meaning |
|------|-----|---------|
| `--root [LABEL=]PATH` (repeatable) | `GLOSSA_ROOTS` (newline-separated) | Corpus source folder(s). Both a bare `--root PATH` and a positional `PATH` auto-label from the basename (`LABEL=PATH` sets the label explicitly instead). Give neither — run from inside the corpus and let `.glossa/` be found by walking up — and keys come out label-free. An explicit `--root` flag list wins outright over `GLOSSA_ROOTS` — never merged. See [configuration.md § Corpus roots and document keys](configuration.md#corpus-roots-and-document-keys) for how this shapes document keys. |
| `--state-dir <PATH>` | `GLOSSA_STATE_DIR` | Local directory holding `.glossa/` state. Defaults to the (sole) corpus root. Point at local disk when the corpus is a network share. |
| `--config <PATH>` | `GLOSSA_CONFIG` | TOML deployment config file — see [configuration.md](configuration.md). Flags and env vars override its settings per-key. |

`kb --version` / `kb <subcommand> --help` are always available (clap-generated).

## `kb search` — BM25-ranked keyword search

```
kb search <pattern> [path] [flags]
```

| Flag | Meaning |
|------|---------|
| `-i, --ignore-case` | Case-insensitive (rg `-i`) |
| `-w, --word-regexp` | Match whole words (rg `-w`) |
| `-F, --fixed-strings` | Treat pattern as a literal string (rg `-F`) |
| `-g, --glob <GLOB>` | Only search paths matching GLOB (rg `-g`) |
| `-t, --type <TYPE>` | Only this file type, e.g. `pdf` |
| `--scope <DOC-OR-GLOB>` | Restrict to one document or path-glob; ANDed with `--glob` |
| `-l, --limit <N>` | Max hits (default `100`) |
| `-s, --scan` | Literal ripgrep-regex scan of raw files instead of the BM25 index (slow, not stemmed) |
| `-u, --no-ignore` | Disable `.gitignore`/`.ignore`/hidden filtering |
| `-f, --format <auto\|rg\|pretty>` | Output style (default `auto`: pretty in a terminal, rg when piped) |

## `kb read` — read a document or a search result

```
kb read <target> [location]
```

`target` is a file path, or a number referencing the last `search`'s Nth result. `location` is an
optional heading or `p.N` to narrow to one section/page.

## `kb cat` — print a file's full extracted text

```
kb cat <target>
```

Reads the file directly — no index, no `.glossa`. A `cat` that understands PDF and Office.

## `kb index` — update the index

```
kb index [path] [flags]
```

| Flag | Meaning |
|------|---------|
| `--force` | Full rebuild from scratch. Also the **only** pass that (re)builds the answer-grounding DF sidecar (`.glossa/df`) — run after large corpus changes to keep rarity counts accurate. |
| `--file <REL>` | Reindex just that one document (picks up an in-place edit) |
| `--ontology <NAME>` | Materialize a baked ontology preset before indexing (see `kb ontology list`) |

No flags: incremental over the whole corpus.

## `kb prune` — remove orphaned notebook notes

*(requires the `notebook` cargo feature, on by default)*

```
kb prune [path] [--dry-run]
```

Deletes notebook notes whose owner document no longer exists in the corpus. `--dry-run` lists
without touching anything.

## `kb grep` — exact/regex search over extracted text

```
kb grep <pattern> [path] [flags]
```

| Flag | Meaning |
|------|---------|
| `-i, --ignore-case` | Case-insensitive |
| `-F, --fixed-strings` | Literal string, not regex |
| `-w, --word-regexp` | Whole-word match |
| `-g, --glob <GLOB>` | Only paths matching GLOB |
| `-t, --type <TYPE>` | Only this file type |
| `--scope <DOC-OR-GLOB>` | Restrict to one document or path-glob; ANDed with `--glob` |
| `-A, --after <N>` | N context lines after each match |
| `-B, --before <N>` | N context lines before each match |
| `-C, --context <N>` | N context lines before AND after each match |
| `-o, --only-matching` | Print only the matched substrings |
| `-n, --line-number` | Prefix each line with its chunk line number |
| `-c, --count` | Print only a match count per chunk |
| `-m, --max-count <N>` | Stop after N matching lines per chunk |
| `-U, --multiline` | Let the pattern span lines |

## `kb glob` — list documents by path pattern

```
kb glob <pattern> [path]
```

Matches document **paths**, not content — e.g. `kb glob '*.pdf'`. For content search use `search`
or `grep`.

## `kb graph` — inspect the knowledge graph

```
kb graph <action> ...
```

| Action | Purpose · positional · key flags |
|--------|-----------------------------------|
| `stats` | Node/edge counts. `[path]` |
| `glossary` (aliases `search`, `find`) | Concept → reasoning chain (the `glossary` MCP tool). `<query> [path]`; `--as-of <DATE>`; `--scope <DOC-OR-GLOB>` |
| `query` | Read-only SQL `SELECT` over the graph (the `sql` MCP tool); empty SQL prints the schema. `[sql] [path]` |
| `ls` | Browse nodes: per-type summary, or `--type T` to list that type. `[path]`; `-t/--type <TYPE>`; `-l/--limit <N>` (default `50`); `--as-of <DATE>`; `--now <DATE>` |
| `generalize` | Run the deterministic derived-layer pass (closure, `SIMILAR`, communities, centrality). `[path]`; `-m/--merge` also collapses near-duplicate nodes (destructive) |
| `doctor` | Diagnose graph health: ungrounded/stale/incomplete/dangling. `[path]`; `--prune-incomplete`; `--prune-ungrounded`; `--prune-dangling`; `--prune-stale`; `--force` (override the mass-wipe guard on `--prune-dangling`; CLI-only, not exposed over MCP); `--relink` (non-destructive: re-points `MENTIONS`+provenance for documents that were relabeled or moved between folders, matched by filename+section; backs up `graph.sqlite` first, along with its `-wal`/`-shm` siblings when present, each run's backup timestamped (`graph.sqlite.pre-relink-<epoch-seconds>`) so repeated `--relink` runs never overwrite an earlier run's backup — see [graph-lifecycle.md](graph-lifecycle.md#you-relabeled-the-corpus-or-moved-a-document-between-folders)). `--prune-ungrounded` refuses while relinkable nodes exist — run `--relink` first, or `--force` to prune anyway |
| `near` (alias `neighbors`) | Nodes reachable from a node id. `<node_id> [path]`; `-d/--depth <N>` (default `1`); `-t/--type <TYPE>` (repeatable); `--as-of <DATE>`; `--now <DATE>`; `--scope <DOC-OR-GLOB>` |
| `node` | Show one node: type, label, provenance, outgoing edges. `<node_id> [path]`; `--as-of <DATE>`; `--now <DATE>` |
| `reach` | Cross-document reasoning bridge (the `reach` MCP tool). `--from <ID>`; `-r/--relation <REL>`; `--to <ID>` (omit for discovery); `[path]`; `--no-bridge`; `-d/--max-depth <N>` (default `6`); `--scope <DOC-OR-GLOB>` |
| `dump` | Dump nodes (optionally filtered) with outgoing edges. `[path]`; `-t/--type <TYPE>`; `-f/--format <text\|json\|dot\|graphml\|html>` (default `text`); `--as-of <DATE>`; `--now <DATE>` |
| `import` | Import a graph JSON file. `<file> <path>`; `-f/--format <FMT>`; `--mode <merge\|replace>` (default `merge`) |
| `prune` | Delete all nodes of a type (and touching edges). `<path>`; `-t/--type <TYPE>` (repeatable, required); `--source <SUBSTRING>` (only nodes grounded in a matching document path); `--dry-run` |
| `build` | *(requires `constraint` feature)* Compile a document's `.csp` limit tables into the constraint graph. `[path]`; `--doc <PATH>` (required); `--tables-dir <DIR>` |

`kb graph reach` replaces the older `path` command; `--to` + `--no-bridge` with no `--relation`
reproduces a plain shortest-path lookup. Full workflow and semantics:
[graph-and-ontology.md](graph-and-ontology.md), [graph-lifecycle.md](graph-lifecycle.md).

## `kb ontology` — browse and apply baked presets

```
kb ontology <action> ...
```

| Action | Purpose |
|--------|---------|
| `list [--family F] [--tier N]` | List the preset catalog (grouped by tier, then family) |
| `show <name>` | Print a preset's TOML (accepts a name or alias) |
| `init [path] -t/--template <NAME> [--force]` | Materialize a preset to `<path>/.glossa/ontology.toml` (no indexing). `--template` is required |
| `suggest <text...>` | Rank presets against a free-text description of your documents (offline, no model call) |

Catalog, aliases, and per-preset details: [ontology-presets.md](ontology-presets.md).

## `kb mcp` — run the MCP server (or a related subcommand)

```
kb mcp [path] [flags]
kb mcp dump-tz-tools [-d/--config-dir <DIR>]
```

| Flag | Env | Meaning |
|------|-----|---------|
| `-p, --profile <NAME>` | | Tool profile: `reader` \| `editor` \| `full`. Default `editor`. |
| `-t, --trace` | | Log every tool call to `<root>/.glossa/traces/*.jsonl` (for the eval harness) |
| `-G, --no-graph` | | Expose only `search`+`read` (graph/index/admin tools hidden) — eval control arm |
| `--vision` | `GLOSSA_VISION` | Enable image output in `read` (embedded figures, `page_image`), served as JPEG. Off by default — a figure-heavy page's base64 payload can overflow the stdio JSON-RPC frame. Safe on `--transport streamable-http`. |
| `--source-file` | `GLOSSA_SOURCE_FILE` | Enable the `get_source_file` tool (delivers the original file behind a citation). Off by default — many clients can't consume the returned resource. |
| `--transport <stdio\|streamable-http>` | `GLOSSA_MCP_TRANSPORT` | Default `stdio`. |
| `--bind <ADDR>` | `GLOSSA_MCP_BIND` | Bind address for `streamable-http`. Default `127.0.0.1:8080`. |
| `--allowed-host <HOST>` (repeatable) | | Extra allowed `Host` header value(s) for `streamable-http` (DNS-rebind guard). Default permits loopback only. |
| `--auth-token <TOKEN>` | `GLOSSA_MCP_TOKEN` | Bearer token guarding `/mcp`. Unset → unauthenticated (loopback-only default). Ignored for `stdio`. |
| `--insecure <true\|false>` | `GLOSSA_MCP_INSECURE` | Override the non-loopback+no-auth startup refusal. Takes a value — bare `--insecure` is a parse error. Logs a loud warning + audit event. |
| `--tls-cert <PATH>` *(`tls` feature)* | `GLOSSA_TLS_CERT` | PEM certificate chain for native TLS. |
| `--tls-key <PATH>` *(`tls` feature)* | `GLOSSA_TLS_KEY` | PEM private key matching `--tls-cert`. |
| `--tls-client-ca <PATH>` *(`tls` feature)* | `GLOSSA_TLS_CLIENT_CA` | PEM client-CA enabling mTLS (client certs required and verified). |
| `--session-idle-secs <N>` | `GLOSSA_MCP_SESSION_IDLE_SECS` | Idle-session timeout for `streamable-http`; `0` (default) disables it. |
| `-N, --noimage` | `GLOSSA_NO_IMAGE` | **Deprecated no-op** — images are already off by default; use `--vision` to enable them. |

`kb mcp dump-tz-tools -d <config_dir>` regenerates TensorZero tool config from the live MCP tool
definitions (default `config_dir`: `eval/tensorzero/config`); equivalent to `just tools`.

Full tool table, profiles, and transports: [mcp.md](mcp.md). Deployment topology, TLS, systemd,
Windows service: [deploy/mcp-server.md](deploy/mcp-server.md). Every env var and the `--config`
file format: [configuration.md](configuration.md).

## See also

- [concepts.md](concepts.md) — the model behind roots, state-dir, scope, profiles
- [configuration.md](configuration.md) — every environment variable and config-file key
- [troubleshooting.md](troubleshooting.md) — common CLI/MCP problems
