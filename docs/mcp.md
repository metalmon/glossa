# MCP server

glossa exposes its knowledge-base tools via the [Model Context Protocol](https://modelcontextprotocol.io/). Run:

```bash
kb mcp ./my-corpus --profile reader
```

Default transport is **stdio** (JSON-RPC on stdin/stdout). For network deployment see [deploy/mcp-server.md](deploy/mcp-server.md). Install without building: [install.md](install.md).

## Profiles

Profiles control which tools are visible. They are **not** RBAC — all instances can refresh the index; profiles only hide write/admin tools from the tool list.

| Profile | Typical use | Disabled tools |
|---------|-------------|----------------|
| `reader` | Answer agents | `index`, the graph-authoring tools (`graph_upsert`/`graph_delete`/`graph_update`/`graph_generalize`/`graph_build`), the graph-admin tools (`graph_doctor`/`graph_stats`), `purge`, `note`, `del`, and the low-level graph-navigation tools `neighbors`, `related`, `resolve`, `constraint_solve` |
| `editor` | Index + graph + notebook write | `purge` |
| `full` | Admin | (none) |

The profile flag has a **default**: `kb mcp <path>` with no `--profile` starts **`editor`**, which is write-capable (index, graph authoring, notebook writes). Pass `--profile reader` explicitly for an answer-only server.

Two of the tools reader hides are not writes. `graph_doctor` is report-only unless a `prune_*` flag is set (see its row below), and `graph_stats` only counts nodes and edges. They are withheld from reader because a lean answer tool set reads better, not because they mutate anything.

Reader's tool set is deliberately lean — a weak answer model chooses better from fewer tools. It keeps the reasoning-graph entry points it actually uses (`glossary`, `reach`, `sql`) plus corpus search/read, `get_ontology`, and **notebook read** (`ls` to list notes; note content is read with the regular `read` tool). The low-level navigation tools (`neighbors`, `related`, `resolve`) that a reader rarely reaches for are withheld — their value is already surfaced in `glossary`'s composed section. The full reader list is `get_ontology`, `glob`, `glossary`, `grep`, `ls`, `reach`, `read`, `search`, `sql` (plus `get_source_file` and `verify` when their gates are open).

`--no-graph` closes the graph gate. It is narrower than "search + read only" in two directions, so do not treat it as a read-only switch:

- **It still leaves nine tools standing** on `editor`/`full` — `search`, `read`, `grep`, `glob`, `get_ontology`, `graph_stats`, and the notebook tools `ls`, `note`, `del`. The notebook **writes** survive, so an eval control arm built on `--no-graph` alone is not read-only; pair it with `--no-default-features` (no notebook) or `--profile reader` for that.
- **`graph_stats` is registered with no gate at all**, so it survives `--no-graph` and still answers from the graph. The graph is not unqueryable, only mostly unreachable.
- It also **removes `purge` from `full`**, because `purge` carries the graph gate. The "Disabled tools: (none)" row above describes `full` without `--no-graph`.

What `--no-graph` does hide: `index`, `resolve`, `neighbors`, `related`, `glossary`, `reach`, `sql`, `purge`, `graph_upsert`, `graph_delete`, `graph_update`, `graph_generalize` and `graph_doctor`. Not `graph_stats` (no gate), and not `graph_build` or `constraint_solve`, which are gated on the `constraint` build feature instead.

Image output is **off by default** and opt-in via `--vision` (or `GLOSSA_VISION=1`). Without it, `read` strips `page_image` and `include_images` from its schema and never returns `Content::image`. This is off by default because a figure-heavy page's base64 image payload can overflow the stdio JSON-RPC frame and drop the connection; enabling `--vision` is safe on `--transport streamable-http`. When enabled, all images are served as JPEG (embedded PNGs re-encoded, real JPEGs passed through). `-N` / `--noimage` is **not** a no-op: it is a veto over `--vision`. `--vision --noimage` leaves images off. It is also bound to an environment variable, `GLOSSA_NO_IMAGE`, and hidden from `kb mcp --help`, so a leftover `GLOSSA_NO_IMAGE` in a service environment silently disables `--vision` with nothing in the startup output to say so. If `--vision` appears to do nothing, check that variable first.

Original-file delivery (`get_source_file`) is likewise **off by default** and opt-in via `--source-file` (or `GLOSSA_SOURCE_FILE=1`). Many clients can't consume the returned file resource, so the tool is withheld unless a client that uses it opts in.

`get_ontology` is available in every profile. `resolve` and `constraint_solve` are editor/full only (withheld from reader). `constraint_solve` and `graph_build` are only registered when the binary is built with `--features constraint` — without it, these tools are absent from the tool list.

## Tools

| Tool | reader | editor | full | Purpose |
|------|:------:|:------:|:----:|---------|
| `search` | ✓ | ✓ | ✓ | BM25 keyword search; each hit is one line, `path#n · heading · snippet` (the heading segment is omitted when the chunk has none). The leading `path#n` is the citation token — copy it exactly; it is what `read` and a node's `source_path` take. An index that cannot be opened or searched answers with **`isError: true`**; zero hits is a normal answer (see [Failures vs non-applicable](#failures-vs-non-applicable)) |
| `grep` | ✓ | ✓ | ✓ | Regex/literal over extracted text. An index walk that fails answers with **`isError: true`**; zero matches, and a pattern the regex engine refuses (the reply says to set `fixed:true`), are normal answers |
| `glob` | ✓ | ✓ | ✓ | List documents by path glob. An index walk that fails answers with **`isError: true`**; no matching document, and a glob the matcher refuses (the reply gives the syntax), are normal answers |
| `read` | ✓ | ✓ | ✓ | Read chunk `#n`, graph node evidence, or a notebook note (path from `ls`). Image output is off unless the server was started with `--vision`; then `page_image: true` returns PDF page `n` as a 200 DPI JPEG for vision models, and embedded/`<img>` images are returned alongside the text (all as JPEG). Without `--vision`, the image params are removed from the schema and responses are text only. A `page_image` call on a PDF page that will not render answers with **`isError: true`** — and carries the chunk's text in that same response, since the text comes from the index rather than the file (see [Failures vs non-applicable](#failures-vs-non-applicable)) |
| `get_source_file` | ✓† | ✓† | ✓† | Deliver the original source file behind a citation (`path`, PDF page `n`) as an embedded resource blob for the client to preview/download — for source attribution, not reading. Whole file when ≤ cap (default 10 MB); a larger PDF returns just the cited page as its own PDF. A file the index has and the disk cannot produce answers with **`isError: true`** (see [Failures vs non-applicable](#failures-vs-non-applicable)). **† Off by default; opt in with `--source-file`** |
| `glossary` | ✓ | ✓ | ✓ | Resolve concept → reasoning chain + anchors. **Both `name` and `query` are required** — `name` is the concept, `query` is the full question, and the composed neighborhood is ranked by what that question needs. A call with `name` alone fails deserialization |
| `reach` | ✓ | ✓ | ✓ | Cross-document reasoning bridge — walk a relation from a node (discovery), or add a candidate to verify the connection. Replaces the old `path` tool (`bridge: false` = in-document connectivity only) |
| `sql` | ✓ | ✓ | ✓ | Read-only SQL SELECT over the reasoning graph — for a ranking or extreme ('which / earliest / largest / first'). Call with an empty query to get the schema and this graph's real edge/node vocabulary |
| `verify` | ✓‡ | ✓‡ | ✓‡ | Answer-grounding gate: pass the final `answer` and the `chunk_paths` it cites; returns serve/abstain. **‡ Advertised in every profile but withheld until the corpus both enables the gate (`[verify].enabled`) and is calibrated in *both* buckets — `[verify.ac.threshold]` (or the legacy `[verify.threshold]`) must carry a `single` **and** a `multi` value; one alone leaves the tool withheld. An uncalibrated gate can only ever abstain, so the route is not served** |
| `related` | | ✓ | ✓ | SIMILAR / COMMUNITY siblings after glossary |
| `neighbors` | | ✓ | ✓ | A node's direct structural edges (typed, 1-hop, with direction) |
| `resolve` | | ✓ | ✓ | Entity resolution by name |
| `get_ontology` | ✓ | ✓ | ✓ | Knowledge-base ontology as JSON: parameters, constraints, relations, graph-building patterns |
| `constraint_solve` | | ✓* | ✓* | CSP solver over the constraint graph (`validate` / `infer` / `check`); editor/full only, and only available with `--features constraint` |
| `ls` | ✓ | ✓ | ✓ | List notebook notes (agent workspace); read note content with `read` |
| `note` | | ✓ | ✓ | Create/replace — or with `append: true` extend — a notebook note (`doc`, `file`, `content`; `.csp` = validated limit table) |
| `del` | | ✓ | ✓ | Delete a notebook note |
| `index` | | ✓ | ✓ | Incremental index; `force: true` for a full rebuild, `path: <doc>` to reindex just one document |
| `graph_upsert` | | ✓ | ✓ | Create/update reasoning nodes and edges |
| `graph_build` | | ✓* | ✓* | Compile `.csp` limit tables into constraint graph; only available with `--features constraint` |
| `graph_delete` | | ✓ | ✓ | Remove nodes/edges by label |
| `graph_update` | | ✓ | ✓ | Rename or retype a node in place |
| `graph_generalize` | | ✓ | ✓ | Recompute derived layer (non-destructive; no longer reports ungrounded) |
| `graph_doctor` | | ✓ | ✓ | Diagnose the four graph doubts — ungrounded, stale, incomplete, and dangling nodes — and, opt-in per bucket (`prune_incomplete` / `prune_ungrounded` / `prune_dangling` / `prune_stale`), delete that bucket; the response then echoes the pruned counts. Report-only unless a flag is set. The non-destructive repair is `relink`: it re-points the groundings of relocated or relabeled documents onto their live targets (the CLI's `--relink`, same shared pre-relink backup), deletes nothing, and never touches the `ambiguous` or `orphans` groups. `relink` with any `prune_*` in one call is refused — the prune would run against the report taken before relink; relink alone first, then a separate call to prune. A destructive bucket that would wipe the reasoning layer wholesale — the ontology-mismatch symptom — is **REFUSED** (the server never force-overrides); the response says so and points at the human override, `kb graph doctor <path> --prune-<bucket> --force`. `prune_ungrounded` carries one more, repairable-first refusal: it is also refused outright while any of its nodes are relinkable, since a plain `relink` call would re-ground them for free — nodes that are merely `ambiguous` or genuine `orphans` never trigger it, since `relink` can't fix those |
| `graph_stats` | | ✓ | ✓ | Node/edge counts and community overview |
| `purge` | | | ✓ | Delete entire `.glossa/` |

Source of truth: [`src/mcp.rs`](../src/mcp.rs).

\* Only available when built with `cargo build --features constraint`. Without it, these tools are absent from the tool list.

## Failures vs non-applicable

A tool that **failed** answers with `isError: true`. A tool that simply **does not apply**, that
found **nothing**, or that can be **asked again better** answers normally. The distinction matters
because a failure returned as ordinary content is indistinguishable from an answer, and a weak
reader will relay the note to the user as one.

`isError: true` — the store could not do what was asked, and rephrasing the call will not help:

- `read(page_image: true)` on a PDF page that will not render (the file moved, was truncated, or
  the renderer cannot draw it). **The response also carries the chunk's text**, because the text
  comes from the index rather than from the file: the agent still gets what it asked to read and
  knows the image did not come. For a document the renderer can never draw, every page errors on
  every call — the text is what keeps it readable.
- `get_source_file` on a file the index has and the disk will not read. No fallback here: the file
  *is* the output, so the response is the error note alone.
- A `read` chunk-range lookup or chunk fetch that comes back an error, and the index probe behind a
  graph node's evidence set: a thin-but-confident evidence list is worse than a loud failure.
- `search` when the index cannot be opened or searched, or the query cannot be turned into
  something runnable.
- `grep` and `glob` when the index walk itself fails. The query's own compiles are checked first
  (see the next list), so what reaches here is the store, not the pattern.

Ordinary (successful) content — the tool does not apply, found nothing, or the call can be
corrected:

- `page_image` on a non-PDF or on a graph node: there is no page image to give.
- **An empty result.** Zero `search` hits (`(no results)`), zero `grep` matches, no document
  matching a `glob` mask: the agent asked whether the corpus holds something and learned that it
  does not. That is true, and it is an answer — erroring would teach a reader to distrust it.
- An out-of-range chunk number: `read` clamps to the valid range and says so in the response.
- A path that does not resolve: the reply says so and *may* carry did-you-mean hints. The hints are
  conditional, not guaranteed — nothing is offered for a basename under four characters, and
  otherwise only a substring match (`*<base>*`) or an overlap of at least two word tokens
  qualifies. There is no edit-distance search, so an ordinary one-character typo (`nots.md` for
  `notes.md`) gets the bare "check the path" reply. **No hint is not evidence that no such document
  exists** — fall back to `glob` or `search`.
- A path that exists on disk but was never indexed: the reply says exactly that
  (`… exists on disk but is not indexed — \`read\` serves indexed documents only`) and points at
  the ignore rules, deliberately *not* at a typo. This is the third not-found arm and the only one
  where re-reading the path is wasted effort.
- Notebook-lock contention: `REJECTED — another process is editing the notebook — retry` is
  content, because retrying is the correct response.
- A `grep` pattern the regex engine refuses: the reply says to set `fixed:true` or to simplify to
  one or two distinctive words. A `glob` the matcher refuses: the reply gives the glob syntax.
  Both name something the agent can change about its next call, and erroring on a fixable pattern
  is what makes a small model loop instead of rewriting it.
- A reasoning node passed to `get_source_file`: it backs evidence in several documents, not one
  file, and the reply says to pass a document path instead.
- An over-cap file: the reply asks for a specific PDF page.
- A corpus started without `--source-file`: the tool is absent from the advertised list, so no
  call reaches it at all.

The line, then, is not "did text come back" but **who can act on it**: a failure the agent cannot
route around is an error, and anything it can either use or retry differently is content. The
message text of each case is stable; `isError` is the part that tells failure from value.

Source of truth: `tools::ToolOut` and its one classifier `tool_failure` in
[`src/tools/mod.rs`](../src/tools/mod.rs), mapped to `isError` by `text_result` / `read_common` /
`source_file_result` in [`src/mcp.rs`](../src/mcp.rs).

## Typical agent workflow

1. **`search`** or **`grep`** — find relevant chunks. Each hit's leading `path#n` token is the one to carry forward; copy it, do not reassemble it.
2. **`read(path, n)`** — open full chunk text (with `--vision`, embedded office images are returned as JPEG vision content; `page_image: true` renders a PDF page as a JPEG for hard-to-parse tables/layout).
3. **`glossary("concept", query)`** — jump to the reasoning graph; get the cause → resolution chain with `read` anchors, plus a composed neighborhood ranked by `query` (pass the full question).
4. **`reach(from, relation)`** — close a multi-hop: walk the relation from a node, crossing document boundaries, or add a candidate to verify the connection.
5. **`sql`** — when the answer is a ranking or extreme, let read-only SQL over the graph decide instead of guessing from prose.
6. **`graph_upsert`** (editor) — add validated reasoning nodes; response shows what was written. Editor/full also expose `related` / `neighbors` / `resolve` for direct graph inspection.

### `graph_upsert` response

Responses are human-readable for the model:

- **`Written:`** — node ids and resolved edges persisted
- **`Merged:`** — duplicate labels merged into existing nodes
- **`N item(s) dropped (the rest WERE written)`** — the batch was **partially** applied. Fix and resend only the items it names; resending the whole batch duplicates work that already landed
- **`REJECTED — nothing written`** — the whole call failed (ontology, missing chunk, bad endpoints, duplicate handle); this is the only outcome where resending everything is correct
- **`Resolved by MORPHOLOGY, not an exact label`** — an endpoint label matched no node exactly and was bound to the closest morphological neighbour. Check each line is the node you meant

Reference endpoints by **label** — the tool takes labels and derives the ids — or by a **node id** you were handed (`sym:…`; ids are hashes, so never assemble one yourself). Do not paste ids into `label` fields.

## Retrieval anti-loop (`--dedup`)

Off by default. Start the server with `--dedup` (env `GLOSSA_MCP_DEDUP`, truthy = on) to enable per-session anti-loop signals on the retrieval tools. When a reader re-fetches results it has already seen, the response carries a neutral repeat/streak/plateau marker — or, for a governed repeat, a next-best-action recovery response — in place of the duplicate body, curbing context bloat for a weak reasoning loop. With dedup off (`config::defaults::DEDUP`), retrieval responses pass straight through unchanged and the session tracker is never consulted.

**For host integrators only:** a per-call `raw: true` bypasses the anti-loop even when dedup is enabled — display and programmatic fetches receive the full body and never touch the tracker. The field is deliberately **not advertised** in any retrieval tool's schema, because a governed reader that could set it would escape the anti-loop; it is set by the host application, not by the model. A model should not send it, and a schema-respecting client cannot. (`get_source_file` has its own, separate, *advertised* `raw` — that one controls original-file delivery and is unrelated.)

Each streamable-http session gets its own tracker, so one client's history never governs another's.

## Transports

### stdio (local)

```bash
kb mcp ./my-corpus --profile editor --transport stdio
```

Use with subprocess-based MCP clients (Claude Desktop, some IDE integrations).

### streamable-http (network)

```bash
kb mcp ./my-corpus --profile reader \
  --transport streamable-http \
  --bind 127.0.0.1:8080 \
  --allowed-host localhost
```

Endpoint: `http://127.0.0.1:8080/mcp`

Environment fallbacks: `GLOSSA_MCP_TRANSPORT`, `GLOSSA_MCP_BIND`.

Quickstart helpers: [`scripts/start-mcp-http.sh`](../scripts/start-mcp-http.sh) / [`scripts/start-mcp-http.ps1`](../scripts/start-mcp-http.ps1) start a streamable-http server against a corpus and print ready-to-paste Cursor `mcpServers` JSON.

Ops endpoints: `/health`, `/ready`, `/metrics` (Prometheus). Bearer auth (`--auth-token`), idle-session timeout (`--session-idle-secs`), JSON logs (`GLOSSA_LOG_FORMAT=json`), and audit events: [security-and-operations.md](security-and-operations.md). Deployment topology: [deploy/mcp-server.md](deploy/mcp-server.md).

## IDE configuration

Install `kb` first: [install.md](install.md). Use your corpus folder path in the examples below.

### Cursor (stdio)

```json
{
  "mcpServers": {
    "glossa": {
      "command": "/usr/local/bin/kb",
      "args": ["mcp", "/path/to/my-documents", "--profile", "reader", "--transport", "stdio"]
    }
  }
}
```

On Windows use the full path to `kb.exe`. Place in `.cursor/mcp.json` (project) or user MCP settings.

### Cursor (HTTP)

If glossa runs as a local HTTP service ([deploy/service.md](deploy/service.md)):

```json
{
  "mcpServers": {
    "glossa-reader": {
      "url": "http://127.0.0.1:8080/mcp"
    }
  }
}
```

Match `--bind` and `--allowed-host` on the server.

See [connect-to-agents.md](connect-to-agents.md) for Claude Desktop and other clients.

## ZeroClaw

→ [integrations/zeroclaw.md](integrations/zeroclaw.md)

## Freshness and maintenance

Every read tool calls `ensure_fresh` (throttled) so new files on disk appear without a manual `index`. Editor instances run a debounced **`graph_generalize`** maintenance loop after index changes, guarded by `.glossa/generalize.lock` across processes. Notebook writes (`note`, `del`) use `.glossa/notebook.lock` the same way.

**What an agent maintains.** An agent authors and tends its *own* nodes: create (`graph_upsert`), edit (`graph_update`), delete one (`graph_delete`), and diagnose — and, per doubt bucket, prune — with (`graph_doctor`). That doubt-scoped pruning still refuses any bucket that would wipe the reasoning layer wholesale (the ontology-mismatch symptom); the server never force-overrides. Two blunt, corpus-wide operations are deliberately **terminal-only** and have no MCP tool — `generalize --merge` (destructive near-duplicate collapse) and `prune -t <Type>` (wipe a whole node type) — so an agent cannot mass-mutate the graph; a human runs those from `kb`. The full create/maintain lifecycle is in [graph-lifecycle.md](graph-lifecycle.md).

Index (re)builds are serialized across processes by `.glossa/index.lock`: `index` (incremental, `force`, or single-`path`) and the `ensure_fresh` background scan hold it for the whole rebuild. If another process already holds it, the call skips with a no-op stat instead of racing it — clearing and reopening `.glossa/index` concurrently would otherwise fail on Windows ("Access is denied"). The index is cooperative, so whoever wins the lock leaves it correct.

## Regenerate external tool schemas

After changing MCP tools:

```bash
just tools
```

Writes schemas to `eval/tensorzero/config/tools/` from the live router definitions. Equivalent CLI: `kb mcp dump-tz-tools -d eval/tensorzero/config` — regenerates TensorZero tool config from the live MCP tool definitions (one source of truth).

## Production

For multi-process topology, TLS termination, systemd, and Windows SCM: [deploy/mcp-server.md](deploy/mcp-server.md).
