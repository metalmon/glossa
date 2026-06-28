# Deploying the glossa MCP server

The `kb mcp` server speaks MCP (JSON-RPC 2.0, protocol `2025-06-18`) over two transports:

- **stdio** — local subprocess (IDE / desktop clients co-located with the binary).
- **streamable-http** — network endpoint at `<bind>/mcp` (the prod transport).

TLS, OAuth2/OIDC/mTLS and rate-limiting are terminated by a **reverse proxy / API gateway** in
front; the binary itself is HTTP-plaintext and expects to sit behind it.

## Topology: one process per (base × profile)

The corpus root (a positional arg) and the `--profile` decide what a process is. Run several
processes for one base, each with its own port:

```
:: indexer/editor — the ONLY writer; runs the generalize loop; exposes write + admin tools
kb.exe mcp C:\kb\base1 --profile editor --transport streamable-http --bind 127.0.0.1:8801 --allowed-host gw.internal

:: reader pool — read tools only; stays fresh via ensure_fresh; no generalize loop
kb.exe mcp C:\kb\base1 --profile reader --transport streamable-http --bind 127.0.0.1:8802 --allowed-host gw.internal
kb.exe mcp C:\kb\base1 --profile reader --transport streamable-http --bind 127.0.0.1:8803 --allowed-host gw.internal
```

- **Freshness is on every instance** — readers serve up-to-date results (the cooperative tantivy
  writer lock makes concurrent `ensure_fresh` safe). The profile gates **tools**, not freshness.
- **Constraint:** instances sharing one `.glossa` must be on the **same host** (the writer lock is a
  local file lock; unreliable over SMB/NFS). For another host, give it its own index copy.
- **Multiple editors** are fine — the heavy generalize pass is serialized across them by
  `.glossa/generalize.lock`.

### Two (or more) bases

Each base is an independent set of processes with its own ports; bases may live on different hosts:

```
kb.exe mcp C:\kb\base2 --profile editor --transport streamable-http --bind 127.0.0.1:8811 ...
kb.exe mcp C:\kb\base2 --profile reader --transport streamable-http --bind 127.0.0.1:8812 ...
```

Gateway routes by prefix: `/base1/*` → base1 reader pool (writes/admin → :8801),
`/base2/*` → base2 pool, etc.

## Config

All knobs are CLI flags, with env fallback (flag overrides env):

| Flag | Env | Default |
|---|---|---|
| `<path>` (positional) | — | nearest indexed root / cwd |
| `--transport stdio\|streamable-http` | `GLOSSA_MCP_TRANSPORT` | `stdio` |
| `--bind <addr>` | `GLOSSA_MCP_BIND` | `127.0.0.1:8080` |
| `--profile reader\|editor\|full` | — | `editor` |
| `--allowed-host <h>` (repeatable) | — | loopback only |
| `RUST_LOG` (log level) | `RUST_LOG` | `info,tantivy=warn` |

## Ops endpoints (streamable-http)

- `GET /health` — liveness (200 `ok`).
- `GET /ready` — readiness: index + graph openable (200 `ready`, else 503).
- `GET /metrics` — Prometheus: `glossa_up`, `glossa_index_chunks`, `glossa_graph_nodes`,
  `glossa_graph_edges`, `glossa_graph_dirty`.

Logs go to **stderr** (stdout is the stdio JSON-RPC channel). Each HTTP request is traced
(method/path/status/latency).

## Graceful shutdown

One signal stops the loop, drains the listener, and tears down sessions together:
- **Linux / containers:** SIGTERM (`systemctl stop`, `docker stop`) or Ctrl-C.
- **Windows:** Ctrl-C (console) or the SCM Stop/Shutdown control (service).

## Running as a service

### Linux (systemd) — native, foreground binary

`kb mcp ... --transport streamable-http` runs in the foreground; systemd supervises it. One unit
per (base, profile, port):

```ini
# /etc/systemd/system/glossa-base1-editor.service
[Unit]
Description=glossa MCP (base1, editor)
After=network.target

[Service]
ExecStart=/opt/glossa/kb mcp /srv/kb/base1 --profile editor --transport streamable-http --bind 127.0.0.1:8801 --allowed-host gw.internal
Environment=RUST_LOG=info,tantivy=warn
Restart=on-failure
# SIGTERM (the default KillSignal) triggers graceful shutdown
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
```

```
systemctl daemon-reload && systemctl enable --now glossa-base1-editor
```

### Windows — native service (SCM)

The binary integrates with the Service Control Manager (`--windows-service`, set in the binPath; not
for manual use). Create one service per (base, profile, port) with `sc.exe` (elevated):

```
sc.exe create glossa-base1-editor binPath= "C:\kb\kb.exe mcp C:\kb\base1 --profile editor --transport streamable-http --bind 127.0.0.1:8801 --allowed-host gw.internal --windows-service" start= auto
sc.exe description glossa-base1-editor "glossa MCP (base1, editor)"
sc.exe start glossa-base1-editor
:: ...
sc.exe stop glossa-base1-editor    :: SCM Stop → graceful shutdown
sc.exe delete glossa-base1-editor
```

Notes:
- The space after `binPath=` / `start=` is required by `sc.exe`.
- Set the service log-on account and grant it read access to the corpus + read/write to `.glossa`.
- Run readers as separate services on their own ports (`glossa-base1-reader-8802`, …).

## Build

```
cargo build --release            # target\release\kb.exe  (or target/release/kb on Linux)
```
