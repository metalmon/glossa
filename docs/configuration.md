# Configuration reference

Every runtime knob glossa reads, in one place: environment variables, the `--config` TOML file,
and how the two combine with CLI flags. For the full CLI flag surface these mirror, see
[cli-reference.md](cli-reference.md).

## Precedence

**CLI flag > environment variable > config file > built-in default**, resolved independently per
setting (not per section) — e.g. one deployment can set `bind` in the config file while overriding
`GLOSSA_MCP_TOKEN` only via a systemd `EnvironmentFile`. clap's `env = "..."` binding already
collapses "flag or env" into a single value before the config-file layer is consulted, so a flag
and its env-var equivalent are never in conflict — whichever is present wins over the file.

**Secret rule: the bearer token is env/flag only, never the config file.** A `token`/`auth_token`
key under `[server]` makes the whole config file **fail to load** (fail closed, not silently
ignored) — set it via `--auth-token` or `GLOSSA_MCP_TOKEN` instead. See
[security-and-operations.md § Authentication](security-and-operations.md#authentication).

## Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `GLOSSA_ROOTS` | — | Corpus root(s), newline-separated `[LABEL=]PATH` entries. Ignored if any `--root` flag is given. |
| `GLOSSA_STATE_DIR` | co-located with corpus | Local directory holding `.glossa/` state. |
| `GLOSSA_CONFIG` | — | Path to a TOML deployment config file (`--config`). |
| `GLOSSA_MCP_TRANSPORT` | `stdio` | `stdio` \| `streamable-http` (`kb mcp --transport`). |
| `GLOSSA_MCP_BIND` | `127.0.0.1:8080` | Bind address for `streamable-http`. |
| `GLOSSA_MCP_TOKEN` | unset (unauthenticated) | Bearer token guarding `/mcp`. |
| `GLOSSA_MCP_INSECURE` | `false` | Override the non-loopback+no-auth startup refusal. Takes `true`/`false`. |
| `GLOSSA_MCP_SESSION_IDLE_SECS` | `0` (disabled) | Idle-session timeout for `streamable-http`. |
| `GLOSSA_MCP_REQUEST_TIMEOUT_SECS` | `120` | Request timeout on `/mcp`. |
| `GLOSSA_MCP_MAX_BODY_BYTES` | `4000000` (4 MB) | Body size limit on `/mcp`. |
| `GLOSSA_MCP_MAX_CONCURRENCY` | unset (off) | Opt-in cap on in-flight `/mcp` requests; the `(n+1)`-th is shed with `503`. |
| `GLOSSA_MCP_RATE_LIMIT_PER_SEC` | unset (off) | Opt-in per-IP token-bucket rate limit. |
| `GLOSSA_MCP_MAX_CONNECTIONS` | unset (off) | Opt-in cap on concurrently open connections. |
| `GLOSSA_MCP_MAX_HANDSHAKES` | `256` | *(`tls` feature build)* Pre-auth TLS handshake concurrency bound; always on, not opt-in. |
| `GLOSSA_MCP_TLS_HANDSHAKE_TIMEOUT_SECS` | `10` | *(`tls` feature build)* Per-handshake timeout. |
| `GLOSSA_TLS_CERT` | — | *(`tls` feature build)* PEM certificate chain path. |
| `GLOSSA_TLS_KEY` | — | *(`tls` feature build)* PEM private key path. |
| `GLOSSA_TLS_CLIENT_CA` | — | *(`tls` feature build)* PEM client-CA path (enables mTLS). |
| `GLOSSA_READ_RETRIES` | `3` | Max retry attempts for a transient corpus-read failure. |
| `GLOSSA_READ_RETRY_BACKOFF_MS` | `200` | Base retry backoff (doubles each attempt). |
| `GLOSSA_FRESHEN_DEADLINE_MS` | `3000` | Wall-clock budget for the awaited freshen walk+reindex on the read path. |
| `GLOSSA_MIN_RESCAN_MS` | `2000` | Minimum spacing between freshen stat-walks (absorbs SMB/NFS attribute-cache lag). |
| `RUST_LOG` | `info,tantivy=warn,pdf_oxide=error` | `tracing`-style log-level filter. Reloadable at runtime — see [security-and-operations.md](security-and-operations.md#reliability--lifecycle). |
| `GLOSSA_LOG_FORMAT` | text | `json` emits one JSON object per line (SIEM/log pipeline); anything else is human-readable. |
| `GLOSSA_VISION` | `false` | Env form of `--vision` — enable image output in `read`. |
| `GLOSSA_SOURCE_FILE` | `false` | Env form of `--source-file` — enable the `get_source_file` tool. |

`GLOSSA_NO_IMAGE` also exists as the env form of the deprecated `-N/--noimage` no-op flag; images
are already off unless `--vision`/`GLOSSA_VISION` is set, so it has no effect and is not listed
above as a knob to reach for.

Two retrieval-tuning env vars are corpus-level, not server-level: `GLOSSA_PPR_SIM_WEIGHT` and
`GLOSSA_PPR_SPINE_WEIGHT` (plus `GLOSSA_PPR_BRIDGE`). They override the matching `[retrieval]`
keys in the corpus's `ontology.toml` — see [Corpus config (`ontology.toml`)](#corpus-config-ontologytoml)
below and [graph-and-ontology.md § Retrieval tuning](graph-and-ontology.md#retrieval-tuning).

## Corpus roots and document keys

A document's **key** — the identifier the index stores, that search results reference, and that a
graph node's grounding (`source_path`) points at — is its path **relative to the corpus root**,
not its absolute filesystem path. Keys are **bare by default** — no label prefix — and a label
appears only when you opt into one via `--root`:

- **Discovery** — no path given; `kb` walks up from the current directory to find `.glossa/` — keys
  come out **label-free**: `manual.pdf`, `guide/intro.pdf`.
- **A positional path** — `kb index /data/plc` — is exactly the same as discovery rooted at that
  path: keys are still **label-free**: `manual.pdf`, `guide/intro.pdf`.
- **`--root PATH`** — attaches a corpus as a *named* root, auto-labeled from the path's basename —
  keys come out prefixed: `plc/manual.pdf`, `plc/guide/intro.pdf`. `--root LABEL=PATH` sets the
  label explicitly instead of deriving it from the basename. This exists for multi-root setups and
  network/shared corpora where a stable, explicit name matters more than brevity.

**Pick one way to address a given corpus and stay consistent.** Switching a corpus between bare
addressing (discovery or a positional path) and labeled addressing (`--root`) changes every
document's key — `manual.pdf` becomes `plc/manual.pdf` or back again — which breaks any
reasoning-layer grounding built under the old key form, even though the file itself never moved. If
that happens, it's non-destructively recoverable: see
[graph-lifecycle.md § You relabeled the corpus, or moved a document between
folders](graph-lifecycle.md#you-relabeled-the-corpus-or-moved-a-document-between-folders).

**Multiple roots** each need a distinct label — a key is `label/relpath` — since two roots sharing
a label would collide on the same keys.

## Config-file keys (`--config` TOML)

The `--config <path>` file (also `GLOSSA_CONFIG`) holds the base settings for one deployment
**role** — one file can drive both `kb index` (reads `[corpus]` only) and `kb mcp` (reads every
section). Unknown sections/keys are rejected at load time (a typo fails closed, not silently
ignored). The canonical, fully-annotated example is
[deploy/glossa.toml](deploy/glossa.toml) — this table maps each key to its flag/env equivalent
rather than duplicating the file.

| Section.key | Flag / env equivalent |
|-------------|------------------------|
| `[corpus].roots` | `--root` / `GLOSSA_ROOTS` |
| `[corpus].state_dir` | `--state-dir` / `GLOSSA_STATE_DIR` |
| `[server].transport` | `--transport` / `GLOSSA_MCP_TRANSPORT` |
| `[server].bind` | `--bind` / `GLOSSA_MCP_BIND` |
| `[server].session_idle_secs` | `--session-idle-secs` / `GLOSSA_MCP_SESSION_IDLE_SECS` |
| `[server].allowed_hosts` | `--allowed-host` (repeatable) |
| `[server].insecure` | `--insecure` / `GLOSSA_MCP_INSECURE` |
| `[server].auth_token` / `[server].token` | **Rejected.** The token is env/flag only — see the secret rule above. |
| `[tls].cert` | `--tls-cert` / `GLOSSA_TLS_CERT` *(requires `tls` feature build; an unrecognized `[tls]` section in a non-`tls` build fails to load)* |
| `[tls].key` | `--tls-key` / `GLOSSA_TLS_KEY` |
| `[tls].client_ca` | `--tls-client-ca` / `GLOSSA_TLS_CLIENT_CA` |
| `[limits].request_timeout_secs` | `GLOSSA_MCP_REQUEST_TIMEOUT_SECS` |
| `[limits].max_body_bytes` | `GLOSSA_MCP_MAX_BODY_BYTES` |
| `[limits].max_concurrency` | `GLOSSA_MCP_MAX_CONCURRENCY` |
| `[limits].rate_limit_per_sec` | `GLOSSA_MCP_RATE_LIMIT_PER_SEC` |
| `[limits].connection_cap` | `GLOSSA_MCP_MAX_CONNECTIONS` |
| `[limits].max_handshakes` | `GLOSSA_MCP_MAX_HANDSHAKES` *(`tls` feature build)* |
| `[retrieval].read_retries` | `GLOSSA_READ_RETRIES` |
| `[retrieval].read_retry_backoff_ms` | `GLOSSA_READ_RETRY_BACKOFF_MS` |
| `[retrieval].freshen_deadline_ms` | `GLOSSA_FRESHEN_DEADLINE_MS` |
| `[retrieval].min_rescan_ms` | `GLOSSA_MIN_RESCAN_MS` |
| `[logging].format` | `GLOSSA_LOG_FORMAT` |
| `[logging].level` | `RUST_LOG` (env still overrides the file) |

**No hot reload.** The file is read once at startup (and by each one-shot `kb index` run). A
structural change (roots, `state_dir`, `bind`, `transport`, TLS paths) needs a process restart —
it is not part of the `SIGHUP` reload path (which covers only the log-level file, a forced
freshen, and re-reading TLS cert/key/client-CA bytes in a `--features tls` build).

## Precedence + secret rule

Per-setting precedence is **flag > env > file > default**, as stated above. The one setting
exempt from "file" as a valid source is the bearer token: it is **env/flag only**, and a config
file that tries to set it fails to load rather than being silently ignored. See
[security-and-operations.md](security-and-operations.md) for the full authentication, TLS, and
hardening picture this configuration surface supports.

## Corpus config (`ontology.toml`)

The deployment `--config` file above is server-role settings. Retrieval and answer-grounding
tuning instead live **per corpus** in that corpus's `ontology.toml` (under its `.glossa/`), read by
both `kb` and the `kbx` toolkit. Every key below is optional; an unset key falls back to the
built-in engine default. The matching env vars (where listed) override the file for one-off sweeps.

### `[retrieval]`

| Key | Values / default | Meaning |
|-----|------------------|---------|
| `sim_weight` | float ≥ 0 · default `0.1` | PPR transition weight of a mechanical `SIMILAR` edge relative to a reasoning edge (`1.0`). Lower = leaner similarity mass (suits a stronger reader); higher = heavier (suits a weaker reader). Env `GLOSSA_PPR_SIM_WEIGHT`. |
| `spine_weight` | float ≥ 0 · default `1.0` | PPR transition weight of a reasoning-spine (`Chaining`-role) edge relative to a plain reasoning edge (`1.0`, a no-op). `> 1.0` boosts the spine so a load-bearing bridge edge isn't diluted by out-degree against grounding/descriptive edges. Env `GLOSSA_PPR_SPINE_WEIGHT`. |
| `bridge` | `"off"` \| `"geomean"` · default `"off"` | Dual-seed combination mode for composed PPR (query-time seeding only). Env `GLOSSA_PPR_BRIDGE`. |

### `[verify]` — answer-grounding gate

The model-free grounding verifier behind the `verify` MCP tool. It stays withheld until it is both
enabled here and has a calibrated threshold.

| Key | Values / default | Meaning |
|-----|------------------|---------|
| `enabled` | bool · default `false` | Turn the gate on. |
| `mode` | `"ac"` \| `"nli"` \| `"combined"` · default `"ac"` | How the lexical/anomaly ("AC") and NLI verdicts combine. |
| `[verify.threshold]` | `single`, `multi` floats | Calibrated grounding thresholds by hop bucket; an unset table (or field) means uncalibrated — the gate abstains rather than guess. Legacy location, superseded by `[verify.ac.threshold]` when present. |
| `[verify.ac.threshold]` | `single`, `multi` floats | The AC verifier's own threshold table (preferred over `[verify.threshold]`). |
| `[verify.nli.threshold]` | `single`, `multi` floats | The NLI verifier's calibrated thresholds by bucket. |

### `[verify.nli]` — NLI support-verifier

| Key | Values / default | Meaning |
|-----|------------------|---------|
| `scorer` | `"in_process"` \| `"http"` | NLI scorer implementation (`http` is not built yet). Unset ⇒ no scorer ⇒ AC-only. |
| `model_dir` | path | Exported NLI model directory (in-process scorer only). |
| `entail_index` | integer | Softmax index of the entailment class in the model's output (model-export-specific). |
| `execution_providers` | ordered list · default `["cpu"]` | ONNX Runtime execution-provider preference for the `nli-ort` engine, tried in order; first available wins. Recognized GPU names: `"cuda"`, `"directml"`, `"coreml"`, `"rocm"`; `"cpu"` and unknown names fall through to ORT's implicit CPU EP. A GPU EP works only in a build that compiled it in. |

## Eval-harness config (`lab.toml`)

The `kbx` eval/train toolkit reads its own `lab.toml`. Two knobs worth calling out here:

- **Per-endpoint `headers`.** Each `[model]` / `[judge]` / `[bridge]` endpoint may carry a
  `headers` table of extra request headers sent with every call to that endpoint. A value may
  contain the `${{session}}` placeholder, which is substituted per call with the active trace/episode
  session id (for gateways that require a prompt-cache / session header, e.g. OpenCode's
  `x-opencode-session`). If no session is available to fill it, that header is dropped rather than
  sent with a literal placeholder; headers without the placeholder pass through unchanged.
- **`[tuning] reasoning_only`.** A reasoning-scope **allowlist** for graph construction: when
  non-empty, `kbx build` mines only documents whose corpus-relative path contains one of these
  substrings (its inverse, `reasoning_exclude`, is a denylist, and exclude wins over only). It
  merges with the `--only` CLI flag. Use it when the support docs are few enough that listing them
  is shorter than excluding every reference doc.

## See also

- [cli-reference.md](cli-reference.md) — every flag these variables mirror
- [deploy/glossa.toml](deploy/glossa.toml) — annotated example config covering every section
- [deploy/mcp-server.md](deploy/mcp-server.md) — deployment topology and the config file in context
- [security-and-operations.md](security-and-operations.md) — auth, TLS, hardening, observability
