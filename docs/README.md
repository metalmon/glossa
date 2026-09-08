# glossa documentation

User-facing documentation for the glossa knowledge-base engine, organized in the order you'll
likely need it: get it running, understand the model, follow a guide for your use case, then
reach for reference material as needed.

## 1. Get started

| Document | Audience | Summary |
|----------|----------|---------|
| [install.md](install.md) | Operators | Download release binary, first index |
| [getting-started.md](getting-started.md) | Developers | Build from source, CLI workflow |

The [project README](../README.md) has the fastest quickstart (three commands, no reading required).

## 2. Concepts

| Document | Audience | Summary |
|----------|----------|---------|
| [concepts.md](concepts.md) | Everyone | Corpus, `.glossa/`, roots, index vs graph, ontology, profiles, scope — the model in one page |
| [architecture.md](architecture.md) | Developers | File-first design, index, graph layers, extraction |

## 3. Guides

| Document | Audience | Summary |
|----------|----------|---------|
| [connect-to-agents.md](connect-to-agents.md) | Operators | Attach your documents to Claude, Cursor, MCP |
| [cli-for-agents.md](cli-for-agents.md) | Agent integrators | Use `kb` as a CLI tool — `cat`/`grep`/`read` for Office & PDF |
| [mcp.md](mcp.md) | Agent integrators | MCP tools, profiles, local IDE config |
| [graph-and-ontology.md](graph-and-ontology.md) | Domain operators | Ontology overlay, grounding, retrieval tuning, graph doctor |
| [graph-lifecycle.md](graph-lifecycle.md) | Operators / agent authors | Build & maintain the graph: create, health, add/edit/delete-a-document workflows |
| [ontology-presets.md](ontology-presets.md) | Domain operators | Baked task ontologies, `--ontology` flag, `kb ontology` commands, catalog |
| [agent-workspace-contract.md](agent-workspace-contract.md) | Agent integrators | Corpus vs notebook: `note`/`ls`/`del`, `.csp` limit-table notes, storage layout |
| [integrations/zeroclaw.md](integrations/zeroclaw.md) | ZeroClaw users | `config.toml` MCP wiring |

## 4. Reference

| Document | Audience | Summary |
|----------|----------|---------|
| [cli-reference.md](cli-reference.md) | Everyone | Complete flag-by-flag reference for every `kb` subcommand |
| [configuration.md](configuration.md) | Operators | Every environment variable and `--config` TOML key, with precedence |
| [troubleshooting.md](troubleshooting.md) | Everyone | FAQ / common problems, each grounded in the causing behavior |

## 5. Deploy & operate

| Document | Audience | Summary |
|----------|----------|---------|
| [deploy/service.md](deploy/service.md) | Ops | Service install (Linux / Windows / macOS) |
| [deploy/mcp-server.md](deploy/mcp-server.md) | DevOps | Advanced HTTP deployment, multi-instance topology, native TLS |
| [security-and-operations.md](security-and-operations.md) | Ops / Security | Auth, TLS, idle timeout, metrics, JSON logs, audit events, readiness scorecard |

### Deploy automation

| Path | Platform |
|------|----------|
| [../deploy/ansible/README.md](../deploy/ansible/README.md) | Linux (Ansible + systemd) |
| [../deploy/windows/README.md](../deploy/windows/README.md) | Windows (PowerShell + SCM) |
| [../deploy/macos/README.md](../deploy/macos/README.md) | macOS (launchd) |

## 6. Develop & extend

| Document | Audience | Summary |
|----------|----------|---------|
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | Contributors | Build, test, PR expectations |
| [testing/e2e.md](testing/e2e.md) | Contributors | End-to-end test harness: spawn the real `kb` binary over real sockets (`e2e`/`tls` features) |
| [constraint-gepa.md](constraint-gepa.md) | Contributors | GEPA prompt optimization for the `.csp` limit-table extraction agent |
| [constraint-tables-compiler.md](constraint-tables-compiler.md) | Contributors | `kb graph build`'s capability set — what `.csp` tables compile into a constraint graph |
| [eval-and-training.md](eval-and-training.md) | Benchmark developers | The `kbx` reasoning-layer pipeline (build/reason/train/distil/eval/export) + the legacy kb-eval/kb-train TensorZero workflow |
| [finetuning-datasets.md](finetuning-datasets.md) | ML engineers | Build SFT/DPO datasets from your graph for Unsloth — teacher distillation + on-policy capture |
| [graph-reasoning-directions.md](graph-reasoning-directions.md) | Contributors | Reasoning-graph direction: Peirce triad, planned inference modes |

## 7. Project

| Document | Audience | Summary |
|----------|----------|---------|
| [ROADMAP.md](ROADMAP.md) | Contributors | Backlog and product direction |
| [../CHANGELOG.md](../CHANGELOG.md) | Everyone | Release history |
| [benchmarks.md](benchmarks.md) | Researchers | Append-only eval run log |
| [../LICENSE](../LICENSE) | Everyone | MIT license |

## Related

- [../README.md](../README.md) — project overview and quickstart
- [../eval/tensorzero/README.md](../eval/tensorzero/README.md) — TensorZero gateway setup
