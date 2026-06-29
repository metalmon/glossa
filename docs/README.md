# glossa documentation

User-facing documentation for the glossa knowledge-base engine.

## Guides

| Document | Audience | Summary |
|----------|----------|---------|
| [install.md](install.md) | **Operators** | Download release binary, first index |
| [connect-to-agents.md](connect-to-agents.md) | **Operators** | Attach your documents to Claude, Cursor, MCP |
| [integrations/zeroclaw.md](integrations/zeroclaw.md) | ZeroClaw users | `config.toml` MCP wiring |
| [getting-started.md](getting-started.md) | Developers | Build from source, CLI workflow |
| [architecture.md](architecture.md) | Developers | File-first design, index, graph, derived layer |
| [mcp.md](mcp.md) | Agent integrators | MCP tools, profiles, local IDE config |
| [graph-and-ontology.md](graph-and-ontology.md) | Domain operators | Ontology overlay, enrich workflow, CLI mirror |
| [eval-and-training.md](eval-and-training.md) | Benchmark developers | kb-eval, kb-train, dataset format, just recipes |
| [deploy/service.md](deploy/service.md) | Ops | Service install (Linux / Windows / macOS) |
| [deploy/mcp-server.md](deploy/mcp-server.md) | DevOps | Advanced HTTP deployment, multi-instance |
| [ROADMAP.md](ROADMAP.md) | Contributors | Backlog and product direction |
| [benchmarks.md](benchmarks.md) | Researchers | Append-only eval run log |

## Deploy automation

| Path | Platform |
|------|----------|
| [../deploy/ansible/README.md](../deploy/ansible/README.md) | Linux (Ansible + systemd) |
| [../deploy/windows/README.md](../deploy/windows/README.md) | Windows (PowerShell + SCM) |
| [../deploy/macos/README.md](../deploy/macos/README.md) | macOS (launchd) |

## Related

- [README.md](../README.md) — project overview and quickstart
- [CONTRIBUTING.md](../CONTRIBUTING.md) — build, test, PR expectations
- [eval/tensorzero/README.md](../eval/tensorzero/README.md) — TensorZero gateway setup
