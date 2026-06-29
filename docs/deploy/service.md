# Install glossa as a service

Run `kb mcp` in the background so agents connect over HTTP without spawning a new process each session.

**Prerequisite:** install `kb` from a [GitHub Release](../install.md) or use the platform scripts below (they download the release for you).

## What gets installed

| Component | Purpose |
|-----------|---------|
| `kb` binary | `/opt/glossa/bin/kb` (Linux), `Program Files\glossa` (Windows), `/usr/local/glossa/bin/kb` (macOS) |
| Corpus directory | Your documents (you choose the path) |
| `.glossa/` | Index + graph inside the corpus dir (created by initial `kb index`) |
| MCP server | `streamable-http` on loopback by default (`127.0.0.1:8080/mcp`) |

Default profile in the install scripts is **`reader`** (query-only tools). Use **`editor`** if the service should expose graph write tools.

## Platform scripts (release binary)

All scripts download a pinned version from GitHub Releases — **no `cargo build`**.

### Linux — Ansible

```bash
cd deploy/ansible
ansible-playbook -i inventory playbook.yml \
  -e glossa_version=1.0.0 \
  -e glossa_corpus_path=/srv/glossa/corpus
```

→ [deploy/ansible/README.md](../../deploy/ansible/README.md)

### Windows — PowerShell (elevated)

```powershell
.\deploy\windows\install-service.ps1 `
  -Version 1.0.0 `
  -CorpusPath "C:\glossa\corpus" `
  -Profile reader `
  -Bind "127.0.0.1:8080"
```

→ [deploy/windows/README.md](../../deploy/windows/README.md)

### macOS — launchd

```bash
./deploy/macos/install-service.sh \
  --version 1.0.0 \
  --corpus "$HOME/Documents/my-kb" \
  --profile reader \
  --bind 127.0.0.1:8080
```

→ [deploy/macos/README.md](../../deploy/macos/README.md)

## Connect agents after install

HTTP endpoint (default): `http://127.0.0.1:8080/mcp`

→ [connect-to-agents.md](../connect-to-agents.md) — Cursor, Claude Desktop (via HTTP-capable client), ZeroClaw

Health checks:

```bash
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/ready
```

## Permissions

The service account must:

- **Read** all files under the corpus path
- **Read/write** `{corpus}/.glossa/` (index updates, graph)

On Windows, configure the service log-on user in `services.msc` and grant folder ACLs accordingly.

## Manual service units

For custom topologies (multiple bases, reader pools, gateway routing), see the full reference:

→ [mcp-server.md](mcp-server.md)

## Uninstall

| Platform | Command |
|----------|---------|
| Linux | `systemctl disable --now glossa-mcp && rm /etc/systemd/system/glossa-mcp.service` |
| Windows | `sc.exe stop glossa-mcp && sc.exe delete glossa-mcp` |
| macOS | `launchctl unload ~/Library/LaunchAgents/com.glossa.mcp.plist` |

Remove the install directory and corpus `.glossa/` if you want a clean slate. Corpus files are never deleted by uninstall scripts.
