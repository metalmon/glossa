# glossa macOS service install (launchd)

Install `kb` from a [GitHub Release](https://github.com/metalmon/glossa/releases) and register a **launchd** agent for MCP (streamable-http).

## Requirements

- macOS 12+ (Apple Silicon or Intel)
- `curl`, `tar`
- Network access to `github.com`

## Install (user agent)

Runs as your login user — good for corpora under `~/Documents`:

```bash
chmod +x deploy/macos/install-service.sh
./deploy/macos/install-service.sh \
  --version 1.0.0 \
  --corpus "$HOME/Documents/my-kb" \
  --profile reader \
  --bind 127.0.0.1:8080
```

Binary installs to `/usr/local/glossa/bin/kb`. Plist: `~/Library/LaunchAgents/com.glossa.mcp.plist`.

## System-wide daemon

```bash
sudo ./deploy/macos/install-service.sh \
  --version 1.0.0 \
  --corpus /srv/glossa/corpus \
  --system
```

Plist: `/Library/LaunchDaemons/com.glossa.mcp.plist`.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--version` | (required) | Release without `v` |
| `--corpus` | (required) | Document folder |
| `--profile` | `reader` | MCP profile |
| `--bind` | `127.0.0.1:8080` | HTTP bind |
| `--install-dir` | `/usr/local/glossa` | Install root + log files |
| `--allowed-host` | `localhost` | MCP allowed host |
| `--system` | off | LaunchDaemon instead of LaunchAgent |

## Connect agents

MCP endpoint: `http://127.0.0.1:8080/mcp`

→ [docs/connect-to-agents.md](../../docs/connect-to-agents.md)

## Manage

```bash
# User agent
launchctl kickstart -k "gui/$(id -u)/com.glossa.mcp"
launchctl bootout "gui/$(id -u)/com.glossa.mcp"

# Logs
tail -f /usr/local/glossa/glossa-mcp.log
```

Re-index after bulk file adds:

```bash
/usr/local/glossa/bin/kb index "$HOME/Documents/my-kb"
```

## Related

- [docs/deploy/service.md](../../docs/deploy/service.md)
- [docs/integrations/zeroclaw.md](../../docs/integrations/zeroclaw.md)
