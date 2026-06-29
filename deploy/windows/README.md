# glossa Windows service install

Install `kb` from a [GitHub Release](https://github.com/metalmon/glossa/releases) and register a **Windows Service** for MCP (streamable-http).

## Requirements

- Windows 10/11 or Windows Server
- PowerShell **run as Administrator**
- Network access to `github.com` for download

## Install

```powershell
cd deploy\windows
.\install-service.ps1 `
  -Version "1.0.0" `
  -CorpusPath "C:\glossa\corpus" `
  -Profile reader `
  -Bind "127.0.0.1:8080"
```

Copy your documents into `CorpusPath` before or after install. Initial index runs automatically if `.glossa` is missing.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-Version` | (required) | Release tag without `v` |
| `-CorpusPath` | (required) | Document folder |
| `-Profile` | `reader` | `reader`, `editor`, or `full` |
| `-Bind` | `127.0.0.1:8080` | HTTP bind |
| `-InstallDir` | `C:\Program Files\glossa` | Extract location |
| `-ServiceName` | `glossa-mcp` | SCM service name |
| `-AllowedHost` | `localhost` | MCP `--allowed-host` |

## Service account

By default the service runs as **Local System**. For a corpus under your user profile, set the service log-on account in `services.msc` and grant read (corpus) + read/write (`.glossa`) permissions.

## Connect agents

MCP endpoint: `http://127.0.0.1:8080/mcp`

→ [docs/connect-to-agents.md](../../docs/connect-to-agents.md)

## Manage

```powershell
sc.exe query glossa-mcp
sc.exe stop glossa-mcp
sc.exe start glossa-mcp
sc.exe delete glossa-mcp   # after stop
```

Manual index after adding files:

```powershell
& "C:\Program Files\glossa\glossa-1.0.0-x86_64-pc-windows-msvc\kb.exe" index C:\glossa\corpus
```

## Related

- [docs/deploy/service.md](../../docs/deploy/service.md) — overview
- [docs/deploy/mcp-server.md](../../docs/deploy/mcp-server.md) — advanced multi-instance topology
