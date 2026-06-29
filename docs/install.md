# Install glossa

Install the `kb` binary from a [GitHub Release](https://github.com/metalmon/glossa/releases). No Rust toolchain required.

Contributors who hack on the source: see [Build from source](#build-from-source) or [CONTRIBUTING.md](../CONTRIBUTING.md).

## Pick your platform

| Platform | Archive | Binary inside |
|----------|---------|---------------|
| Linux x86_64 | `glossa-{version}-x86_64-unknown-linux-gnu.tar.gz` | `kb` |
| Windows x86_64 | `glossa-{version}-x86_64-pc-windows-msvc.zip` | `kb.exe` |
| macOS Apple Silicon | `glossa-{version}-aarch64-apple-darwin.tar.gz` | `kb` |
| macOS Intel | `glossa-{version}-x86_64-apple-darwin.tar.gz` | `kb` |

Replace `{version}` with the tag you download (e.g. `1.0.0` for release `v1.0.0`).

## Manual install

### Linux / macOS

```bash
VERSION=1.0.0
TARGET=x86_64-unknown-linux-gnu   # or aarch64-apple-darwin / x86_64-apple-darwin

curl -LO "https://github.com/metalmon/glossa/releases/download/v${VERSION}/glossa-${VERSION}-${TARGET}.tar.gz"
tar -xzf "glossa-${VERSION}-${TARGET}.tar.gz"
sudo install -m 755 "glossa-${VERSION}-${TARGET}/kb" /usr/local/bin/kb
kb --version
```

### Windows (PowerShell)

```powershell
$Version = "1.0.0"
$Zip = "glossa-$Version-x86_64-pc-windows-msvc.zip"
$Url = "https://github.com/metalmon/glossa/releases/download/v$Version/$Zip"
Invoke-WebRequest -Uri $Url -OutFile $Zip
Expand-Archive -Path $Zip -DestinationPath "$env:ProgramFiles\glossa" -Force
# Add to PATH (current user):
[Environment]::SetEnvironmentVariable("Path", $env:Path + ";$env:ProgramFiles\glossa\glossa-$Version-x86_64-pc-windows-msvc", "User")
```

Adjust paths if you install elsewhere.

## Automated install (service + index)

Scripts download the same release artifacts and register a background MCP server:

| Platform | Path | Guide |
|----------|------|-------|
| Linux | [`deploy/ansible/`](../deploy/ansible/README.md) | Ansible playbook + systemd |
| Windows | [`deploy/windows/`](../deploy/windows/README.md) | PowerShell + Windows Service |
| macOS | [`deploy/macos/`](../deploy/macos/README.md) | Shell script + launchd |

See [deploy/service.md](deploy/service.md) for an overview.

## First index

Point glossa at **any folder** that contains your documents (PDF, Office, Markdown, text, etc.):

```bash
mkdir -p ~/Documents/my-kb
# copy or symlink your files into ~/Documents/my-kb
kb index ~/Documents/my-kb
```

This creates `~/Documents/my-kb/.glossa/` (index + graph). Your files stay the source of truth — deleting `.glossa/` only loses derived state; re-run `kb index` to rebuild.

Quick sanity check:

```bash
cd ~/Documents/my-kb
kb search "connection timeout"
kb grep maxTsdr
```

After the first index, MCP and CLI calls **auto-refresh** when files change — you usually do not need to re-run `kb index` manually.

## Connect to an agent

Once indexed, attach the corpus to Claude, Cursor, ZeroClaw, or any MCP client:

→ [connect-to-agents.md](connect-to-agents.md)

## Build from source

```bash
git clone https://github.com/metalmon/glossa.git
cd glossa
cargo build --release
# binary: target/release/kb  (or target\release\kb.exe on Windows)
```

See [getting-started.md](getting-started.md) for the full developer workflow.

## Next steps

- [connect-to-agents.md](connect-to-agents.md) — Claude Desktop, Cursor, ZeroClaw
- [deploy/service.md](deploy/service.md) — run as a system service
- [mcp.md](mcp.md) — tool reference
- [integrations/zeroclaw.md](integrations/zeroclaw.md) — ZeroClaw MCP config
