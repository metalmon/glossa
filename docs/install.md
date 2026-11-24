# Install glossa

Install the `kb` binary from a [GitHub Release](https://github.com/metalmon/glossa/releases). No Rust toolchain required.

Contributors who hack on the source: see [Build from source](#build-from-source) or [CONTRIBUTING.md](../CONTRIBUTING.md).

## Pick your platform

Every archive is named `glossa-{version}-{target}-{engine}` — the `{engine}` part is not optional,
there is no archive without it:

| Platform | Target | Archive | Binary inside |
|----------|--------|---------|---------------|
| Linux x86_64 | `x86_64-unknown-linux-gnu` | `glossa-{version}-x86_64-unknown-linux-gnu-{engine}.tar.gz` | `kb` |
| Windows x86_64 | `x86_64-pc-windows-msvc` | `glossa-{version}-x86_64-pc-windows-msvc-{engine}.zip` | `kb.exe` |
| macOS Apple Silicon | `aarch64-apple-darwin` | `glossa-{version}-aarch64-apple-darwin-{engine}.tar.gz` | `kb` |
| macOS Intel | `x86_64-apple-darwin` | `glossa-{version}-x86_64-apple-darwin-none.tar.gz` | `kb` |

Replace `{version}` with the tag you download, without the leading `v` (`0.5.9` for release `v0.5.9`).

## Pick your engine

`{engine}` is the execution provider the GPU-accelerated parts (reranking, the NLI grounding gate)
are built against. **It decides what must already be installed on the machine** — the wrong choice
installs fine and then fails at query time, so take `none` unless you know the host has the runtime.

| Engine | Needs on the machine | Published for |
|--------|----------------------|---------------|
| `none` | nothing — CPU only. **Start here.** | every target |
| `cuda13` | NVIDIA driver + CUDA 13 and cuDNN 9 libraries on the loader path | Linux x86_64, Windows |
| `directml` | a DirectX 12 GPU with a current vendor driver | Windows only |
| `rocm` | an AMD GPU with the ROCm runtime installed | Linux x86_64 only |
| `vulkan` | a Vulkan 1.2+ driver (any vendor); bundles nothing else | Linux x86_64, Windows |
| `coreml` | nothing extra — Apple's own stack | Apple Silicon only |

macOS Intel (`x86_64-apple-darwin`) has **only** the `none` archive.

`glossa-{version}-{target}-cuda13-nvidia-runtime` is **not** an install target: it carries NVIDIA's
CUDA 13 / cuDNN 9 libraries and no `kb` at all. Download it next to a `cuda13` install on a host
that lacks those libraries, and unpack it beside `kb`.

## Manual install

### Linux / macOS

```bash
VERSION=…                         # the release tag you are installing, without the v
TARGET=x86_64-unknown-linux-gnu   # or aarch64-apple-darwin / x86_64-apple-darwin
ENGINE=none                       # see "Pick your engine" above
STEM="glossa-${VERSION}-${TARGET}-${ENGINE}"

curl -LO "https://github.com/metalmon/glossa/releases/download/v${VERSION}/${STEM}.tar.gz"
tar -xzf "${STEM}.tar.gz"
sudo install -m 755 "${STEM}/kb" /usr/local/bin/kb
kb --version
```

### Windows (PowerShell)

```powershell
$Version = "…"                    # the release tag you are installing, without the v
$Engine  = "none"                 # see "Pick your engine" above
$Stem = "glossa-$Version-x86_64-pc-windows-msvc-$Engine"
$Url = "https://github.com/metalmon/glossa/releases/download/v$Version/$Stem.zip"
Invoke-WebRequest -Uri $Url -OutFile "$Stem.zip"
Expand-Archive -Path "$Stem.zip" -DestinationPath "$env:ProgramFiles\glossa" -Force
# Add to PATH (current user):
[Environment]::SetEnvironmentVariable("Path", $env:Path + ";$env:ProgramFiles\glossa\$Stem", "User")
```

Adjust paths if you install elsewhere.

## Automated install (service + index)

Scripts download the same release artifacts and register a background MCP server:

| Platform | Path | Guide |
|----------|------|-------|
| Linux | [`deploy/ansible/`](../deploy/ansible/README.md) | Ansible playbook + systemd |
| Windows | [`deploy/windows/`](../deploy/windows/README.md) | PowerShell + Windows Service |
| macOS | [`deploy/macos/`](../deploy/macos/README.md) | Shell script + launchd |

Each of them picks the engine for you with the same safe default — `none` — and takes an engine
parameter (`-Engine`, `--engine`, `glossa_engine`) to change it. A combination that has no
published archive is refused before the download, with the engines valid for that target named.

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
