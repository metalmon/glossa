#!/usr/bin/env bash
# Install glossa kb from GitHub Release and register a launchd agent (MCP streamable-http).
set -euo pipefail

VERSION=""
CORPUS=""
PROFILE="reader"
BIND="127.0.0.1:8080"
INSTALL_DIR="/usr/local/glossa"
ALLOWED_HOST="localhost"
LABEL="com.glossa.mcp"
SYSTEM_WIDE=false

usage() {
  cat <<EOF
Usage: $0 --version VERSION --corpus PATH [options]

  --version VERSION     Release tag without v (e.g. 0.1.0)
  --corpus PATH         Document folder (required)
  --profile PROFILE     reader | editor | full (default: reader)
  --bind ADDR           HTTP bind (default: 127.0.0.1:8080)
  --install-dir DIR     Install root (default: /usr/local/glossa)
  --allowed-host HOST   MCP allowed host (default: localhost)
  --system              Install LaunchDaemon in /Library/LaunchDaemons (requires sudo)
  -h                    Help

Example:
  $0 --version 0.1.0 --corpus "\$HOME/Documents/my-kb"
EOF
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version) VERSION="$2"; shift 2 ;;
    --corpus) CORPUS="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --bind) BIND="$2"; shift 2 ;;
    --install-dir) INSTALL_DIR="$2"; shift 2 ;;
    --allowed-host) ALLOWED_HOST="$2"; shift 2 ;;
    --system) SYSTEM_WIDE=true; shift ;;
    -h|--help) usage 0 ;;
    *) echo "Unknown option: $1" >&2; usage 1 ;;
  esac
done

[[ -n "$VERSION" ]] || { echo "Missing --version" >&2; usage 1; }
[[ -n "$CORPUS" ]] || { echo "Missing --corpus" >&2; usage 1; }

CORPUS="${CORPUS/#\~/$HOME}"
CORPUS="$(cd "$(dirname "$CORPUS")" && pwd)/$(basename "$CORPUS")"

ARCH="$(uname -m)"
case "$ARCH" in
  arm64) TARGET="aarch64-apple-darwin" ;;
  x86_64) TARGET="x86_64-apple-darwin" ;;
  *) echo "Unsupported macOS arch: $ARCH" >&2; exit 1 ;;
esac

REPO="metalmon/glossa"
STEM="glossa-${VERSION}-${TARGET}"
TARBALL="${STEM}.tar.gz"
URL="https://github.com/${REPO}/releases/download/v${VERSION}/${TARBALL}"
BIN_DIR="${INSTALL_DIR}/bin"
KB="${BIN_DIR}/kb"

echo "Downloading ${URL} ..."
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
curl -fsSL "$URL" -o "${TMP}/${TARBALL}"
tar -xzf "${TMP}/${TARBALL}" -C "$TMP"

sudo mkdir -p "$BIN_DIR"
sudo install -m 755 "${TMP}/${STEM}/kb" "$KB"

mkdir -p "$CORPUS"
if [[ ! -f "${CORPUS}/.glossa/manifest.json" ]]; then
  echo "Running initial index ..."
  "$KB" index "$CORPUS"
fi

PLIST_BODY="$(cat <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>${KB}</string>
    <string>mcp</string>
    <string>${CORPUS}</string>
    <string>--profile</string>
    <string>${PROFILE}</string>
    <string>--transport</string>
    <string>streamable-http</string>
    <string>--bind</string>
    <string>${BIND}</string>
    <string>--allowed-host</string>
    <string>${ALLOWED_HOST}</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>RUST_LOG</key>
    <string>info,tantivy=warn</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>${INSTALL_DIR}/glossa-mcp.log</string>
  <key>StandardErrorPath</key>
  <string>${INSTALL_DIR}/glossa-mcp.err.log</string>
</dict>
</plist>
PLIST
)"

if $SYSTEM_WIDE; then
  PLIST_PATH="/Library/LaunchDaemons/${LABEL}.plist"
  echo "$PLIST_BODY" | sudo tee "$PLIST_PATH" >/dev/null
  sudo launchctl bootout "system/${LABEL}" 2>/dev/null || true
  sudo launchctl bootstrap system "$PLIST_PATH"
  sudo launchctl enable "system/${LABEL}"
  sudo launchctl kickstart -k "system/${LABEL}"
else
  PLIST_PATH="${HOME}/Library/LaunchAgents/${LABEL}.plist"
  mkdir -p "$(dirname "$PLIST_PATH")"
  echo "$PLIST_BODY" > "$PLIST_PATH"
  launchctl bootout "gui/$(id -u)/${LABEL}" 2>/dev/null || true
  launchctl bootstrap "gui/$(id -u)" "$PLIST_PATH"
  launchctl enable "gui/$(id -u)/${LABEL}"
  launchctl kickstart -k "gui/$(id -u)/${LABEL}"
fi

echo ""
echo "Installed."
echo "  Binary:  ${KB}"
echo "  Corpus:  ${CORPUS}"
echo "  MCP URL: http://${BIND}/mcp"
echo "  Plist:   ${PLIST_PATH}"
echo ""
echo "Logs: ${INSTALL_DIR}/glossa-mcp.log"
echo "Unload: launchctl bootout gui/$(id -u)/${LABEL}  # user agent"
