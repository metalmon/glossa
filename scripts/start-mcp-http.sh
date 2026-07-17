#!/usr/bin/env bash
# Start glossa MCP (streamable-http) and print Cursor mcpServers JSON for copy-paste.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KB_ROOT="${1:-$REPO_ROOT/kb-gost-docs}"
BIND="${BIND:-127.0.0.1:8080}"
PROFILE="${PROFILE:-editor}"
SERVER_NAME="${SERVER_NAME:-glossa-kb}"

KB_EXE="${KB_EXE:-}"
if [[ -z "$KB_EXE" ]]; then
  for c in "$REPO_ROOT/target/release/kb" "$REPO_ROOT/target/debug/kb"; do
    [[ -x "$c" ]] && KB_EXE="$c" && break
  done
fi
if [[ -z "$KB_EXE" ]]; then
  KB_EXE="$(command -v kb || true)"
fi
if [[ -z "$KB_EXE" || ! -x "$KB_EXE" ]]; then
  echo "kb not found; build release or set KB_EXE" >&2
  exit 1
fi

KB_ROOT="$(cd "$KB_ROOT" && pwd)"
MCP_URL="http://${BIND}/mcp"

JSON=$(cat <<EOF
{
  "mcpServers": {
    "${SERVER_NAME}": {
      "url": "${MCP_URL}"
    }
  }
}
EOF
)

echo ""
echo "KB:      $KB_ROOT"
echo "kb:      $KB_EXE"
echo "bind:    $BIND"
echo "profile: $PROFILE"
echo "endpoint:$MCP_URL"
echo ""
echo "======== copy-paste into Cursor MCP config (.cursor/mcp.json or Settings → MCP) ========"
echo "$JSON"
echo "======================================================================================"
echo ""
echo "Health:  http://${BIND}/health"
echo "Starting MCP (Ctrl+C to stop)..."
echo ""

exec "$KB_EXE" mcp "$KB_ROOT" \
  --profile "$PROFILE" \
  --transport streamable-http \
  --bind "$BIND" \
  --allowed-host localhost \
  --allowed-host 127.0.0.1
