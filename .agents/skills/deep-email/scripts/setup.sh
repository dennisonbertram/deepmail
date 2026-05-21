#!/usr/bin/env bash
# Deepmail setup script
# Checks if the MCP server is configured and dependencies are installed

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKILL_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$SKILL_DIR")"

echo "Deepmail setup"
echo "=============="
echo ""

# Check if deepmail is installed
if command -v deepmail &>/dev/null; then
    echo "[ok] deepmail CLI found"
else
    echo "[!!] deepmail CLI not found"
    echo "     Install: pip install deepmail"
    echo "     Or run:  uvx deepmail"
fi

# Check auth
if command -v deepmail &>/dev/null; then
    AUTH_STATUS=$(deepmail whoami 2>/dev/null || echo "not-authenticated")
    if echo "$AUTH_STATUS" | grep -qi "email"; then
        echo "[ok] Gmail authenticated"
    else
        echo "[!!] Not authenticated. Run:"
        echo "     deepmail auth"
    fi
fi

echo ""
echo "Quick setup: run 'deepmail setup' for interactive walkthrough."
echo ""
echo "MCP server config for .mcp.json (or run 'deepmail init'):"
echo ""
echo "{"
echo "  \"mcpServers\": {"
echo "    \"deepmail\": {"
echo "      \"type\": \"stdio\","
echo "      \"command\": \"uvx\","
echo "      \"args\": [\"deepmail\"]"
echo "    }"
echo "  }"
echo "}"
