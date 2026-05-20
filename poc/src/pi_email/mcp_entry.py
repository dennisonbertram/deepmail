#!/usr/bin/env python3
"""Entry point for the Deep Email MCP server.

Run via:
  deep-email             (unified CLI, default = MCP server)
  deep-email serve       (explicit serve command)
  uv run python -m pi_email.mcp_entry
"""

from pi_email.mcp_server import main

if __name__ == "__main__":
    main()
