#!/usr/bin/env python3
"""Entry point for the Deepmail MCP server.

Run via:
  deepmail             (unified CLI, default = MCP server)
  deepmail serve       (explicit serve command)
  uv run python -m pi_email.mcp_entry
"""

from pi_email.mcp_server import main

if __name__ == "__main__":
    main()
