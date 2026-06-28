"""Entry point: ``python -m cardiac_mcp`` -> stdio MCP server.

Remote later: swap ``mcp.run()`` for ``mcp.run(transport="streamable-http")`` (FastMCP serves the
same tools over HTTP); no change to core/server.
"""
from cardiac_mcp.server import mcp

if __name__ == "__main__":
    mcp.run()
