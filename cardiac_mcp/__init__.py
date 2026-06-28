"""cardiac_mcp — Model Context Protocol server exposing the cardiac_core simulation API.

A thin, host-agnostic adapter so any MCP host (Claude Desktop, Claude Code, an IDE) can run cardiac
electrophysiology simulations conversationally. ``core`` holds the logic; ``server`` wires it to the
transport. Run with ``python -m cardiac_mcp`` (stdio).
"""

__version__ = "0.1.0"
