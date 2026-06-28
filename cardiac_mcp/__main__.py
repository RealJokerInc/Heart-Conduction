"""Entry point: ``python -m cardiac_mcp`` / the ``cardiac-mcp`` console script.

Default transport is stdio. Set ``CARDIAC_MCP_TRANSPORT=http`` (alias ``streamable-http``) to serve
over HTTP on localhost — UNAUTHENTICATED and localhost-only by design; see ``REMOTE_DEPLOY.md`` for the
auth/security stack required before any non-localhost deploy.
"""
from cardiac_mcp.server import mcp


def main():
    """Console-script entry point (`cardiac-mcp`). Transport via CARDIAC_MCP_TRANSPORT (stdio|http)."""
    import os
    import sys

    transport = os.environ.get("CARDIAC_MCP_TRANSPORT", "stdio")
    if transport in ("http", "streamable-http"):
        print("WARNING: HTTP transport is UNAUTHENTICATED — localhost only. See REMOTE_DEPLOY.md.",
              file=sys.stderr)
        mcp.run(transport="streamable-http")  # FastMCP binds 127.0.0.1 -> DNS-rebinding protection on
    else:
        mcp.run()


if __name__ == "__main__":
    main()
