"""cardiac-core MCP server — thin FastMCP wrapper over cardiac_mcp.core.

All logic lives in ``core`` (transport-agnostic, unit-testable). This module only registers those
functions as MCP tools/resources. Promoting to remote later is a one-line transport swap in
``__main__`` — the tool surface here does not change.
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from cardiac_mcp import __version__, core

mcp = FastMCP("cardiac-core", instructions=core.SERVER_INSTRUCTIONS)
# FastMCP exposes no `version`/`title` constructor kwarg, so set serverInfo.version on the
# underlying low-level server (it otherwise falls back to the installed mcp SDK version).
mcp._mcp_server.version = __version__

# Register core functions as tools with intentional annotations. FastMCP derives each tool's
# input/output schema from the signature + docstring; annotations are honest (untrusted) hints so a
# careful host can gate correctly — read-only tools vs. the one tool that executes generated code.
_READ_ONLY = dict(readOnlyHint=True, openWorldHint=False)  # pure query, self-contained
mcp.add_tool(core.simulate, title="Run quick CV simulation",          # DIRECT / exploration
             annotations=ToolAnnotations(**_READ_ONLY))
mcp.add_tool(core.build_manifest, title="Build experiment manifest",  # GATED step 1 (writes nothing)
             annotations=ToolAnnotations(**_READ_ONLY))
mcp.add_tool(core.list_experiments, title="List recorded experiments",
             annotations=ToolAnnotations(**_READ_ONLY))
mcp.add_tool(core.commit_experiment, title="Commit experiment to Lab/",  # GATED step 2 — additive write
             annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False,
                                         idempotentHint=False, openWorldHint=False))
mcp.add_tool(core.run_experiment, title="Execute a committed experiment",  # GATED step 3 — runs code
             annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True,
                                         idempotentHint=False, openWorldHint=False))


@mcp.resource("cardiac://cheatsheet", mime_type="text/markdown")
def cheatsheet() -> str:
    """Canonical cardiac_core API cheatsheet — generate simulation params against THIS, never invent API."""
    return core.read_cheatsheet()


@mcp.resource("cardiac://notebook", mime_type="text/markdown")
def notebook() -> str:
    """The Lab notebook index — every recorded experiment (date, slug, goal, engine, status, result)."""
    return core.read_notebook()


# Prompts — reusable workflow templates (the project's "recipes" as first-class MCP entry points).
@mcp.prompt(title="Measure conduction velocity")
def measure_cv(tissue: str = "healthy ventricle") -> str:
    """Guide a CV measurement through the accountability gate."""
    return (
        f"I want to measure conduction velocity in {tissue}. "
        "Read the cardiac://cheatsheet resource, then call build_manifest, show me the manifest, "
        "and only call commit_experiment after I confirm."
    )


@mcp.prompt(title="Control vs knockdown CV series")
def control_vs_knockdown(control_sigma_i: float = 1.74, knockdown_fraction: float = 0.5) -> str:
    """Guide a paired control/knockdown CV experiment + comparison."""
    return (
        "Run a paired CV experiment: a control strip and a knockdown strip with sigma_i scaled by "
        f"{knockdown_fraction}x (control sigma_i={control_sigma_i}). Use build_manifest for each, "
        "gate each, then compare the two CVs."
    )
