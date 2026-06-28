<!-- mcp-name: io.github.<user>/cardiac-core -->
# cardiac-core MCP server

A thin, host-agnostic [Model Context Protocol](https://modelcontextprotocol.io) server that exposes the
`cardiac_core` cardiac-electrophysiology simulation API to any MCP host (Claude Desktop, Claude Code, IDEs).
It is an **adapter**: all logic lives in `cardiac_mcp/core.py` (transport-agnostic, unit-tested); `server.py`
only wires those functions to MCP. Standardized against MCP spec revision **2025-11-25** (FastMCP / `mcp` SDK ≥ 1.2.0).

## Tools

Two tracks (a "quick look" vs an accountable, recorded experiment):

| Tool | Annotation | What it does |
|------|-----------|--------------|
| `simulate` | read-only | DIRECT/exploration — quick conduction-velocity run on a tissue strip; ephemeral, no `Lab/` record. |
| `build_manifest` | read-only | GATED step 1 — build a plain-text experiment manifest + a self-signed `experiment_token`; writes nothing. |
| `commit_experiment` | additive write | GATED step 2 — **refuses unless `confirmed=True` and the token verifies**; writes `Lab/{date}_{slug}/` (MANIFEST.md + run.py) + a notebook row. |
| `run_experiment` | **destructive** (runs code) | GATED step 3 — executes a committed `run.py` (restricted to `Lab/`), records the result. |
| `list_experiments` | read-only | List recorded experiments in `Lab/`. |

### The accountability gate
`build_manifest` → (you review the manifest and say "go") → `commit_experiment(token, confirmed=True)`. The token
embeds the exact manifest + params and is signed, so the committed script is provably the one you reviewed — the
server **never** runs an experiment you didn't confirm.

## Resources

- `cardiac://cheatsheet` (`text/markdown`) — the canonical `cardiac_core` API cheatsheet (the anti-hallucination
  source the model generates against).
- `cardiac://notebook` (`text/markdown`) — the `Lab/NOTEBOOK.md` index of recorded experiments.

## Prompts

- `measure_cv(tissue)` — guide a conduction-velocity measurement through the gate.
- `control_vs_knockdown(control_sigma_i, knockdown_fraction)` — guide a paired control/knockdown CV series.

## Install

The server ships in the repo's `cardiac-core` package (editable install at the repo root provides both
`cardiac_core` and the `cardiac-mcp` console script):

```bash
conda run -n heart-conduction pip install -e .
```

## Register with a host

`.mcp.json` (Claude Code, project scope) / `claude_desktop_config.json` (Claude Desktop) — same `mcpServers` key:

```json
{
  "mcpServers": {
    "cardiac-core": {
      "type": "stdio",
      "command": "/home/norepinephrine/.conda/envs/heart-conduction/bin/cardiac-mcp"
    }
  }
}
```

Run directly: `cardiac-mcp` (or `python -m cardiac_mcp`). The server logs to stderr; stdout is the protocol channel.

## Test

```bash
conda run -n heart-conduction python -m pytest cardiac_mcp/tests -q
# interactive surface check:
npx @modelcontextprotocol/inspector cardiac-mcp     # or: conda run -n heart-conduction python -m mcp dev
```

## Remote deployment

stdio (local) needs no auth. Exposing this server over HTTP requires the full auth/security stack — see
[`REMOTE_DEPLOY.md`](REMOTE_DEPLOY.md). **Do not** expose `run_experiment` beyond localhost without the sandbox
documented there (it executes generated Python).
