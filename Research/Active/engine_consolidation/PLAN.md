# PLAN: cardiac_mcp standardization — Tiers 1–4 (MCP spec 2025-11-25 audit)

Created: 2026-06-28
Engine(s): None (the `cardiac_mcp/` MCP server + `.mcp.json`; drives the shipped `cardiac_core` API — engines untouched)
Research question: [engine_consolidation](README.md)
Source: [IDEALOG.md](IDEALOG.md) — 2026-06-28 "MCP audit + blueprint" thread entry; reference table in KNOWLEDGE "Goal-2 MCP server — standardization audit (2026-06-28)"

## Objective
Bring `cardiac_mcp` (the cardiac-core MCP server, shipped 2026-06-26) from "working" to "standardized" against the
official MCP spec **revision 2025-11-25**. Four tiers, each a phase: (1) honest metadata + two path-traversal input-
validation fixes, (2) completeness (output schemas, README, prompts, installable packaging), (3) remote-readiness
(sandbox the code-executing tool + HTTP transport + an auth design doc), (4) optional registry publishing artifacts.
The public tool/resource surface and all `cardiac_core` behaviour stay unchanged — this is metadata, validation,
packaging, and docs.

## Success Criteria
- [x] Phase 1: all 5 tools carry intentional `ToolAnnotations`; `serverInfo.version == "0.1.0"`; markdown resources serve `text/markdown`; `run_experiment`/`commit_experiment` reject path-traversal inputs; new tests prove it. **DONE 2026-06-28 — 13/13 tests green.**
- [ ] Phase 2: tools return typed results → MCP `outputSchema` + `structuredContent` populated; `cardiac_mcp/README.md` exists; ≥1 prompt registered; server installs as a `cardiac-mcp` console script and `.mcp.json` no longer needs `PYTHONPATH`.
- [ ] Phase 3: `run_experiment` runs under resource limits + a provenance check; an HTTP transport switch exists (localhost-bound); `cardiac_mcp/REMOTE_DEPLOY.md` documents the auth/security stack required before any non-localhost deploy.
- [ ] Phase 4 (optional): `server.json` + README ownership marker + LICENSE validate with the MCP Inspector.
- [ ] All existing tests pass (no regressions): `cardiac_mcp/tests/` + `cardiac_core/tests/`.

## Architecture Changes
- MOD: `cardiac_mcp/server.py` — replace the `add_tool` loop with per-tool `add_tool(fn, annotations=ToolAnnotations(...))`; set `mcp._mcp_server.version`; add `mime_type` to both `@mcp.resource`; register prompts (P2); add HTTP transport switch (P3).
- MOD: `cardiac_mcp/core.py` — input validation in `run_experiment` (~376) + `build_manifest` (~282); typed `TypedDict` return models (P2); subprocess hardening in `run_experiment` (P3).
- MOD: `cardiac_mcp/__main__.py` — add `main()` entry point (console script) + transport selection (P2/P3).
- MOD: `cardiac_mcp/tests/test_core.py` — add validation tests (P1); structured-output assertions (P2).
- MOD: `.mcp.json` — switch to the `cardiac-mcp` console script, drop `PYTHONPATH` (P2).
- MOD: `./pyproject.toml` (repo root) — add `cardiac_mcp*` to the package include, a `cardiac-mcp` console script, and the `mcp>=1.2.0` dependency (P2; **Option B** — no separate `cardiac_mcp/pyproject.toml`).
- NEW: `cardiac_mcp/README.md` — tools/resources/prompts + config snippet (P2).
- NEW: `cardiac_mcp/REMOTE_DEPLOY.md` — auth/security checklist for HTTP (P3).
- NEW: `cardiac_mcp/server.json`, `LICENSE`, `Dockerfile` (P4, optional).

## Known Failures / Pitfalls (from IDEALOG + audit)
- **Do NOT conflate the official SDK's vendored FastMCP 1.x (`mcp.server.fastmcp`) with the community `fastmcp` 2.x (gofastmcp.com).** This plan targets the installed `mcp` 1.28.0 only.
- **FastMCP has no `version`/`title` constructor kwarg** — earlier assumption was wrong. `serverInfo.version` MUST be set via `mcp._mcp_server.version` (verified 2026-06-28: it flows into `create_initialization_options().server_version`).
- **Tool annotations are untrusted hints** — set them for honest UX/gating, but never rely on a *client* honoring them; the server-side validation (P1 path checks, P3 limits) is the real safety boundary.
- **stdio MUST keep stdout clean** — never `print()` to stdout in the server process; FastMCP logs to stderr (OK), and `run_experiment` MUST keep `capture_output=True` so the child's stdout never reaches the server's stdout.

---

## Phase 1: Tier 1 — Honest metadata + input-validation fixes

**Goal**: Every primitive carries correct, intentional metadata, and the two path-traversal holes are closed. Cheap, high-value, ships the honest safety profile. Independently deliverable.
**Tier**: medium
**Estimated scope**: 2 steps — server.py metadata (annotations + version + MIME); core.py input validation + tests.

### Phase Context
- The server is `cardiac_mcp/server.py`: `mcp = FastMCP("cardiac-core", instructions=core.SERVER_INSTRUCTIONS)`, then a `for _fn in (...): mcp.add_tool(_fn)` loop registering 5 tools, then two `@mcp.resource(...)` functions.
- All tool LOGIC lives in `cardiac_mcp/core.py` (transport-agnostic). Do NOT move logic into server.py.
- FastMCP 1.28.0 verified signatures: `add_tool(fn, name=None, title=None, description=None, annotations: ToolAnnotations | None = None, icons=None, meta=None, structured_output=None)`; `resource(uri, *, name=None, title=None, description=None, mime_type: str | None = None, ...)`. `ToolAnnotations` fields: `title, readOnlyHint, destructiveHint, idempotentHint, openWorldHint` (import from `mcp.types`).
- Package version is `cardiac_mcp.__version__ = "0.1.0"` (in `cardiac_mcp/__init__.py`).
- Tests run with: `conda run -n heart-conduction python -m pytest cardiac_mcp/tests/test_core.py -q` (set `PYTHONPATH=` repo root until Phase 2 makes it installable).
- Annotation policy (the truth table for our 5 tools):
  | tool | readOnlyHint | destructiveHint | idempotentHint | openWorldHint |
  |------|---|---|---|---|
  | `simulate` | True | (n/a) | (n/a) | False |
  | `list_experiments` | True | (n/a) | (n/a) | False |
  | `build_manifest` | True | (n/a) | (n/a) | False |
  | `commit_experiment` | False | False | False | False |
  | `run_experiment` | False | True | False | False |
  (`destructiveHint`/`idempotentHint` are only meaningful when `readOnlyHint=False`; omit them for the read-only tools.)

### Step 1.1: server.py — set version, per-tool annotations, resource MIME types
**Model**: opus

#### Read First
- `cardiac_mcp/server.py` (whole file, ~40 lines) — the `add_tool` loop + two resource decorators.
- `cardiac_mcp/__init__.py` — confirm `__version__`.

#### Why
Unset annotations make FastMCP fall back to spec defaults (`destructiveHint=true, openWorldHint=true`) for ALL tools — dishonest for `simulate`/`build_manifest`/`list_experiments` (pure reads) and it fails to single out `run_experiment` (the only code-runner). `serverInfo.version` is `None` → reports the SDK's `1.28.0`, misidentifying the server. Markdown served as `text/plain` renders poorly in hosts.

#### Implementation Spec
**Files to modify:** `cardiac_mcp/server.py`.
**Interfaces / Signatures:**
- `from mcp.types import ToolAnnotations`
- `from cardiac_mcp import __version__`
- After `mcp = FastMCP("cardiac-core", instructions=core.SERVER_INSTRUCTIONS)` add: `mcp._mcp_server.version = __version__`.
- Replace the `for _fn in (...): mcp.add_tool(_fn)` loop with 5 explicit calls, each passing `annotations=ToolAnnotations(...)` per the truth table, plus a human `title=`.
- Add `mime_type="text/markdown"` to both `@mcp.resource(...)` decorators.

#### Pseudocode
```python
from mcp.types import ToolAnnotations
from cardiac_mcp import __version__

mcp = FastMCP("cardiac-core", instructions=core.SERVER_INSTRUCTIONS)
mcp._mcp_server.version = __version__   # FastMCP exposes no version kwarg (verified 2026-06-28)

_RO = dict(readOnlyHint=True, openWorldHint=False)   # read-only, closed-world
mcp.add_tool(core.simulate,          title="Run quick CV simulation",      annotations=ToolAnnotations(**_RO))
mcp.add_tool(core.build_manifest,    title="Build experiment manifest",    annotations=ToolAnnotations(**_RO))
mcp.add_tool(core.list_experiments,  title="List recorded experiments",    annotations=ToolAnnotations(**_RO))
mcp.add_tool(core.commit_experiment, title="Commit experiment to Lab/",
             annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False,
                                         idempotentHint=False, openWorldHint=False))
mcp.add_tool(core.run_experiment,    title="Execute a committed experiment",
             annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True,
                                         idempotentHint=False, openWorldHint=False))

@mcp.resource("cardiac://cheatsheet", mime_type="text/markdown")
def cheatsheet() -> str: ...
@mcp.resource("cardiac://notebook", mime_type="text/markdown")
def notebook() -> str: ...
```

#### Test Spec
- `cardiac_mcp/tests/test_core.py::test_server_metadata` — import `cardiac_mcp.server.mcp`; assert `mcp._mcp_server.version == "0.1.0"`; `list_tools()` returns a `list`, so build `tools = {t.name: t for t in asyncio.run(mcp.list_tools())}` and assert `tools["simulate"].annotations.readOnlyHint is True` and `tools["run_experiment"].annotations.destructiveHint is True`. Importing the server is torch-free (`core` imports `cardiac_core` lazily) so the test is fast. (Audit M4/L5: gives this pure-wiring step its own automated check; R2-L2: look the tool up by name — `list_tools` is a list, not a name-keyed dict.)

#### Checklist
- [ ] Import `ToolAnnotations` + `__version__`.
- [ ] Set `mcp._mcp_server.version`.
- [ ] Replace loop with 5 annotated `add_tool` calls (correct truth-table values).
- [ ] Add `mime_type="text/markdown"` to both resources.
- [ ] Add `test_server_metadata` (version + annotations) to `tests/test_core.py`.

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
PYTHONPATH=$PWD conda run -n heart-conduction python -c "
import asyncio; from cardiac_mcp.server import mcp
print('version', mcp._mcp_server.version)
ts = asyncio.run(mcp.list_tools())
for t in ts: print(t.name, t.annotations.readOnlyHint, t.annotations.destructiveHint, t.annotations.openWorldHint)
"
```
Expect `version 0.1.0`; `simulate/build_manifest/list_experiments` → `readOnlyHint=True … openWorldHint=False`; `run_experiment` → `readOnlyHint=False destructiveHint=True`.

#### Exit Criteria
- [ ] `serverInfo.version` is `0.1.0`.
- [ ] All 5 tools report annotations matching the truth table.
- [ ] Both resources report `text/markdown`.

#### Risk
`mcp._mcp_server` is a private attribute — a future SDK bump could rename it. Mitigation: it's verified for 1.28.0; the version assertion in Verify catches a break immediately.

### Step 1.2: core.py — close the two path-traversal holes + tests
**Model**: opus

#### Read First
- `cardiac_mcp/core.py:362-398` — `run_experiment`; the `d = (REPO_ROOT / experiment_dir).resolve()` at ~376.
- `cardiac_mcp/core.py:246-305` — `build_manifest`; `date = date or _today()` at ~282.
- `cardiac_mcp/core.py:14-30` — module constants (`REPO_ROOT`, `LAB`), confirm `re` is imported.

#### Why
Spec MUST: validate all tool inputs. `(REPO_ROOT / experiment_dir)` resets to absolute if `experiment_dir` is absolute (`Path("/x")` semantics) and `..` segments escape — so `run_experiment` will execute ANY `run.py` on disk. `commit_experiment` builds `LAB / f"{date}_{slug}"` from the model-supplied `date`, which is unsanitized (only `slug` is `_slugify`'d) — a `date` like `../../x` traverses out of `Lab/`. Both are low-risk locally but become real attack surface the moment the server is remote (Phase 3); fix at the source now.

#### Implementation Spec
**Files to modify:** `cardiac_mcp/core.py`.
- In `run_experiment`, after computing `d`, before the `run.py` existence check: reject `d` not under `LAB`.
- In `build_manifest`, right after `date = date or _today()`: reject a `date` not matching `^\d{4}-\d{2}-\d{2}$`.
**Interfaces:** use `Path.is_relative_to` (Python 3.11). `LAB` and `REPO_ROOT` already module-level.

#### Pseudocode
```python
# run_experiment, after: d = (REPO_ROOT / experiment_dir).resolve()
if not d.is_relative_to(LAB.resolve()):
    raise ValueError(f"run_experiment(): experiment_dir must be inside Lab/ (got {experiment_dir!r}).")

# build_manifest, after: date = date or _today()
if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
    raise ValueError(f"build_manifest(): date must be YYYY-MM-DD (got {date!r}).")
```

#### Test Spec
- `cardiac_mcp/tests/test_core.py::test_run_experiment_rejects_traversal` — assert BOTH an absolute path `core.run_experiment("/etc")` AND a `..`-escape `core.run_experiment("../../etc")` raise `ValueError` matching "inside Lab/". The guard is lexical after `.resolve()`, so it raises whether or not `Lab/` exists and never executes a subprocess. (Audit M3: keep it robust by asserting both input shapes; the test does not depend on the real `Lab/` contents.)
- `cardiac_mcp/tests/test_core.py::test_build_manifest_rejects_bad_date` — `core.build_manifest(goal="x", date="../../x")` raises `ValueError` matching "YYYY-MM-DD"; a valid `date="2026-06-28"` still succeeds.

#### Checklist
- [ ] Add the `is_relative_to(LAB)` guard in `run_experiment`.
- [ ] Add the date-regex guard in `build_manifest`.
- [ ] Add both tests.
- [ ] Confirm `re` is imported (it is — used by `_slugify`/`_parse_cv`).

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
PYTHONPATH=$PWD conda run -n heart-conduction python -m pytest cardiac_mcp/tests/test_core.py -q
```
Expect all tests pass (prior 10 + Step 1.1's `test_server_metadata` + 2 traversal/date tests = 13).

#### Exit Criteria
- [ ] Traversal inputs raise before any subprocess/file write.
- [ ] Valid inputs unaffected (existing gate/commit tests still green).

#### Risk
`is_relative_to` on a non-existent path is fine (pure lexical after `.resolve()`). Mitigation: `.resolve()` is called first so symlink games are normalized.

### Phase 1 Verification
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
PYTHONPATH=$PWD conda run -n heart-conduction python -m pytest cardiac_mcp/tests/test_core.py -q
```

### Phase 1 Exit Criteria
- [ ] All new tests pass; all existing `cardiac_mcp` tests pass (10 prior + 3 new = 13 total).
- [ ] `serverInfo.version == 0.1.0`, annotations correct, resources `text/markdown`.
- [ ] Path-traversal inputs rejected.

### Phase 1 Cleanup
- [ ] No `print()` to stdout added anywhere in the server process.
- [ ] No `cardiac_core`/torch import added at server.py top level (keep the lazy import in core).
- [ ] Annotation truth-table values double-checked against the Phase Context table.

**-> Commit point: `git commit` after Phase 1 passes** (suggested branch: `mcp-standardization`; we are on `main`).

---

## Phase 2: Tier 2 — Completeness (schemas, prompts, README, installable)

**Goal**: Structured outputs, discoverable docs, prompt templates, and a proper installable package so `.mcp.json` drops the `PYTHONPATH` hack. Independently deliverable.
**Tier**: medium
**Estimated scope**: 4 steps.

### Phase Context
- Tools currently return bare `dict` → FastMCP emits no `outputSchema` and returns results as JSON **text** only (`structuredContent` is `None`; verified in the 2026-06-26 roundtrip).
- FastMCP derives `outputSchema` + returns `structuredContent` when the tool's return is a typed structure (`TypedDict`/dataclass/pydantic). Alternatively `add_tool(..., structured_output=True)` forces it.
- `cardiac_core` is editable-installed (scoped to `cardiac_core*`); `cardiac_mcp` is NOT a package yet (reached via `PYTHONPATH` in `.mcp.json`).
- Keep the public surface identical — typing returns must not change field names/values.

### Step 2.1: Typed return models → `outputSchema` + `structuredContent`
**Model**: opus

#### Read First
- `cardiac_mcp/core.py` — the `return {...}` of `simulate` (~141), `build_manifest` (~298), `commit_experiment` (~354), `run_experiment` (~389), `list_experiments` (~404). (Audit L2: anchors corrected to actual lines.)
- FastMCP behaviour: confirm with `python -c "from mcp.server.fastmcp import FastMCP; help(FastMCP.add_tool)"` that `structured_output` + typed returns produce an output schema.

#### Why
Spec SHOULD: provide `outputSchema` and return conforming `structuredContent` for structured results — it lets any host validate + parse results structurally instead of re-parsing JSON text. This is the single highest-value completeness item for multi-host use.

#### Implementation Spec
**Files to modify:** `cardiac_mcp/core.py`.
- Define `TypedDict` result models (top of core, after imports): `SimulateResult`, `ManifestResult`, `CommitResult`, `RunResult`, `ListResult`. Use concrete field types; nested objects may be `dict[str, Any]` where shape varies (still yields a valid object schema).
- Annotate each tool's return type (`-> SimulateResult`, etc.). Do NOT change the returned values.
- If FastMCP does not auto-emit a schema from `TypedDict` alone, pass `structured_output=True` in the corresponding `add_tool` call in `server.py`.

#### Pseudocode
```python
from typing import TypedDict, Any
class SimulateResult(TypedDict):
    engine: str; ionic: str; grid: dict[str, Any]; conductivity: dict[str, Any]
    cv_cm_per_s: float | None; cv_indices: dict[str, int]; activated: bool
    frames: list[int]; note: str
# ... ManifestResult{manifest_text:str, slug:str, experiment_token:str, next:str}
# ... CommitResult{experiment_dir:str, files:list[str], status:str, next:str}
# ... RunResult{experiment_dir:str, status:str, cv_cm_per_s:float|None, returncode:int, stdout:str, stderr:str}
# ... ListResult{count:int, experiments:list[str]}  (always include count → stable schema)
def simulate(...) -> SimulateResult: ...
```

#### Test Spec
- `cardiac_mcp/tests/test_core.py::test_structured_output_via_client` (slow/integration, optional) — spawn the stdio server, `call_tool("build_manifest", {...})`, assert `res.structuredContent` is a dict with `slug`. (build_manifest writes nothing — safe.)
- `cardiac_mcp/tests/test_core.py::test_list_experiments_always_has_count` (Audit M1) — with `tmp_lab` empty → `core.list_experiments()["count"] == 0`; after one committed experiment → `count == 1`. Covers the empty-case shape change (no prior test existed for `list_experiments`).
- Minimum: a boot check that each tool's `outputSchema` is non-None via `list_tools()`.

#### Checklist
- [ ] Define 5 `TypedDict` models.
- [ ] Annotate the 5 return types (values unchanged).
- [ ] Make `list_experiments` always return `count` (so the schema is stable).
- [ ] Add `test_list_experiments_always_has_count` (empty + non-empty `tmp_lab`).
- [ ] If needed, set `structured_output=True` per tool in server.py.
- [ ] Verify `outputSchema` present + `structuredContent` populated.

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
PYTHONPATH=$PWD conda run -n heart-conduction python -c "
import asyncio; from cardiac_mcp.server import mcp
ts = asyncio.run(mcp.list_tools()); print({t.name: bool(t.outputSchema) for t in ts})"
PYTHONPATH=$PWD conda run -n heart-conduction python -m pytest cardiac_mcp/tests/test_core.py -q
```
Expect every tool `True` (has outputSchema); tests green.

#### Exit Criteria
- [ ] All tools expose `outputSchema`; calls return `structuredContent`.
- [ ] No renames or value changes to EXISTING fields. (Audit M1 — intentional, documented exception: `list_experiments` now ALWAYS includes `count` — a schema-stabilizing addition to the previously-`count`-less empty case, covered by `test_list_experiments_always_has_count`.)

#### Risk
Over-typing nested dicts can make FastMCP's schema generation brittle. Mitigation: use `dict[str, Any]` for variable nested shapes; only the top-level fields need precise types.

### Step 2.2: Register prompt templates
**Model**: opus

#### Read First
- `cardiac_mcp/server.py` — where resources are registered (add prompts alongside).
- `.claude/skills/sim-experiment/reference/recipes.md` — the recipe vocabulary (R1 CV, R4 edge/bath) to mirror.

#### Why
Prompts are user-controlled workflow templates — exactly our "recipes." Optional in the spec but high-fit: they let a host surface "measure CV" / "control vs knockdown" as first-class entry points, reinforcing the guided, accountable workflow.

#### Implementation Spec
**Files to modify:** `cardiac_mcp/server.py`.
- Add `@mcp.prompt()` functions returning a `str` (a user message).
- `measure_cv(tissue: str = "healthy ventricle")` — guides `build_manifest` → gate → `commit_experiment`.
- `control_vs_knockdown(control_sigma_i: float = 1.74, knockdown_fraction: float = 0.5)` — guides a paired CV series + comparison.

#### Pseudocode
```python
@mcp.prompt(title="Measure conduction velocity")
def measure_cv(tissue: str = "healthy ventricle") -> str:
    return (f"I want to measure conduction velocity in {tissue}. "
            "Read cardiac://cheatsheet, then call build_manifest, show me the manifest, "
            "and only commit_experiment after I confirm.")

@mcp.prompt(title="Control vs knockdown CV series")
def control_vs_knockdown(control_sigma_i: float = 1.74, knockdown_fraction: float = 0.5) -> str:
    return ("Run a paired CV experiment: a control strip and a knockdown strip with sigma_i scaled by "
            f"{knockdown_fraction}× (control sigma_i={control_sigma_i}). Use build_manifest for each, "
            "gate each, then compare the two CVs.")
```

#### Test Spec
- Boot check: `asyncio.run(mcp.list_prompts())` returns ≥2 prompts with the expected names.

#### Checklist
- [ ] Add 2 `@mcp.prompt()` functions with titles.
- [ ] Verify they list.

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
PYTHONPATH=$PWD conda run -n heart-conduction python -c "
import asyncio; from cardiac_mcp.server import mcp
print([p.name for p in asyncio.run(mcp.list_prompts())])"
```
Expect `measure_cv` and `control_vs_knockdown` present.

#### Exit Criteria
- [ ] ≥2 prompts registered and listable; server still boots.

#### Risk
Prompt arg types must be simple (str/float/int) for schema gen. Mitigation: keep args primitive.

### Step 2.3: `cardiac_mcp/README.md`
**Model**: sonnet

#### Read First
- `cardiac_mcp/core.py` docstrings (tool descriptions) + `server.py` (resources/prompts) — to list the surface accurately.
- `.mcp.json` — to copy the exact config snippet.

#### Why
The README is the core "supporting document" the audit flagged — and the registry's ownership check (Phase 4) keys off it. It's how any host/user learns the surface without reading code.

#### Implementation Spec
**Files to create:** `cardiac_mcp/README.md` — sections: what it is (adapter over `cardiac_core`), Tools (5, one-line each + annotation), Resources (2), Prompts (2), the accountability gate (build→confirm→commit), Install/run (`python -m cardiac_mcp` + the `.mcp.json` snippet), Testing (pytest + MCP Inspector `uv run mcp dev` / `npx @modelcontextprotocol/inspector`).
Note (R2-L4): the `.mcp.json` snippet here is PROVISIONAL until Step 2.4 finalizes the launch command — 2.4's checklist re-syncs it. If practical, author the Install/run section last (after 2.4) to avoid a brief stale snippet.

#### Pseudocode
```markdown
# cardiac-core MCP server
Adapter exposing the `cardiac_core` simulation API over MCP.
## Tools     simulate [read-only] · build_manifest [read-only] · commit_experiment · run_experiment [destructive] · list_experiments [read-only]
## Resources cardiac://cheatsheet · cardiac://notebook   (text/markdown)
## Prompts   measure_cv · control_vs_knockdown
## Accountability gate   build_manifest → (you confirm) → commit_experiment
## Install / run         <.mcp.json snippet — provisional until Step 2.4>
## Testing               pytest cardiac_mcp/tests · MCP Inspector: `uv run mcp dev`
```

#### Test Spec
- None (doc). Lint: markdown renders; the `.mcp.json` snippet matches the real file.

#### Checklist
- [ ] Tools/resources/prompts lists match the code.
- [ ] Config snippet matches `.mcp.json` (update after Step 2.4 if the command changes).

#### Verify
```bash
test -f /home/norepinephrine/Documents/Heart-Conduction/cardiac_mcp/README.md && echo OK
```

#### Exit Criteria
- [ ] README accurately lists the current surface.

#### Risk
README drifts from code. Mitigation: re-check at each phase commit.

### Step 2.4: Make `cardiac_mcp` installable via the ROOT package; simplify `.mcp.json`
**Model**: opus

#### Read First
- `cardiac_mcp/__main__.py` — current `if __name__ == "__main__": mcp.run()`.
- `./pyproject.toml` (REPO ROOT — the ONLY pyproject; `name="cardiac-core"`, setuptools backend, `[tool.setuptools.packages.find] where=["."] include=["cardiac_core*"]`, `requires-python=">=3.11"`, no `[project.scripts]`/`dependencies` yet). **There is NO `cardiac_core/pyproject.toml`.**
- `.mcp.json` — current `command`/`args`/`env`.
- Confirm the existing single editable install: `conda run -n heart-conduction pip list --editable` (expect exactly one: `cardiac-core` rooted at the repo).

#### Why
The `PYTHONPATH` hack in `.mcp.json` is fragile/non-standard; the SDK convention is a console-script entry point. **Option B (audit H1+H2):** extend the EXISTING root `cardiac-core` editable rather than create a SECOND editable for `cardiac_mcp` — two editables over the same repo tree risk shadowing (audit flagged it), and the earlier plan also pointed at a non-existent `cardiac_core/pyproject.toml` and prescribed hatchling while the repo uses setuptools. One install, one console script, the repo's existing setuptools backend (no hatchling divergence, no `requires-python` mismatch — resolves L1). **Accepted trade-off:** the `cardiac-core` distribution now contains `cardiac_mcp` and depends on `mcp` — fine for this single-repo workspace (the user wants them together).

#### Implementation Spec
**Files to modify:**
- `./pyproject.toml` (repo root):
  - `[tool.setuptools.packages.find]` → `include = ["cardiac_core*", "cardiac_mcp*"]`
  - add to `[project]` → `dependencies = ["mcp>=1.2.0"]`
  - add `[project.scripts]` → `cardiac-mcp = "cardiac_mcp.__main__:main"`
  - leave `requires-python = ">=3.11"` (matches the 3.11.14 env; code uses `is_relative_to` (3.9+) and `X | None` (3.10+) — both fine).
- `cardiac_mcp/__main__.py` — add `def main(): mcp.run()`; keep `if __name__ == "__main__": main()`.
- `.mcp.json` — set `command` to the env console script (`/home/norepinephrine/.conda/envs/heart-conduction/bin/cardiac-mcp`), drop `args`/`PYTHONPATH`, keep `"type": "stdio"`.
**Reinstall:** `conda run -n heart-conduction pip install -e .` at the repo ROOT (refreshes the `cardiac-core` editable to also expose `cardiac_mcp` + the console script).
**Do NOT create** a separate `cardiac_mcp/pyproject.toml`.

#### Pseudocode
```toml
# ./pyproject.toml — additions to the EXISTING file
[project]
# ...existing name/version/description/requires-python (>=3.11)...
dependencies = ["mcp>=1.2.0"]

[project.scripts]
cardiac-mcp = "cardiac_mcp.__main__:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["cardiac_core*", "cardiac_mcp*"]
```
```python
# cardiac_mcp/__main__.py
def main():
    mcp.run()
if __name__ == "__main__":
    main()
```

#### Test Spec
- After reinstall: BOTH `import cardiac_core` and `import cardiac_mcp` resolve (the existing cardiac-core editable not clobbered), `cardiac-mcp` is on the env PATH, and it launches the stdio server (spawn + initialize via the stdio client, then exit).

#### Checklist
- [ ] Edit the ROOT `./pyproject.toml` (include filter + `mcp` dependency + console script).
- [ ] Add `main()` to `cardiac_mcp/__main__.py`.
- [ ] `pip install -e .` at the repo root.
- [ ] Assert `import cardiac_core; import cardiac_mcp` BOTH succeed (no clobber).
- [ ] Update `.mcp.json` to the console-script command; drop PYTHONPATH.
- [ ] Re-run a stdio client roundtrip with the new launch command.
- [ ] Update `cardiac_mcp/README.md` snippet to match.

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
conda run -n heart-conduction pip install -e . -q
conda run -n heart-conduction python -c "import cardiac_core, cardiac_mcp; print('both import OK')"
conda run -n heart-conduction which cardiac-mcp
conda run -n heart-conduction python -m pytest cardiac_mcp/tests/test_core.py -q   # no PYTHONPATH now
```
Expect `both import OK`; a `cardiac-mcp` path; tests green without `PYTHONPATH`.

#### Exit Criteria
- [ ] `import cardiac_core` AND `import cardiac_mcp` both work without `PYTHONPATH`.
- [ ] `cardiac-mcp` console script launches the stdio server.
- [ ] `.mcp.json` no longer sets `PYTHONPATH`; still exactly one editable install.

#### Risk
Re-running `pip install -e .` could in principle disturb the cardiac-core install. Mitigation: the dual-`import` assertion in Verify catches any clobber immediately; the change is additive (wider include + new script + dep), not a rename. (R2-L1 note: `cardiac_mcp*` also matches `cardiac_mcp.tests` — harmless for the editable install, exactly as `cardiac_core*` already captures `cardiac_core.tests`; if a Phase-4 published wheel should omit test code, add an `exclude=["*.tests*"]` to the find filter then.)

### Phase 2 Verification
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
conda run -n heart-conduction python -m pytest cardiac_mcp/tests/test_core.py -q
conda run -n heart-conduction python -c "import asyncio; from cardiac_mcp.server import mcp; \
print('tools', [(t.name, bool(t.outputSchema)) for t in asyncio.run(mcp.list_tools())]); \
print('prompts', [p.name for p in asyncio.run(mcp.list_prompts())])"
```

### Phase 2 Exit Criteria
- [ ] outputSchema/structuredContent on all tools; ≥2 prompts; README present; installable console script; `.mcp.json` simplified; all tests green.

### Phase 2 Cleanup
- [ ] Exactly one editable install (`cardiac-core` at root, now exposing `cardiac_mcp`); `mcp` pinned once in the root pyproject; no separate `cardiac_mcp/pyproject.toml` left behind.
- [ ] README surface matches code exactly.
- [ ] `.mcp.json` and README config snippet identical.

**-> Commit point: `git commit` after Phase 2 passes.**

---

## Phase 3: Tier 3 — Remote-readiness (sandbox + transport + auth design)

**Goal**: Make the code-executing tool safe under resource limits + provenance now, add a localhost-bound HTTP transport switch, and DOCUMENT the full auth/security stack required before any non-localhost deploy. The auth implementation itself is deferred to a real deployment target.
**Tier**: large
**Estimated scope**: 3 steps (3.1 implement; 3.2 implement minimal + design; 3.3 design doc).

### Phase Context
- `run_experiment` (core.py ~362) runs `subprocess.run([sys.executable, run_py], cwd=REPO_ROOT, capture_output=True, timeout=...)`. Phase 1 already restricts `experiment_dir` to `Lab/`.
- Spec: a code-executing tool SHOULD be sandboxed (filesystem restricted, least privilege); stdio needs no auth (env creds) but HTTP requires the full OAuth 2.1 stack + Origin checks + secure sessions + SSRF defenses.
- FastMCP auto-enables DNS-rebinding protection when `host` ∈ {127.0.0.1, localhost, ::1} (verified in the constructor source). Keep HTTP localhost-only until auth lands.
- This phase does NOT make the server safely public — it makes it *hardened locally* and *documented for remote*.

### Step 3.1: Harden `run_experiment` (resource limits + provenance)
**Model**: opus

#### Read First
- `cardiac_mcp/core.py:362-398` — `run_experiment` (post-Phase-1, with the `is_relative_to(LAB)` guard).
- `cardiac_mcp/core.py` `render_run_script` — the generated header line (used as the provenance marker).

#### Why
Even local, a code-runner should fail safe: bound CPU/memory/output so a runaway/malformed script can't hang or OOM the host, and only run scripts WE generated (not arbitrary `run.py` dropped in `Lab/`). The concrete, now-deliverable slice of the spec's "sandbox code execution"; full containerization is deferred to 3.3.

#### Implementation Spec
**Files to modify:** `cardiac_mcp/core.py` (`run_experiment`).
- Provenance: read `run.py`; require the EXACT generated-header marker `"generated by the cardiac-core MCP server"` (verified present at `core.py:194` in `render_run_script` — match it literally). Else raise.
- Resource limits (POSIX only): `preexec_fn` setting `RLIMIT_CPU` (≈ `timeout_s`) and `RLIMIT_FSIZE` (cap any file the script writes). **Do NOT set `RLIMIT_AS`** (Audit M2) — it caps *virtual* address space, which CUDA/torch reserve in the tens of GB at import, so it aborts torch init even when real memory use is tiny. Real RSS/memory isolation belongs in the Step 3.3 container, not here. Guard `preexec_fn=None` on non-POSIX.
- Keep `timeout`, `capture_output=True`, `cwd=REPO_ROOT`.

#### Pseudocode
```python
import resource, platform
def _limits():
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))      # ~ timeout_s
    resource.setrlimit(resource.RLIMIT_FSIZE, (FSIZE_BYTES, FSIZE_BYTES))    # cap output file size
    # NO RLIMIT_AS — virtual-AS cap aborts torch/CUDA init even at tens of GB (Audit M2)

# in run_experiment, after the Phase-1 LAB guard + run_py existence:
if "generated by the cardiac-core MCP server" not in run_py.read_text():
    raise ValueError("run_experiment(): run.py is not a cardiac-core-generated script; refusing to execute.")
preexec = _limits if platform.system() != "Windows" else None
proc = subprocess.run([sys.executable, str(run_py)], cwd=str(REPO_ROOT),
                      capture_output=True, text=True, timeout=timeout_s, preexec_fn=preexec)
```

#### Test Spec
- `test_run_experiment_rejects_foreign_script` (Audit H3) — under `tmp_lab` (which monkeypatches `core.LAB` to a tmp dir), create `tmp_lab/"2026-06-28_x"/run.py` whose content LACKS the marker, then call `core.run_experiment(str(tmp_lab / "2026-06-28_x"))` — i.e. pass the **ABSOLUTE** dir so `(REPO_ROOT / abs).resolve()` collapses to that path and PASSES the Phase-1 `is_relative_to(LAB)` guard, reaching the provenance check; assert `ValueError` matching "not a cardiac-core-generated script". (A relative `"Lab/..._x"` would be rejected by the LAB guard first — passing for the WRONG reason / wrong message.)
- `test_run_experiment_under_limits` (slow, **MANDATORY for this step** — Audit R2-L5) — under `tmp_lab`, `build_manifest`→`commit_experiment` a real small-grid experiment, then `run_experiment(experiment_dir)`; assert `status=="done"` with a physiological CV. This is the ONLY test that exercises the actual **subprocess + `preexec_fn` limits path**: the in-process `test_simulate_end_to_end` calls `core.simulate()` directly and never touches `run_experiment`. Confirms the CPU/FSIZE limits don't break a real torch run in the child process.

#### Checklist
- [ ] Provenance marker check (exact string from `render_run_script`, `core.py:194`).
- [ ] `_limits` helper (CPU + FSIZE only, NO `RLIMIT_AS`) + `preexec_fn` (POSIX-guarded).
- [ ] Foreign-script test passes the ABSOLUTE `tmp_lab` dir (H3).
- [ ] `test_run_experiment_under_limits` (commit→run a real experiment through the subprocess) — MANDATORY.
- [ ] Tune CPU limit ≥ `timeout_s`; FSIZE generous (e.g. ≥1 GB for media).
- [ ] Tests.

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
# The @slow end-to-end is NOT auto-deselected (conftest only registers the marker), so a plain run
# EXECUTES it — and it is the ONLY check that the CPU/FSIZE limits don't break torch init. Confirm it ran.
conda run -n heart-conduction python -m pytest cardiac_mcp/tests/test_core.py -q -v | grep -i "under_limits\|foreign"
```
Expect both `test_run_experiment_rejects_foreign_script` and `test_run_experiment_under_limits` PASSED.

#### Exit Criteria
- [ ] Foreign/non-generated scripts refused; a real committed experiment completes `done` THROUGH the subprocess under the CPU/FSIZE limits (`test_run_experiment_under_limits` actually ran — it's not auto-deselected).

#### Risk
Dropping `RLIMIT_AS` leaves no virtual-memory cap (intentional — it would break torch). A runaway-memory script is bounded only by `RLIMIT_CPU`/`timeout`/OS until the 3.3 container lands. Mitigation: documented; CPU+timeout bound runtime; true memory isolation is explicitly a container concern (3.3).

### Step 3.2: HTTP transport switch (localhost-only) + Origin protection
**Model**: opus

#### Read First
- `cardiac_mcp/__main__.py` — `main()`.
- FastMCP `run` signature: `python -c "from mcp.server.fastmcp import FastMCP; import inspect; print(inspect.signature(FastMCP.run))"` (confirm `transport` arg + `streamable-http`).

#### Why
Delivers "structured for remote later" without a rewrite: select transport by env var, bind HTTP to 127.0.0.1 (FastMCP then auto-enables DNS-rebinding/Origin protection). Stays local-only/unauthenticated by design until 3.3's auth lands.

#### Implementation Spec
**Files to modify:** `cardiac_mcp/__main__.py`.
- `main()` reads `CARDIAC_MCP_TRANSPORT` (default `stdio`); if `http`/`streamable-http`, call `mcp.run(transport="streamable-http")` (host stays FastMCP default 127.0.0.1 → DNS-rebinding protection on). Print a one-line stderr warning that HTTP is unauthenticated/localhost-only.

#### Pseudocode
```python
import os, sys
def main():
    t = os.environ.get("CARDIAC_MCP_TRANSPORT", "stdio")
    if t in ("http", "streamable-http"):
        print("WARNING: HTTP transport is UNAUTHENTICATED — localhost only. See REMOTE_DEPLOY.md.", file=sys.stderr)
        mcp.run(transport="streamable-http")
    else:
        mcp.run()
```

#### Test Spec
- Smoke (manual/optional): `CARDIAC_MCP_TRANSPORT=http` starts and serves on `http://127.0.0.1:8000/mcp`.

#### Checklist
- [ ] Env-var transport switch.
- [ ] stderr warning for HTTP mode.
- [ ] Confirm `streamable-http` is the correct transport string for 1.28.0.

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
# Launch HTTP mode with the Bash tool's run_in_background (do NOT use a foreground `sleep` — the
# sandbox blocks it). Capture the child PID for a targeted kill (never a blanket pkill).
CARDIAC_MCP_TRANSPORT=http conda run -n heart-conduction cardiac-mcp    # run_in_background: true
# once listening, probe the DEFAULT FastMCP endpoint (port=8000, streamable_http_path="/mcp"):
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/mcp
# then kill ONLY that child PID.
```
Expect a non-200 (e.g. 400/406 without the MCP `Accept` header) — proves it's listening + rejecting non-MCP requests.

#### Exit Criteria
- [ ] stdio unchanged (default); HTTP mode serves on localhost with rebinding protection.

#### Risk
Port conflict / lingering process; the curl target hardcodes the FastMCP DEFAULTS (`port=8000`, `streamable_http_path="/mcp"`) — a `FASTMCP_PORT`/`FASTMCP_*` settings override would change them (Audit L3). Mitigation: kill the specific child PID (process-kill-safety); document the port assumption; derive the URL from the configured port if non-default.

### Step 3.3: `REMOTE_DEPLOY.md` — auth/security checklist (design, deferred impl)
**Model**: opus

#### Read First
- KNOWLEDGE "Goal-2 MCP server — standardization audit (2026-06-28)" — the Remote (HTTP) delta + Security bullets.

#### Why
Going public is a project unto itself; the responsible deliverable now is a precise, spec-cited checklist so a future deploy can't skip a MUST. It also records WHY HTTP stays localhost-only today.

#### Implementation Spec
**Files to create:** `cardiac_mcp/REMOTE_DEPLOY.md` — sections, each citing the 2025-11-25 spec:
- Auth: OAuth 2.1 + PKCE(S256); RFC 9728 Protected Resource Metadata; RFC 8707 Resource Indicators (audience binding); per-request `Authorization` bearer (never via session); registered exact redirect URIs.
- Transport/session: `Origin` validation → 403; secure non-deterministic session IDs; MUST NOT authenticate via session; bind `<user_id>:<session_id>`.
- Network: SSRF defenses (HTTPS, block private/loopback/link-local incl. 169.254.169.254); no token passthrough (validate audience == this server).
- Code-execution: containerize `run_experiment` (filesystem scoped to `Lab/`, no network, non-root, per-call ephemeral) — REQUIRED before remote, not just local limits.
- A "do NOT expose to non-localhost until all the above are MUST-complete" gate.

#### Pseudocode
```markdown
# REMOTE_DEPLOY.md   (spec revision: MCP 2025-11-25)
## Auth        OAuth 2.1 + PKCE(S256) · RFC 9728 Protected Resource Metadata · RFC 8707 resource indicators · per-request Authorization bearer
## Transport   Origin header → 403 · secure non-deterministic session IDs · MUST NOT authenticate via session
## Network     SSRF: HTTPS only, block private/loopback/link-local (incl. 169.254.169.254) · no token passthrough (audience == this server)
## Code-exec   containerize run_experiment: filesystem scoped to Lab/, no network, non-root, per-call ephemeral
## GATE        do NOT expose beyond localhost until EVERY MUST above is complete
```

#### Test Spec
- None (design doc).

#### Checklist
- [ ] All MUST items from the audit captured with spec citations.
- [ ] The "localhost-only until complete" gate stated.

#### Verify
```bash
test -f /home/norepinephrine/Documents/Heart-Conduction/cardiac_mcp/REMOTE_DEPLOY.md && echo OK
```

#### Exit Criteria
- [ ] Checklist complete and spec-cited; referenced from README + the HTTP-mode stderr warning.

#### Risk
Doc rots vs. spec revisions. Mitigation: stamp the spec revision (2025-11-25) at the top.

### Phase 3 Verification
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
conda run -n heart-conduction python -m pytest cardiac_mcp/tests/test_core.py -q
test -f cardiac_mcp/REMOTE_DEPLOY.md && echo doc-ok
```

### Phase 3 Exit Criteria
- [ ] `run_experiment` hardened (limits + provenance); foreign scripts refused.
- [ ] HTTP transport switch works, localhost-bound; stdio default unchanged.
- [ ] `REMOTE_DEPLOY.md` complete; all tests green.

### Phase 3 Cleanup
- [ ] No secrets/paths leaked in logs or tool outputs.
- [ ] HTTP mode prints the unauthenticated/localhost warning.
- [ ] Resource-limit values documented in `run_experiment` docstring.

**-> Commit point: `git commit` after Phase 3 passes.**

---

## Phase 4: Tier 4 — Registry publishing artifacts (OPTIONAL)

**Goal**: Only if public discoverability is wanted — the artifacts that make `cardiac-mcp` publishable to the official MCP registry. Skippable; nothing else depends on it.
**Tier**: medium
**Estimated scope**: 2 steps.

### Phase Context
- Required ONLY to publish to registry.modelcontextprotocol.io; the server is fully usable (local + the docs above) without it.
- Reverse-DNS name: `io.github.<user>/cardiac-core`; `server.json` version immutable + semver; README needs an ownership marker for the PyPI/registry check.

### Step 4.1: `server.json` + README ownership marker + reverse-DNS name
**Model**: opus
#### Read First
- The CURRENT `server.json` schema — WebFetch `https://static.modelcontextprotocol.io/schemas/<current>/server.schema.json` and the registry "generic-server-json" + "package-types" docs (field names + the PyPI ownership-marker rule).
- `cardiac_mcp/README.md` (P2.3) and the ROOT `./pyproject.toml` (`name`/`version` to mirror).
- The user's GitHub handle (for the reverse-DNS namespace) — ASK if unknown.
#### Why
`server.json` is the registry manifest; the README marker proves ownership. The concrete "supporting documents" for distribution. **Option-B consequence:** the publishable PyPI package is `cardiac-core` (it now bundles `cardiac_mcp`), so the manifest `packages.identifier` and the ownership marker live with the **root** `cardiac-core` package — NOT a separate `cardiac-mcp` PyPI dist. (If you later want `cardiac_mcp` published independently, that reopens the 2-package question from Step 2.4; default to the bundled dist.)
#### Implementation Spec
**Files to create:** `cardiac_mcp/server.json` — `$schema` (current static URL), `name="io.github.<user>/cardiac-core"`, `description`, `version="0.1.0"`, `repository`, `packages[{registryType:"pypi", identifier:"cardiac-core", version:"0.1.0", transport:{type:"stdio"}}]`.
**Files to modify:** the ROOT package README (the one PyPI publishes — i.e. the repo `README` for the `cardiac-core` dist) gets `<!-- mcp-name: io.github.<user>/cardiac-core -->`; mirror it in `cardiac_mcp/README.md` for discoverability. The marker MUST match `server.json` `name`.
#### Pseudocode
```json
{
  "$schema": "https://static.modelcontextprotocol.io/schemas/<current>/server.schema.json",
  "name": "io.github.<user>/cardiac-core",
  "description": "Run cardiac electrophysiology simulations (cardiac_core) over MCP.",
  "version": "0.1.0",
  "repository": {"url": "https://github.com/<user>/Heart-Conduction", "source": "github"},
  "packages": [{"registryType": "pypi", "identifier": "cardiac-core", "version": "0.1.0",
                "transport": {"type": "stdio"}}]
}
```
#### Test Spec
- Validate `server.json` against the published schema (`mcp-publisher --dry-run` if installed, else JSON-Schema validate).
#### Checklist
- [ ] Resolve the reverse-DNS name (GitHub handle).
- [ ] Write `cardiac_mcp/server.json` (identifier = `cardiac-core` per Option B).
- [ ] Add the ownership marker to the published README; mirror in `cardiac_mcp/README.md`.
- [ ] Confirm marker == `server.json` `name`.
#### Verify
```bash
conda run -n heart-conduction python -c "import json; json.load(open('cardiac_mcp/server.json')); print('valid json')"
```
#### Exit Criteria
- [ ] `server.json` parses; `name` matches the README marker; reverse-DNS chosen; identifier = the bundled `cardiac-core` dist.
#### Risk
Publishing requires the PyPI package to exist first, and bundling `cardiac_mcp` into `cardiac-core` couples their release cadence. Mitigation: keep Phase 4 dry-run only unless actually publishing; revisit a split package if independent release is needed.

### Step 4.2: LICENSE + Dockerfile + MCP Inspector validation
**Model**: sonnet
#### Read First
- Repo root — check whether a `LICENSE` already exists (`ls LICENSE*`); confirm the project's license policy before adding one (ASK the user if unstated — do not invent a license).
- ROOT `./pyproject.toml` (the install target inside the container) and the env `cardiac-mcp` console script.
- MCP Inspector docs (`modelcontextprotocol.io/docs/tools/inspector`).
#### Why
Reference-quality servers ship a LICENSE + Dockerfile; the Inspector is the standard interactive validator. Under Option B the container installs the ROOT package (brings `cardiac_core` + `cardiac_mcp` + `mcp`) and uses the `cardiac-mcp` entrypoint.
#### Implementation Spec
**Files to create:**
- `LICENSE` (match the repo's chosen policy — confirm first).
- `cardiac_mcp/Dockerfile` — `python:3.11-slim`; copy the repo; `pip install .` at root (installs `cardiac-core` incl. `cardiac_mcp` + CPU-only torch); `ENTRYPOINT ["cardiac-mcp"]`.
**Validate:** `npx @modelcontextprotocol/inspector cardiac-mcp` (or `uv run mcp dev`) — exercise tools/resources/prompts interactively.
#### Pseudocode
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir .
ENTRYPOINT ["cardiac-mcp"]
```
#### Test Spec
- Manual via Inspector: lists 5 annotated tools, 2 `text/markdown` resources, 2 prompts; a `simulate` call returns `structuredContent`.
#### Checklist
- [ ] Confirm + add `LICENSE`.
- [ ] Write `cardiac_mcp/Dockerfile` (CPU-only torch, root install, `cardiac-mcp` entrypoint).
- [ ] Run the MCP Inspector against `cardiac-mcp`; spot-check the surface.
- [ ] (Optional) build the image to confirm it succeeds.
#### Verify
```bash
test -f cardiac_mcp/Dockerfile && test -f LICENSE && echo OK
```
#### Exit Criteria
- [ ] Inspector validates the full surface; LICENSE present; Dockerfile builds (CPU-only).
#### Risk
Docker torch image size / GPU assumptions. Mitigation: CPU-only wheel index (the server defaults to CPU); document image size; this step is optional (publishing only).

### Phase 4 Exit Criteria
- [ ] `server.json` + marker + LICENSE + Dockerfile present and consistent; Inspector-validated.

### Phase 4 Cleanup
- [ ] `server.json` `name`/`version` consistent with `pyproject` + README marker.

**-> Commit point: `git commit` after Phase 4 (if pursued).**

---

## Final Cleanup (cross-phase)
- [ ] No `print()` to stdout anywhere in the server process; logs to stderr only.
- [ ] `cardiac_core` engines untouched (V5.3 read-only; this plan is server/packaging/docs only).
- [ ] No code duplication — server logic stays in `cardiac_mcp/core.py`; `cardiac_core` API unchanged.
- [ ] README, `.mcp.json`, `server.json` config all mutually consistent.
- [ ] All tests green: `conda run -n heart-conduction python -m pytest cardiac_mcp/tests/ cardiac_core/tests/ -q`.
- [ ] Update KNOWLEDGE "Goal-2 MCP server — standardization audit" with the SHIPPED status + which tiers landed.
- [ ] Archive this plan:
```bash
mkdir -p Research/Active/engine_consolidation/plans
cp Research/Active/engine_consolidation/PLAN.md "Research/Active/engine_consolidation/plans/$(date +%Y-%m-%d)_cardiac-mcp-standardization-tiers-1-4.md"
```
- [ ] Revert the bottom tmux pane from PLAN.md back to WHITEBOARD.md (see skill command).

## Mutation Log

**MUTATED 2026-06-28**: Step 2.4 MODIFIED — audit **H1+H2+L1**: replaced the non-existent `cardiac_core/pyproject.toml` anchor + hatchling divergence + a 2nd overlapping editable install with **Option B** — extend the ROOT `./pyproject.toml` (widen `include` to `cardiac_mcp*`, add a `cardiac-mcp` console script + `mcp>=1.2.0` dep), single `pip install -e .`, assert BOTH `import cardiac_core` and `import cardiac_mcp`. Keeps setuptools + `requires-python>=3.11` (resolves L1). Architecture-Changes line updated (NEW pyproject → MOD `./pyproject.toml`); Phase-2 Cleanup reworded.
**MUTATED 2026-06-28**: Step 3.1 MODIFIED — audit **M2**: dropped `RLIMIT_AS` (virtual-AS cap aborts torch/CUDA init); keep `RLIMIT_CPU`+`RLIMIT_FSIZE`+`timeout`; real memory isolation deferred to the 3.3 container; Verify now ensures the `@slow` end-to-end is actually run (not deselected). Audit **H3**: foreign-script test now passes an ABSOLUTE `tmp_lab` dir so it clears the Phase-1 `is_relative_to(LAB)` guard before reaching the provenance check.
**MUTATED 2026-06-28**: Step 1.1 MODIFIED — audit **M4/L5**: added `test_server_metadata` (asserts `serverInfo.version=="0.1.0"` + tool annotations) so the pure-wiring step has its own automated Test Spec instead of deferring to a manual verify.
**MUTATED 2026-06-28**: Step 2.1 MODIFIED — audit **M1**: added `test_list_experiments_always_has_count` and reworded the exit criterion ("no renames/value-changes to EXISTING fields"; the `count` addition is an intentional, tested schema stabilization). Audit **L2**: corrected the `simulate` return anchor (~150 → ~141; others verified).
**MUTATED 2026-06-28**: Step 1.2 MODIFIED — audit **M3**: clarified the traversal test asserts the raise for BOTH absolute and `..` inputs and does not depend on `Lab/` existing.
**MUTATED 2026-06-28**: Step 3.2 MODIFIED — audit **L3**: noted port 8000 + `/mcp` are FastMCP DEFAULTS (a `FASTMCP_PORT`/settings override changes them); replaced the foreground `sleep` probe with a background-launch + targeted-PID kill (sandbox blocks foreground `sleep`).
**MUTATED 2026-06-28**: Steps 4.1 & 4.2 MODIFIED — audit **L5**: added the missing Read First / Pseudocode / Checklist sections (now full 9-section structure); fixed the Option-B publishing consequence (publishable dist = bundled `cardiac-core`, so `server.json` `identifier`/marker reference it, not a separate `cardiac-mcp` PyPI package).
**MUTATED 2026-06-28**: test counts MODIFIED — audit **L4**: Phase-1 total 12 → 13 (10 prior + 3 new); Step-1.2 verify count updated.

--- round-2 audit (0 critical / 0 high / 0 medium / 5 low; all 9 round-1 fixes verified CORRECT) ---
**MUTATED 2026-06-28**: Step 3.1 MODIFIED — audit **R2-L5**: replaced the soft "and/or" end-to-end note with a MANDATORY `test_run_experiment_under_limits` (commit→run a real experiment through the subprocess) — the in-process `test_simulate_end_to_end` never touches `run_experiment`, so it didn't cover the `preexec_fn` limits path. Verify/Checklist/Exit updated.
**MUTATED 2026-06-28**: Step 1.1 MODIFIED — audit **R2-L2**: `test_server_metadata` now looks tools up by name (`{t.name: t for t in list_tools()}`) — `list_tools()` returns a list, not a name-keyed dict.
**MUTATED 2026-06-28**: Steps 2.3 & 3.3 MODIFIED — audit **R2-L3**: added the missing `#### Pseudocode` (doc-skeleton) so the two doc-only steps are 9/9 like the rest.
**MUTATED 2026-06-28**: Step 2.3 MODIFIED — audit **R2-L4**: noted the `.mcp.json` snippet is provisional until Step 2.4 (author Install/run last to avoid a brief stale snippet).
**MUTATED 2026-06-28**: Step 2.4 MODIFIED — audit **R2-L1**: noted `cardiac_mcp*` also matches `cardiac_mcp.tests` (harmless for editable; add `exclude=["*.tests*"]` only if a Phase-4 wheel should omit test code).
