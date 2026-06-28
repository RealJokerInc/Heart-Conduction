# REMOTE_DEPLOY.md — exposing cardiac-core MCP beyond localhost

> **Spec revision: MCP 2025-11-25.** Stdio (local) needs no auth (credentials come from the
> environment). The moment this server is reachable over HTTP from another machine, the MUST-level
> requirements below apply. **Do NOT expose `run_experiment` (it executes generated Python) beyond
> localhost without the sandbox in §4.** Until every MUST here is met, keep `CARDIAC_MCP_TRANSPORT=http`
> bound to `127.0.0.1` only (the current default — FastMCP binds loopback and auto-enables
> DNS-rebinding/Origin protection).

## 1. Authorization (HTTP transport) — MUST
- Implement **OAuth 2.1** for the authorization server (confidential + public clients).
- Clients **MUST** use **PKCE** with method `S256`; verify AS PKCE support via metadata.
- Implement **RFC 9728** Protected Resource Metadata; the AS provides **RFC 8414** or OIDC discovery.
- **RFC 8707 Resource Indicators**: the `resource` parameter (this server's canonical URI) on both the
  authorization and token requests; **validate the token audience is THIS server** (reject otherwise).
- All AS endpoints over **HTTPS**; redirect URIs registered and validated by **exact** string match.
- **No token passthrough** — never accept or forward a token not issued *to* this server.
- Per-request `Authorization: Bearer`; tokens never in the query string.

## 2. Transport & session — MUST / SHOULD
- Validate the `Origin` header → **403** on mismatch (DNS-rebinding defense). [MUST]
- Bind to `127.0.0.1` for local; only expose a public interface behind the auth stack above. [SHOULD]
- Session IDs: cryptographically secure, non-deterministic; **MUST NOT** authenticate via the session;
  verify every request; SHOULD bind `<user_id>:<session_id>`.

## 3. Network — MUST / SHOULD
- **SSRF** defenses on any outbound (OAuth discovery) fetch: HTTPS only; block private/loopback/
  link-local ranges (incl. `169.254.169.254`); validate redirect targets.

## 4. Code-execution sandbox (`run_experiment`) — REQUIRED before remote
The local hardening (provenance-marker check + `RLIMIT_CPU`/`RLIMIT_FSIZE` + wall `timeout`, no
`RLIMIT_AS`) is **not** sufficient for untrusted/remote callers. Before exposing the tool:
- Run each invocation in a **container/sandbox**: filesystem scoped to `Lab/` only, **no network**,
  **non-root**, per-call **ephemeral**, with real RSS/memory + PID limits (cgroups) — not just
  `RLIMIT_*` (virtual-AS caps break torch; see `core.py::run_experiment`).
- Keep the host **consent** principle: the host MUST get explicit user approval before invoking a tool;
  treat tool descriptions/annotations as untrusted.

## 5. Deployment gate
- [ ] §1 Authorization complete
- [ ] §2 Origin/session complete
- [ ] §3 SSRF complete
- [ ] §4 `run_experiment` sandbox complete

**Do NOT bind the HTTP server to any non-`127.0.0.1` interface until every box above is checked.**
