# harness_mini_v1 — MLflow Tracker PRD

_Owner: lc836 · Date: 2026-04-22 · Status: Draft_

## Problem

Training runs (currently the V4 ionic surrogate) execute on the SSH box.
Today, checking progress means `ssh` in, `tail -f` a log file, and eyeballing
text. No loss curves, no run comparison, no phone check from the couch. As
soon as multiple V4 variants are sweeping in parallel, this breaks down.

## Goal

Open a laptop browser (from anywhere) and see live training metrics of
any V4 run in an MLflow UI. No per-session SSH tunnel. Fully self-hosted.

## Success criteria

- [ ] From laptop browser on the tailnet, `http://<ssh-box-tailscale>:5000`
  loads the MLflow UI.
- [ ] A short V4 run appears in the UI with per-epoch `train_loss` and
  `val_loss` curves updating in real time.
- [ ] The MLflow server survives SSH session termination and SSH box reboot.
- [ ] The UI is reachable only from tailnet members (not from the public
  internet).
- [ ] Instrumentation adds ≤ 10 lines to the V4 training script and does
  not change training behavior (same loss trajectory as an un-instrumented
  run, within numerical noise).

## In scope

- Tailscale install + auth on SSH box and laptop.
- MLflow tracking server on SSH box (SQLite backend, local filesystem
  artifact store).
- `systemd --user` service so the server survives logout and reboots.
- Minimal instrumentation of the V4 training script (`log_params` +
  `log_metrics` per epoch + optional `log_artifact` for best checkpoint).
- A toy run script to sanity-check the pipeline before touching V4.
- README explaining how to start/stop the server and how to point a new
  script at it.

## Out of scope (deferred)

- Hydra config composition (lives in `harness_v1/` if needed).
- Optuna sweeps.
- Callback protocol / pluggable hooks.
- SHAP or any post-hoc analysis tooling.
- Public URL (Cloudflare Tunnel) — add only when a collaborator needs
  access.
- Postgres backend. SQLite is fine for single-writer personal use.
- Automated artifact pruning / retention policies.
- Multi-machine training coordination.

## Decisions already made

| Decision | Reason |
|---|---|
| Tailscale (not ngrok, not public VPS) | Self-hosted, private by default, free personal tier, stable hostnames via MagicDNS. |
| Self-hosted MLflow (not DagsHub / W&B) | Data stays on my machines; no external dependency. |
| SQLite backend store | One writer, personal scale. Postgres is overkill. |
| Artifact store on local filesystem, **outside** git repo | Avoid the `rm -rf outputs_parity/` class of mishap that bit the `cardiac_ml` Step 4.4 run. |
| Server lives on the SSH box | Training process writes to `localhost`; browser reaches it via tailnet. |
| V4 is the first real client (not V3, not a toy model) | Real use-case validates the pipeline; V4 is actively trained and actively overfitting — live curves will earn their keep immediately. |

## Constraints

- No code leaves the user's machines.
- Free (Tailscale personal plan, MLflow OSS).
- V4 is mid-debug (overfits T1's 25 trajectories; `input_ref` hotfix in
  place). The tracker **cannot** alter training behavior.
- Minimal dependency churn. MLflow 2.22.4 is already in the conda env
  from the `cardiac_ml` work; reuse it.
- Tracker calls must degrade gracefully — a dead server should not kill
  a running training job.

## Non-goals that have burned past harnesses (explicitly forbidden)

- ❌ `mlflow.pytorch.log_model(...)` — pickles the full nn.Module, fragile
  for custom classes. Use `mlflow.log_artifact(state_dict_path)`.
- ❌ `torch.set_default_dtype(...)` globally. Explicit `.to(device, dtype)`.
- ❌ A Trainer class. Mini has no Trainer yet; direct `mlflow.log_*` calls
  in the V4 script are fine at this scope.
- ❌ Hydra or Optuna plumbing "while we're in there."

## Open questions

| # | Question | Resolution path |
|---|---|---|
| Q1 | Exact hostname to use in docs: `<ssh-box>.tail-xyz.ts.net` or a short alias? | Set once Tailscale is up (Phase 1). MagicDNS gives both. |
| Q2 | Backup cadence for `~/mlflow/db/mlflow.db`? | Deferred. Weekly `rsync` to laptop is a Phase 6 item. |
| Q3 | Artifact store path on SSH box? | Proposed: `~/mlflow/artifacts`. Confirm disk has headroom (V4 state_dicts ~32 KB each — non-issue even for 10k runs). |
| Q4 | Does V4 training currently log hyperparams anywhere readable? | Answered during Phase 5 step 5.0 (locate entry point). |

## Risks

| Risk | Severity | Mitigation |
|---|---|---|
| MLflow server down mid-run → V4 training crashes | HIGH | Wrap `mlflow.log_metric` calls in `try/except`; log failures to stderr and continue training. |
| SQLite contention if V4 logs per-step at high rate | LOW | Log per-epoch + optional per-step sampler (every N steps). V4 epoch-level is ~minutes — no contention. |
| Tailscale tailnet admin console misconfigured → UI briefly exposed publicly | LOW | Default Tailscale posture is deny-all from outside tailnet. Verify with `curl` from a non-tailnet machine before declaring done. |
| Accidental `rm -rf` of `~/mlflow/` while server is running | MEDIUM | Document the "never touch ~/mlflow while training is live" rule in README. Backups (Phase 6) cover worst case. |

## Future work

- Cloudflare Tunnel on top if a collaborator needs access.
- Promote the `harness_mini_v1` pattern into `harness_v1/` if the
  MLflow-server-over-Tailscale approach proves superior to file-backed
  `./mlruns/`.
- Revisit the pedagogical 7-phase rebuild (Logger → Trainer → callbacks →
  Hydra → Optuna) once basic visibility is in production use.
