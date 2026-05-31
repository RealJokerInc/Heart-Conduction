# harness_mini_v1 — Implementation Plan

Target: one afternoon, six phases, each independently verifiable + committable.

> **Golden rule for this build**: at the end of every phase, the thing we
> just added must work on its own. If Phase N breaks, Phase N−1 should
> still be useful. No half-finished phases. No anticipating Phase N+1.

---

## Phase 1 — Tailscale on both machines

**Goal**: laptop and SSH box on the same tailnet; MagicDNS resolves by
hostname in both directions.

### Steps

| # | Action | Where |
|---|---|---|
| 1.1 | `curl -fsSL https://tailscale.com/install.sh \| sh` | SSH box |
| 1.2 | `sudo tailscale up` → follow the URL in any browser → sign in (Google/Microsoft/GitHub) | SSH box |
| 1.3 | Install Tailscale desktop client for Linux / macOS / Windows | Laptop |
| 1.4 | Sign in to the same tailnet | Laptop |
| 1.5 | In admin console → Settings → DNS → toggle **MagicDNS on** | Web (any browser, one-time) |
| 1.6 | Pick a short hostname for the SSH box (e.g. `heartbox`). Rename in admin console → Machines if needed. | Web |

### Verify

```bash
# on SSH box
tailscale status            # laptop shown, status "active"
tailscale ip -4             # record the 100.x.y.z address

# on laptop
tailscale status            # ssh box shown, status "active"
ping heartbox               # MagicDNS resolves; <1ms RTT typical on LAN
ssh user@heartbox           # SSH via tailnet hostname works
```

Expected: both commands succeed; `ping heartbox` resolves via MagicDNS
without any `/etc/hosts` edit.

### Rollback

`sudo tailscale logout` on either node removes it from the tailnet.
No side effects on existing SSH config.

### Commit

Nothing in the repo to commit yet. Phase 1 is all out-of-repo infra.

---

## Phase 2 — MLflow server on SSH box (manual start)

**Goal**: MLflow server running on the SSH box, reachable from the
laptop browser over tailnet. Ephemeral — killed with the process.

### Steps

| # | Action | Where |
|---|---|---|
| 2.1 | Create server dirs outside the repo: `mkdir -p ~/mlflow/db ~/mlflow/artifacts` | SSH box |
| 2.2 | Confirm mlflow in conda env: `conda activate heart-conduction && python -c 'import mlflow; print(mlflow.__version__)'` — expect 2.22.4 | SSH box |
| 2.3 | Start the server in the foreground (first time only): see command below | SSH box |
| 2.4 | Open `http://heartbox:5000` in laptop browser | Laptop |

```bash
# Step 2.3 — foreground test run
mlflow server \
  --host 0.0.0.0 --port 5000 \
  --backend-store-uri "sqlite:///$HOME/mlflow/db/mlflow.db" \
  --default-artifact-root "$HOME/mlflow/artifacts"
```

### Verify

- Laptop browser shows MLflow UI with "No experiments yet" banner.
- `curl -s http://heartbox:5000/api/2.0/mlflow/experiments/search -H 'Content-Type: application/json' -d '{}'` from the laptop returns JSON.
- `ss -tlnp | grep 5000` on the SSH box shows the Python process owning the port.
- **Negative check**: from a non-tailnet machine (e.g. phone on mobile
  data), `http://<ssh-box-public-ip>:5000` times out. Tailscale firewall
  behaving correctly.

### Rollback

Ctrl-C the mlflow process. `~/mlflow/db/mlflow.db` and `~/mlflow/artifacts/`
are empty so far; delete the directory if you want a clean slate.

### Commit

Nothing repo-tracked yet. Still infra.

---

## Phase 3 — systemd user service (persistent)

**Goal**: server survives logout and reboots. Starts automatically on boot.

### Steps

| # | Action | Where |
|---|---|---|
| 3.1 | Enable user-service lingering: `sudo loginctl enable-linger "$USER"` | SSH box |
| 3.2 | Write `~/.config/systemd/user/mlflow.service` (content below) | SSH box |
| 3.3 | `systemctl --user daemon-reload && systemctl --user enable --now mlflow` | SSH box |
| 3.4 | Check status: `systemctl --user status mlflow` — expect `active (running)` | SSH box |
| 3.5 | Reboot the SSH box (`sudo reboot`) | SSH box |
| 3.6 | After reboot: laptop browser should still reach `http://heartbox:5000` | Laptop |

Service file content:

```ini
[Unit]
Description=MLflow Tracking Server (personal)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
# Replace /home/USER/miniconda3 with the actual conda prefix — run
# `conda info --base` on SSH box to get it.
ExecStart=/home/USER/miniconda3/envs/heart-conduction/bin/mlflow server \
  --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:///%h/mlflow/db/mlflow.db \
  --default-artifact-root %h/mlflow/artifacts
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

### Verify

- `systemctl --user is-active mlflow` → `active`
- After reboot + a fresh SSH session, the UI is still reachable.
- `journalctl --user -u mlflow --since "10 min ago"` shows the startup log.

### Rollback

```bash
systemctl --user disable --now mlflow
rm ~/.config/systemd/user/mlflow.service
systemctl --user daemon-reload
```

### Commit

Check the service file into the repo so it's documented and
reproducible:

```
harness_mini_v1/
├── server/
│   └── mlflow.service      # template; USER is substituted during install
└── server/README.md        # how to install this service
```

---

## Phase 4 — toy run sanity test

**Goal**: confirm Python-side logging works end-to-end before touching V4.

### Steps

| # | Action | Where |
|---|---|---|
| 4.1 | Write `harness_mini_v1/sanity/toy_log.py` (content below) | Repo |
| 4.2 | On SSH box: `python harness_mini_v1/sanity/toy_log.py` | SSH box |
| 4.3 | Laptop browser: open UI, navigate to `_sanity/toy`, see the loss curve | Laptop |

```python
# harness_mini_v1/sanity/toy_log.py
"""Minimum viable MLflow client. Logs 20 fake loss values."""
import random
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")  # from SSH box: localhost is the server
mlflow.set_experiment("_sanity")

with mlflow.start_run(run_name="toy"):
    mlflow.log_params({"lr": 1e-3, "arch": "toy"})
    for step in range(20):
        mlflow.log_metric("loss", 1 / (step + 1) + random.random() * 0.1, step=step)
```

### Verify

- Run `_sanity/toy` appears in UI.
- Params tab shows `lr=0.001`, `arch=toy`.
- Metrics tab renders a 20-point descending-noise loss curve.
- Re-run while UI is open: second run's curve updates live (within the
  ~5 s MLflow UI refresh interval).

### Rollback

Delete the `_sanity` experiment via the UI. No other state touched.

### Commit

```
harness_mini_v1/
└── sanity/
    └── toy_log.py
```

---

## Phase 5 — V4 instrumentation

**Goal**: V4 training script logs per-epoch train/val loss and best
checkpoint to MLflow. Zero change to training numerics.

### Step 5.0 — locate V4 entry point (read-only)

Before editing anything: determine which script actually starts a V4
run today. Candidates in `Surrogate/surrogate/training/`:
- `trainer.py`, `train_node.py` (known to have a latent `node.nfe` bug),
  `node_rollout.py`, `phases.py`, `rollout.py`.
- Possibly a top-level `Surrogate/run_*.py` that was not archived.
- The `archive/runs_legacy/` folder contains old strategies — ignore.

Ask: "what command currently trains V4?" Record it in this PLAN before
step 5.1.

### Steps 5.1–5.4

| # | Action | Detail |
|---|---|---|
| 5.1 | At the top of the entry script, read tracking URI from env: `MLFLOW_TRACKING_URI` defaults to `http://localhost:5000` | Portability — laptop-side debugging still works |
| 5.2 | `mlflow.set_experiment("v4")` before the training loop | One experiment, many runs |
| 5.3 | Inside `with mlflow.start_run(...)`, call `mlflow.log_params(cfg_dict)` once with the V4 hyperparam dict | Params are write-once per run |
| 5.4 | Inside the epoch loop: `mlflow.log_metrics({"train_loss": tl, "val_loss": vl}, step=epoch)` | Wrapped in try/except so a dead server does not kill training |
| 5.5 | On best-val-loss improvement: `mlflow.log_artifact(best_ckpt_path)` | Small file (~32 KB); safe to log every improvement |

Instrumentation sketch:

```python
import os
import mlflow

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("v4")

def _safe_log_metrics(d, step):
    try:
        mlflow.log_metrics(d, step=step)
    except Exception as e:
        print(f"[mlflow] log_metrics failed: {e}", flush=True)

with mlflow.start_run(run_name=f"v4_{tag}"):
    mlflow.log_params(cfg_flat)   # flat dict of hyperparams
    for epoch in range(num_epochs):
        tl = train_one_epoch(...)
        vl = validate(...)
        _safe_log_metrics({"train_loss": tl, "val_loss": vl}, step=epoch)
        if vl < best_vl:
            best_vl = vl
            save_state_dict(best_ckpt_path)
            try:
                mlflow.log_artifact(best_ckpt_path)
            except Exception as e:
                print(f"[mlflow] log_artifact failed: {e}", flush=True)
```

### Verify

- Short run (2 epochs, reduced dataset): `v4/<run_name>` appears in UI.
- `train_loss` and `val_loss` curves render after epoch 1.
- Best-epoch `state_dict.pt` is downloadable from the run's Artifacts
  tab, and `torch.load(...)` on the laptop reproduces the model weights.
- **Numerics check**: run the same V4 training for 2 epochs with vs.
  without instrumentation. Final loss values identical to float64 machine
  epsilon. (MLflow logging is a side effect only.)

### Rollback

`git revert` the instrumentation commit. V4 trains as before, no
MLflow calls, no env-var lookups.

### Commit

Commit the instrumentation as a **separate** commit from any V4 code
changes so rollback is trivial. Message prefix: `v4: add mlflow
tracking (no numerics change)`.

---

## Phase 6 (optional) — Hardening

Only if the basic setup is stable and in daily use.

- **6.1** Weekly backup cron on laptop:
  `rsync -a heartbox:~/mlflow/ ~/backups/mlflow-$(date +%F)/`
- **6.2** Log rotation for the systemd service journal (defaults are
  fine for personal use).
- **6.3** Firewall audit: `sudo ufw status` on SSH box, confirm port
  5000 not exposed outside tailnet.
- **6.4** Tailscale ACL: restrict MLflow port to your own user's devices
  only. Relevant if you ever share the tailnet with others.

---

## Non-phases (explicitly deferred)

| Deferred | Reason |
|---|---|
| Hydra config tree | Lives in `harness_v1/`. Mini has no config complexity. |
| Optuna sweeps | No sweep needed until V4 overfitting is understood. |
| Callback protocol | Direct `mlflow.log_*` is enough at this scale. |
| Cloudflare Tunnel / public URL | Only when a collaborator needs access. |
| Postgres backend | SQLite scales to years of single-user logging. |
| Auth beyond Tailscale tailnet isolation | Tailnet is the auth boundary. |
| Artifact retention policies | V4 state_dicts are 32 KB; disk is not a constraint for years. |

---

## Commit sequence

```
phase1: (no repo commits — Tailscale is out-of-repo infra)
phase2: (still no repo commits — just a manual process test)
phase3: harness_mini_v1/server: systemd user service + README
phase4: harness_mini_v1/sanity: toy MLflow logger script
phase5: v4: add mlflow tracking (no numerics change)
```

Phases 1 and 2 produce notes for a human-readable `harness_mini_v1/README.md`
at the end — a "how to set up on a fresh machine" guide — committed alongside
Phase 3.

---

## Open items from the PRD that must be resolved during execution

- **Q1** (exact hostname): resolved at Phase 1 step 1.6.
- **Q3** (artifact store path): resolved at Phase 2 step 2.1 — proposal
  is `~/mlflow/artifacts`; confirm disk headroom before starting.
- **Q4** (where V4 currently logs hyperparams): resolved at Phase 5
  step 5.0.
- **Q2** (backup cadence): deferred to Phase 6.
