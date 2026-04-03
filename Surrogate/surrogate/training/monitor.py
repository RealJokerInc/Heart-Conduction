"""Training monitor: JSONL logging, control file, divergence detection.

Writes per-batch JSONL logs, per-epoch phase summaries, and polls a control
file for pause/resume/stop commands. Detects NaN, loss spikes, and plateaus.
"""

import json
import math
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class TrainingStoppedError(Exception):
    pass


class TrainingMonitor:
    def __init__(self, run_dir: str, control_path: Optional[str] = None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / 'training_log.jsonl'
        self.summary_path = self.run_dir / 'phase_summary.json'
        self.control_path = Path(control_path) if control_path else self.run_dir / 'training_control.json'

        # Initialize control file if it doesn't exist
        if not self.control_path.exists():
            self.update_control(status='running', current_phase='', current_epoch=0,
                                best_val_loss=float('inf'), message='')

        # EMA for divergence detection
        self._loss_ema = None
        self._grad_ema = None
        self._ema_alpha = 0.01
        self._epochs_no_improve = 0

    def log_batch(self, phase: str, epoch: int, batch: int, step: int,
                  loss: float, lr: float, grad_norm: float,
                  rollout: int = 1, sched_p: float = 0.0, wall_s: float = 0.0) -> None:
        entry = {
            'phase': phase, 'epoch': epoch, 'batch': batch, 'step': step,
            'loss': loss, 'lr': lr, 'grad_norm': grad_norm,
            'rollout': rollout, 'sched_p': sched_p, 'wall_s': wall_s,
            'timestamp': datetime.now().isoformat(),
        }
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

        # Update EMA
        if self._loss_ema is None:
            self._loss_ema = loss
            self._grad_ema = grad_norm
        else:
            self._loss_ema = (1 - self._ema_alpha) * self._loss_ema + self._ema_alpha * loss
            self._grad_ema = (1 - self._ema_alpha) * self._grad_ema + self._ema_alpha * grad_norm

    def log_epoch(self, phase: str, epoch: int, train_loss: float, val_metrics: dict) -> None:
        summary = {}
        if self.summary_path.exists():
            summary = json.loads(self.summary_path.read_text())

        summary[phase] = {
            'latest_epoch': epoch,
            'train_loss': train_loss,
            'val_metrics': val_metrics,
            'timestamp': datetime.now().isoformat(),
        }
        self.summary_path.write_text(json.dumps(summary, indent=2, default=str))

    def log_phase_transition(self, from_phase: str, to_phase: str, metrics: dict) -> None:
        entry = {
            'event': 'phase_transition', 'from': from_phase, 'to': to_phase,
            'metrics': metrics, 'timestamp': datetime.now().isoformat(),
        }
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(entry, default=str) + '\n')

    def check_control(self) -> str:
        if not self.control_path.exists():
            return 'running'
        ctrl = json.loads(self.control_path.read_text())
        status = ctrl.get('status', 'running')
        if status == 'stop_requested':
            raise TrainingStoppedError("Stop requested via control file")
        if status == 'pause_requested':
            ctrl['status'] = 'paused'
            self.control_path.write_text(json.dumps(ctrl, indent=2))
            logger.info("Training paused. Set status to 'running' to resume.")
            while True:
                time.sleep(5)
                ctrl = json.loads(self.control_path.read_text())
                if ctrl['status'] == 'running':
                    logger.info("Resuming training.")
                    return 'running'
                if ctrl['status'] == 'stop_requested':
                    raise TrainingStoppedError("Stop requested while paused")
        # Check for intervention
        intervention = ctrl.get('intervention')
        if intervention:
            return 'intervention'
        return status

    def update_control(self, **kwargs) -> None:
        ctrl = {}
        if self.control_path.exists():
            ctrl = json.loads(self.control_path.read_text())
        ctrl.update(kwargs)
        self.control_path.write_text(json.dumps(ctrl, indent=2))

    def get_intervention(self) -> Optional[dict]:
        if not self.control_path.exists():
            return None
        ctrl = json.loads(self.control_path.read_text())
        intervention = ctrl.get('intervention')
        if intervention:
            ctrl['intervention'] = None
            self.control_path.write_text(json.dumps(ctrl, indent=2))
        return intervention

    def check_divergence(self, loss: float, grad_norm: float) -> Optional[str]:
        if math.isnan(loss) or math.isinf(loss):
            return 'nan'
        if self._loss_ema is not None and self._loss_ema > 0:
            if loss > 3 * self._loss_ema:
                return 'spike'
        if self._grad_ema is not None and self._grad_ema > 0:
            if grad_norm > 10 * self._grad_ema:
                return 'grad_explosion'
        return None

    def apply_intervention(self, intervention: dict, optimizer=None) -> str:
        action = intervention.get('action', '')
        if action == 'reduce_lr':
            factor = intervention.get('factor', 0.5)
            if optimizer:
                for pg in optimizer.param_groups:
                    pg['lr'] *= factor
            return f"Reduced LR by {factor}"
        elif action == 'pause':
            return "Paused"
        elif action == 'transition_phase':
            return "Phase transition requested"
        elif action == 'rollback':
            return "Rollback requested"
        return f"Unknown action: {action}"
