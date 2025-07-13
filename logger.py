from pathlib import Path
from typing import List, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from torch.utils.tensorboard import SummaryWriter


class TBLogger:
    """
    • Logs per-epoch Loss / Accuracy / Precision / Recall / F1
      for *train* and *eval* phases.
    • Logs averaged trading KPIs (Final Value, Sharpe, …) for eval.
    • Writes everything to TensorBoard.
    """

    def __init__(self, log_dir: str | Path, run_name, comment: str = "", ):
        log_dir = Path(log_dir) / run_name
        self.writer = SummaryWriter(log_dir=str(log_dir), comment=comment)

    # ───────────────────  classification metrics  ────────────────── #
    @staticmethod
    def _cls_metrics(y_true, y_pred):
        return {
            "Accuracy":  accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "Recall":    recall_score(y_true, y_pred, average="macro", zero_division=0),
            "F1":        f1_score(y_true, y_pred, average="macro", zero_division=0),
        }

    def log_epoch(
        self,
        phase: str,             # "train" or "eval"
        epoch: int,
        loss: float,
        y_true,
        y_pred,
    ) -> None:
        """Write scalar metrics for one phase of one epoch."""
        m = self._cls_metrics(y_true, y_pred)
        m["Loss"] = loss

        # Write each metric under `phase/Metric`
        for k, v in m.items():
            self.writer.add_scalar(f"{phase}/{k}", v, epoch)

    # ─────────────────────  trading-specific logging  ────────────── #
    def log_trading(
        self,
        epoch: int,
        results: List[Dict[str, float]],
    ) -> None:
        """Average a list of trading-result dicts and write scalars."""
        if not results:
            return

        avg = {k: float(np.mean([r[k] for r in results])) for k in results[0]}
        for k, v in avg.items():
            safe_key = (
                k.replace(" ", "_")
                .replace("/", "_")
                .replace("%", "pct")
                .replace("(", "_")
                .replace(")", "_")
            )
            tag = f"trading/{safe_key}"
            self.writer.add_scalar(tag, v, epoch)

    # ─────────────────────────── housekeeping ───────────────────── #
    def close(self) -> None:
        self.writer.close()
