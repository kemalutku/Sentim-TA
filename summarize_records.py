import json
from pathlib import Path
from typing import Dict, Any

try:
    import config
    DEFAULT_RECORD_DIR = Path(config.record_dir)
except Exception:
    DEFAULT_RECORD_DIR = Path("records")

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _load_events(run_dir: Path) -> EventAccumulator | None:
    ea = EventAccumulator(str(run_dir))
    try:
        ea.Reload()
    except Exception:
        return None
    return ea


def _best_step(ea: EventAccumulator, tag: str) -> int | None:
    if tag not in ea.Tags().get("scalars", []):
        return None
    scalars = ea.Scalars(tag)
    if not scalars:
        return None
    best = max(scalars, key=lambda x: x.value)
    return best.step


def _value_at_step(ea: EventAccumulator, tag: str, step: int) -> float | None:
    scalars = ea.Scalars(tag)
    for s in scalars:
        if s.step == step:
            return s.value
    return None


def _trading_metrics(ea: EventAccumulator, step: int) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    for tag in ea.Tags().get("scalars", []):
        if not tag.startswith("trading/"):
            continue
        val = _value_at_step(ea, tag, step)
        if val is not None:
            metrics[tag.split("/", 1)[1]] = val
    return metrics


def summarize(record_dir: Path, ticker: str, metric_tag: str = "eval/F1"):
    best_by_exp: Dict[str, Dict[str, Any]] = {}
    if not record_dir.exists():
        return best_by_exp

    for run_dir in record_dir.iterdir():
        if not run_dir.is_dir():
            continue
        name = run_dir.name
        if ticker not in name:
            continue
        experiment = name.rsplit("_", 1)[-1]

        ea = _load_events(run_dir)
        if ea is None:
            continue
        step = _best_step(ea, metric_tag)
        if step is None:
            continue
        metric_val = _value_at_step(ea, metric_tag, step)
        if metric_val is None:
            continue
        trading = _trading_metrics(ea, step)

        result = {
            "run": name,
            "step": step,
            metric_tag: metric_val,
            **trading,
        }
        prev = best_by_exp.get(experiment)
        if prev is None or result[metric_tag] > prev[metric_tag]:
            best_by_exp[experiment] = result

    return best_by_exp


if __name__ == "__main__":
    # ---- user adjustable parameters ----
    record_dir = DEFAULT_RECORD_DIR
    ticker = "AAPL"  # filter run names containing this ticker
    metric_tag = "eval/F1"  # metric used to select the best step
    output_file = Path("summary.json")

    summary = summarize(record_dir, ticker, metric_tag)
    for exp, data in summary.items():
        print(f"{exp}: {data[metric_tag]:.4f} (step {data['step']}) -> {data['run']}")
        for k, v in data.items():
            if k in {"run", "step", metric_tag}:
                continue
            print(f"  {k}: {v:.4f}")

    with output_file.open("w") as f:
        json.dump(summary, f, indent=2)

