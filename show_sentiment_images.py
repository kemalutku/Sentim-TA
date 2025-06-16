from __future__ import annotations

# ───────────────────────── CONFIGURATION ────────────────────────────
DATA_DIR = r"C:\Users\KemalUtkuLekesiz\Documents\Kod\Okul\NLP\SentimTA\data\cleaned"
OUT_DIR  = r"C:\Users\KemalUtkuLekesiz\Documents\Kod\Okul\NLP\SentimTA\data\samples"
# ────────────────────────────────────────────────────────────────────

import json
import random
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd


def _load_processed(proc: Path) -> Tuple[pd.DataFrame, dict[int, str]]:
    """Return merged DF and {id: name}."""
    topics = pd.read_csv(proc / "headline_topics.csv")
    sents  = pd.read_csv(proc / "headline_sentiments.csv")
    df = topics.merge(sents, on="headline", how="left")
    names = {int(k): v for k, v in json.load(open(proc / "topic_names.json")).items()}
    return df, names


def _pivot(df: pd.DataFrame, names: dict[int, str]) -> pd.DataFrame:
    """Top‑15 topics × all dates (mean sentiment)."""
    df = df.copy()
    df["topic"] = df.topic_id.map(names)
    df["date"]  = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])

    top = df.groupby("topic").size().nlargest(15).index
    return (
        df[df.topic.isin(top)]
        .pivot_table(index="topic", columns="date", values="sentiment",
                     aggfunc="mean", fill_value=0.0)
        .reindex(index=top)
    )


def _save_window(pv: pd.DataFrame, out_png: Path) -> None:
    """Pick random 15‑day slice, save heat‑map."""
    if pv.shape[1] < 15:
        print(f"⚠ {out_png.stem}: <15 date columns — skipped")
        return

    start = random.randint(0, pv.shape[1] - 15)
    sub   = pv.iloc[:, start:start + 15]

    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    im = ax.imshow(sub.values, vmin=-1, vmax=1, cmap="coolwarm",
                   aspect="auto", origin="lower")

    ax.set_xticks(range(15))
    ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in sub.columns],
                       rotation=90, fontsize=6)

    ax.set_yticks(range(sub.shape[0]))
    ax.set_yticklabels(sub.index, fontsize=7)

    title = f"{out_png.stem}   {sub.columns[0]} → {sub.columns[-1]}"
    ax.set_title(title)

    fig.colorbar(im, ax=ax, shrink=0.75, label="Sentiment")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print("✓", out_png.name)

def main() -> None:
    data_dir = Path(DATA_DIR)
    out_root = Path(OUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    for csv_path in sorted(data_dir.glob("*.csv")):
        stem = csv_path.stem
        proc = csv_path.with_name(f"sentiment_{stem}")
        if not proc.exists():
            print("✗ missing", proc.name)
            continue

        raw = pd.read_csv(csv_path)
        date_col = "Date" if "Date" in raw.columns else "date"
        if date_col not in raw.columns:
            print("✗ no Date column in", csv_path.name)
            continue
        raw["date"] = pd.to_datetime(raw[date_col]).dt.date
        raw["headline"] = raw["Headline"].astype(str).str.strip().str.lower()

        proc_df, names = _load_processed(proc)
        merged = proc_df.merge(raw[["headline", "date"]], on="headline", how="left")
        pv = _pivot(merged, names)

        _save_window(pv, out_root / f"heatmap_{stem}.png")


if __name__ == "__main__":
    main()
