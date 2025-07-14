"""Utility functions to analyze when sentiment improves trading models."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataset.MultiModalDataset import MultiModalDataset
from model.CnnTaFusion import CnnTaFusion


def headline_coverage(sentiment_path: Path, last_n: int = 30) -> pd.Series:
    """Return daily headline count for the last ``n`` days."""
    df = pd.read_csv(sentiment_path)
    date_col = "date" if "date" in df.columns else "Date"
    df["date"] = pd.to_datetime(df[date_col], unit="ms")
    topic_cols = [c for c in df.columns if c.startswith("t")]
    df["count"] = df[topic_cols].sum(axis=1)
    daily = df.groupby(df["date"].dt.date)["count"].sum()
    return daily.tail(last_n)


def lagged_correlation(sentiment_path: Path, finance_path: Path, max_lag: int = 5):
    """Compute Pearson and Spearman correlations over lags 0..max_lag."""
    s = pd.read_csv(sentiment_path)
    f = pd.read_csv(finance_path)
    s_date = "date" if "date" in s.columns else "Date"
    f_date = "Date"
    s["date"] = pd.to_datetime(s[s_date], unit="ms")
    f["date"] = pd.to_datetime(f[f_date], unit="ms")
    topic_cols = [c for c in s.columns if c.startswith("t")]
    s["sentiment"] = s[topic_cols].sum(axis=1)
    f["return"] = f["Close"].pct_change()
    merged = pd.merge(s[["date", "sentiment"]], f[["date", "return"]], on="date", how="inner")
    res = []
    for k in range(max_lag + 1):
        shifted = merged["sentiment"].shift(k)
        pear = shifted.corr(merged["return"], method="pearson")
        spear = shifted.corr(merged["return"], method="spearman")
        res.append({"lag": k, "pearson": pear, "spearman": spear})
    return pd.DataFrame(res)


def fusion_saliency(model: CnnTaFusion, loader: DataLoader, device: torch.device):
    """Return average gradient magnitude for finance and sentiment branches."""
    model.eval()
    fin_grads, sent_grads = [], []
    for _, _, imgs, _ in loader:
        imgs = imgs.to(device).requires_grad_()
        out = model(imgs[:, 0:1], imgs[:, 1:2])
        cls = out.max(1)[1]
        loss = out.gather(1, cls.view(-1, 1)).sum()
        loss.backward()
        grad = imgs.grad.detach().abs().mean(dim=(2, 3))
        fin_grads.append(grad[:, 0])
        sent_grads.append(grad[:, 1])
        imgs.grad.zero_()
    fin = torch.cat(fin_grads).mean().item()
    sent = torch.cat(sent_grads).mean().item()
    return {"finance": fin, "sentiment": sent}


def auc_difference(finance_path: Path, sentiment_path: Path, indicators: list[str]):
    """Train quick models with and without sentiment and return AUC difference."""
    ds_sent = MultiModalDataset(finance_path, sentiment_path, indicators, include_sentiment=True)
    ds_fin = MultiModalDataset(finance_path, sentiment_path, indicators, include_sentiment=False)
    loader_sent = DataLoader(ds_sent, batch_size=64, shuffle=False)
    loader_fin = DataLoader(ds_fin, batch_size=64, shuffle=False)

    def evaluate(loader, with_sent):
        model = CnnTaFusion(num_topics=ds_sent.num_topics)
        model.eval()
        probs, labels = [], []
        for _, _, imgs, lbl in loader:
            with torch.no_grad():
                if with_sent:
                    out = model(imgs[:, 0:1], imgs[:, 1:2])
                else:
                    out = model.cnn(imgs[:, 0:1])
            probs.extend(torch.softmax(out, 1)[:, 1].cpu().tolist())
            labels.extend(torch.argmax(lbl, 1).cpu().tolist())
        return roc_auc_score(labels, probs)

    auc_sent = evaluate(loader_sent, True)
    auc_fin = evaluate(loader_fin, False)
    return auc_sent - auc_fin


def scatter_headline_vs_auc(pairs):
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    plt.scatter(x, y)
    plt.xlabel("Headline count")
    plt.ylabel("Delta AUC")
    plt.title("Headline count vs Δ-AUC")
    plt.show()


