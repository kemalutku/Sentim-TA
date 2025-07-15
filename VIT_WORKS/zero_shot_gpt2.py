# zero_shot_gpt2.py - Zero-shot VIT-GPT2 prediction on finance images
from __future__ import annotations

import argparse
import io
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from transformers import pipeline

from trade import trading
import config

MODEL_ID = "nlpconnect/vit-gpt2-image-captioning"
SEQ_LEN = 15


def load_finance_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'Date' not in df.columns:
        raise ValueError('CSV must contain a Date column')
    return df.dropna()


def window_to_image(window: pd.DataFrame) -> Image.Image:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=80)
    ax.imshow(window.to_numpy(), cmap='gray', aspect='auto')
    ax.set_xticks(np.arange(len(window.columns)))
    ax.set_xticklabels(window.columns, rotation=90, fontsize=6)
    ax.set_yticks(np.arange(len(window)))
    ax.set_yticklabels(window.index, fontsize=6)
    ax.set_xlabel('Indicators')
    ax.set_ylabel('Row')
    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert('L')
    return img


def generate_images(df: pd.DataFrame, features: list[str]):
    images = []
    timestamps = []
    closes = []
    for i in range(SEQ_LEN, len(df)):
        window = df[features].iloc[i-SEQ_LEN:i]
        img = window_to_image(window)
        images.append(img)
        timestamps.append(df['Date'].iloc[i])
        closes.append(df['Close'].iloc[i])
    return timestamps, closes, images


def predict_labels(images: list[Image.Image]):
    pipe = pipeline('image-to-text', model=MODEL_ID)
    labels = []
    for img in images:
        caption = pipe(img)[0]['generated_text'].lower()
        if any(k in caption for k in ['increase', 'up', 'rise', 'bull']):
            labels.append(1)  # buy
        elif any(k in caption for k in ['decrease', 'down', 'fall', 'bear']):
            labels.append(2)  # sell
        else:
            labels.append(0)  # hold
    return labels


def run(csv_path: Path):
    df = load_finance_csv(csv_path)
    features = [c for c in config.indicators if c in df.columns]
    if not features:
        raise ValueError('No indicator columns found in CSV')
    ts, closes, imgs = generate_images(df, features)
    preds = predict_labels(imgs)
    result, trades = trading(ts, closes, preds)
    print(result)
    print(trades)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zero-shot VIT-GPT2 finance prediction')
    parser.add_argument('csv', type=Path, help='Path to finance CSV file')
    args = parser.parse_args()
    run(args.csv)
