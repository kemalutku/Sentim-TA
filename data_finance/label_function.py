import numpy as np


def sliding_window_labeling(data, window_size=11, relabel_range=2):
    def relabel_labels(labels, relabel_range=2):
        labels_copy = labels.copy()
        n = len(labels)

        for i in range(n):
            if labels[i] in [1, 2]:
                target = labels[i]

                for j in range(1, relabel_range + 1):
                    if i + j < n and labels_copy[i + j] == 0:
                        labels_copy[i + j] = target

                # Spread label backward within range
                for j in range(1, relabel_range + 1):
                    if i - j >= 0 and labels_copy[i - j] == 0:
                        labels_copy[i - j] = target

        return labels_copy

    half_window = window_size // 2
    labels = []

    for i in range(len(data)):
        if i < half_window or i >= len(data) - half_window:
            labels.append(0)
            continue

        window = data['Close'][i - half_window: i + half_window + 1]
        middle_price = data['Close'].values[i]

        if middle_price == window.min():
            labels.append(1)  # Buy
        elif middle_price == window.max():
            labels.append(2)  # Sell
        else:
            labels.append(0)  # Hold

    if relabel_range > 0:
        labels = relabel_labels(labels, relabel_range)

    data['Label'] = labels
    return data


def label_threshold_based(data, horizon: int = 2, threshold: float = 0.02):
    prices = data["Close"]
    future_returns = (prices.shift(-horizon) - prices) / prices
    labels = np.zeros_like(future_returns)

    labels[future_returns >= threshold] = 1  # Buy
    labels[future_returns <= -threshold] = 2  # Sell
    labels[np.abs(future_returns) < threshold] = 0  # Hold

    data['Label'] = labels
    return data
