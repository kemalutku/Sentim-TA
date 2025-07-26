# Sentim-TA

This repository contains experiments with CNN and Transformer models for technical indicator based trading.

## Transformer input format

The `AttnTa` model processes a sequence of daily indicator vectors. Each day provides a vector of technical indicators (``config.indicators``). These vectors are projected to the model hidden dimension and combined with positional embeddings before passing the sequence to a Transformer encoder. The number of days per window is controlled by ``config.sequence_len``.

Refer to `model/AttnTA.py` for implementation details.

## AttnTA Lightning Training

1. Preprocess one or more directories of finance CSVs into a single parquet:
   ```bash
   python data_finance/build_parquet.py <dir1> [dir2 ...] features.parquet
   ```

2. Train the attention model using the sequence window dataset:
   ```bash
   python -m train_attn_lightning
   ```

Hyper-parameters such as `window_lengths`, `d_models` and `n_layers` are defined
in `configs/attn_sweep.yml` for grid search.
