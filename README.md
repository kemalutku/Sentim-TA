# Sentim-TA

This repository contains experiments with CNN and Transformer models for technical indicator based trading.

## Transformer input format

The `AttnTa` model processes a sequence of daily indicator vectors. Each day provides a vector of technical indicators (``config.indicators``). These vectors are projected to the model hidden dimension and combined with positional embeddings before passing the sequence to a Transformer encoder. The number of days per window is controlled by ``config.sequence_len``.

Refer to `model/AttnTA.py` for implementation details.
