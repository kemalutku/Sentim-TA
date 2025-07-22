# Sentim-TA

This repository contains experiments with CNN and Transformer models for technical indicator based trading.

## Transformer input format

The `AttnTa` model processes a sequence of daily indicator vectors. Each day provides a vector of 15 technical indicators (optionally for multiple channels such as sentiment). These vectors are first mapped into the model hidden dimension using a small linear layer and then positional embeddings are added before passing the sequence to a Transformer encoder.

Refer to `model/AttnTA.py` for implementation details.
