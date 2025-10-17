# CSCI 662 - Homework 2: GPT Implementation with Custom Attention

A PyTorch implementation of a GPT-like language model with custom multi-head attention mechanisms and text classification capabilities.

## Overview

This project implements a minimal GPT model (`gpt-nano` and `gpt-mini`) with:
- Custom multi-head attention with various similarity methods (dot product, cosine, correlation)
- Different attention initialization strategies (orthogonal, identity bias)
- Weight sharing options for Q, K, V projections
- Text classification capabilities
- BPE tokenization

## Files

- `train.py` - Main training script for both pretraining and fine-tuning for classification
- `evaluate.py` - Model evaluation for perplexity and classification accuracy
- `model.py` - GPT model implementation (based on MinGPT of Karpathy)
- `multi_head_attention.py` - Custom attention mechanism implementations
- `part1.py` - Attention mechanism exercises
- `bpe.py` - Byte Pair Encoding tokenizer
- `trainer.py` - Training loop and utilities

## Usage

### Training
```bash
# Pretraining (language modeling)
python train.py -t pretrain -i datasets/1b_benchmark.train.tokens -v datasets/1b_benchmark.dev.tokens

# Fine-tuning for classification
python train.py -t finetune -i datasets/4dim/train.txt -v datasets/4dim/val.txt
```

### Evaluation
```bash
# Evaluate perplexity
python evaluate.py -t lm -m best.model -i test_data.txt

# Evaluate classification accuracy
python evaluate.py -t classification -m best.model -i test_data.txt -l labels.txt
```

## Attention Mechanisms

The implementation supports various attention configurations:
- **Similarity methods**: dot product, cosine, average, correlation
- **Initialization**: standard linear, orthogonal, identity bias initialization of attention
- **Weight sharing**: QK, QV, KV, QKV projections
- **Nonlinearities**: tanh, relu, sigmoid

## Datasets

The project includes several datasets in the `datasets/` directory:
- `1b_benchmark/` - Language modeling data
- `4dim/`, `news/`, `products/`, `questions/` - Classification datasets

## Requirements

- PyTorch
- NumPy
- tqdm
- wandb (for logging)
- regex
