# AssameseLM

A decoder-only Assamese language model built from scratch in PyTorch. The repository contains the model, tokenizer training, data preparation, training, evaluation, inference, and attention visualization code.

## Model

- Architecture: decoder-only Transformer language model
- Vocabulary: 16,000 SentencePiece BPE tokens
- Embedding size: 640
- Layers: 15
- Attention: grouped-query attention with 10 query heads and 2 key/value heads
- Feed-forward network: SwiGLU
- Position encoding: rotary position embeddings (RoPE)
- Context length: 512 tokens
- Normalization: RMSNorm
- Parameter sharing: tied token embedding and language-model head

## Repository Layout

```text
data/
  clean.py                 Clean collected Assamese text
  collect.py               Collect raw text data
  dedup.py                 Deduplicate processed text
inference/
  generate.py              Generate Assamese text from prompts
  visualize_attention.py   Visualize attention weights
model/
  model.py                 Model architecture and configuration
  config_check.py          Model configuration checks
  test_model.py            Model smoke tests
tokenizer/
  train_tokenizer.py       Train the SentencePiece tokenizer
  test_tokenizer.py        Tokenizer smoke tests
training/
  dataset.py               Dataset implementation
  train.py                 Training loop
  evaluate.py              Perplexity and generation metrics
```

## Requirements

Python 3.10 or newer is recommended. Install the core dependencies in a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install torch sentencepiece matplotlib numpy
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install torch sentencepiece matplotlib numpy
```

Install a CUDA-enabled PyTorch build when training with an NVIDIA GPU. Choose the command recommended by the official PyTorch installation selector for your CUDA version.

## Data and Model Artifacts

Datasets, tokenizer binaries, fonts, generated outputs, and checkpoints are excluded by `.gitignore` and are not included in this repository.

Before running the tokenizer, training, or inference scripts, provide these local files:

```text
data/raw/sangraha_as.txt
data/processed/clean_as.txt
data/tokenizer/assamese_bpe.model
data/tokenizer/assamese_bpe.vocab
training/checkpoints_84m_v3/best.pt
```

The scripts use repository-relative paths, so run commands from the project root. Do not commit private, licensed, or sensitive source data.

## Data Preparation

The data utilities are intended to be run in this order:

```bash
python data/collect.py
python data/clean.py
python data/dedup.py
```

Review the scripts and input/output paths before processing a new corpus. The tokenizer training script expects `data/processed/clean_as.txt`:

```bash
python tokenizer/train_tokenizer.py
```

This creates `data/tokenizer/assamese_bpe.model` and `data/tokenizer/assamese_bpe.vocab`.

## Training

Configure the constants near the top of `training/train.py`, then run:

```bash
python training/train.py
```

The default configuration uses CUDA when available, a sequence length of 512, batch size 16, one epoch, and saves checkpoints under `training/checkpoints/`. Training writes model checkpoints that are intentionally ignored by Git.

## Evaluation and Inference

The evaluation and generation scripts currently load:

```text
training/checkpoints_84m_v3/best.pt
```

After placing a compatible checkpoint and tokenizer at the expected paths:

```bash
python training/evaluate.py
python inference/generate.py
```

`generate.py` runs several Assamese sample prompts with focused, balanced, and creative sampling settings. Adjust the prompts and sampling parameters in the script for other experiments.

To inspect attention weights visually:

```bash
python inference/visualize_attention.py
```

## Tests

Run the model and tokenizer smoke tests from the project root:

```bash
python model/test_model.py
python tokenizer/test_tokenizer.py
```

## License and Data

No license is currently declared for this repository. Check the terms of any data source, font, tokenizer artifact, or pretrained checkpoint before redistribution.
