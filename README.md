# AssameseLM

A decoder-only Assamese language model built from scratch in PyTorch. The repository contains the model, tokenizer training, data preparation, training, evaluation, inference, and attention visualization code.

## Model

- Architecture: decoder-only Transformer language model
- Parameters: 74,771,840 (~74.8M)
- Vocabulary: 16,000 SentencePiece BPE tokens
- Embedding size: 640
- Layers: 15
- Attention: grouped-query attention, 10 query heads / 2 key-value heads
- Feed-forward network: SwiGLU, hidden dim 1728
- Position encoding: rotary position embeddings (RoPE), base 10000
- Context length: 512 tokens
- Normalization: RMSNorm
- Parameter sharing: tied token embedding and language-model head
- All linear layers use no bias (Llama convention)
- Default attention path uses `F.scaled_dot_product_attention` (Flash-Attention backed); a manual QK^T path is used only when attention weights are requested for visualization

An earlier architecture (LayerNorm, learned absolute position embeddings, standard multi-head attention, GELU feed-forward, 84.4M params) was trained through 3 epochs, which beats it at every comparable epoch with 12% fewer parameters.

## Repository Layout

```text
data/
  collect.py               Collect raw text data
  clean.py                 Clean collected Assamese text
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

Python 3.10 or newer is recommended. PyTorch 2.0 or newer is required — `F.scaled_dot_product_attention` (used in the default attention path) is not available in earlier versions.

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

Attention visualization renders Assamese script labels via Pillow with the RAQM layout engine. If glyphs render as boxes, install a Bengali/Assamese-script font (e.g. Noto Sans Bengali) and point `BENGALI_FONT_PATH` in `inference/visualize_attention.py` at it.

## Data and Model Artifacts

Datasets, tokenizer binaries, fonts, generated outputs, and checkpoints are excluded by `.gitignore` and are not included in this repository.

Before running the tokenizer, training, or inference scripts, provide these local files:

```text
data/processed/combined_clean.txt
data/processed/clean_as.txt
data/tokenizer/assamese_bpe.model
data/tokenizer/assamese_bpe.vocab
training/checkpoints_84m_v3/best.pt
```

The scripts use repository-relative paths, so run commands from the project root. Do not commit private, licensed, or sensitive source data.

**Checkpoint directory convention:** each model architecture gets its own checkpoint directory (e.g. `checkpoints_84m_v3/`, not shared with the earlier architecture's `checkpoints/`). The two architectures' tensor shapes are incompatible — loading one into the other raises a `state_dict` shape-mismatch error. Keep this convention for any future architecture change.

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

This creates `data/tokenizer/assamese_bpe.model` and `data/tokenizer/assamese_bpe.vocab`. The tokenizer is trained once and reused across architecture changes — retraining it would break compatibility with any existing checkpoint's token embeddings.

## Training

Configure the constants near the top of `training/train.py`, then run:

```bash
python training/train.py
```

Default configuration: CUDA when available, sequence length 512, batch size 16, 3 epochs, AdamW with cosine-annealed learning rate (3e-4 → 1e-5),gradient clipping at norm 1.0, mixed precision via torch.amp.Checkpoints save under `training/checkpoints_84m_v3/` — `latest.pt` every 2000 steps, `epoch{N}.pt` at the end of each epoch, and `best.pt` whenever validation loss improves. Training resumes automatically from `latest.pt` if present, including mid-epoch, and is safe across interrupted sessions. Checkpoints are intentionally ignored by Git.

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

`evaluate.py` reports perplexity, repetition rate, and distinct-n lexical diversity on a held-out slice, plus qualitative generation samples. `generate.py` runs several Assamese sample prompts with focused, balanced, and creative sampling settings. Adjust the prompts and sampling parameters in either script for other experiments.

To inspect attention weights visually:

```bash
python inference/visualize_attention.py
```

This uses the manual attention path (`return_weights=True`), not the default Flash-Attention path — the fast path never materializes a full attention matrix, so there is nothing to extract for visualization. Expect deep layers (roughly layer 7 onward) to show most attention mass on `[BOS]` for many heads; this is a well-documented attention-sink effect seen across transformer models generally, not specific to this one. `top_attended_tokens()` supports an `exclude_bos` option that renormalizes scores after removing `[BOS]`, surfacing whatever secondary structure a head has beneath the sink.

## Results

Current checkpoint (`checkpoints_84m_v3/best.pt`, epoch 3 of 3):

| Metric | Value |
|---|---|
| Validation loss | 3.7018 |
| Perplexity (held-out eval slice) | 48.89 |
| Repetition rate | 0.052 |
| Distinct-1 | 0.948 |
| Distinct-2 | 0.996 |

## Tests

Run the model and tokenizer smoke tests from the project root:

```bash
python model/test_model.py
python tokenizer/test_tokenizer.py
```

## License and Data

No license is currently declared for this repository. Check the terms of any data source, font, tokenizer artifact, or pretrained checkpoint before redistribution.
