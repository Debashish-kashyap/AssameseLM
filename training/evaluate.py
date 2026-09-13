# training/evaluate.py

import torch
import sentencepiece as spm
import math
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import AssameseGPT, ModelConfig

# ── Load model ────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sp = spm.SentencePieceProcessor()
sp.load("data/tokenizer/assamese_bpe.model")

cfg   = ModelConfig()
model = AssameseGPT(cfg).to(device)

checkpoint = torch.load("training/checkpoints_84m_v3/best.pt", map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()
print("Model loaded.")

BOS = sp.piece_to_id("[BOS]")
EOS = sp.piece_to_id("[EOS]")
PAD = sp.piece_to_id("[PAD]")

# ── 1. Perplexity ─────────────────────────────────────────────────────────────

def compute_perplexity(filepath, n_lines=5000):
    print(f"\nComputing perplexity on last {n_lines} lines...")

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    test_lines = lines[-n_lines:]

    total_loss   = 0
    total_tokens = 0
    skipped      = 0

    with torch.no_grad():
        for line in test_lines:
            line = line.strip()
            if not line:
                continue

            ids = [BOS] + sp.encode(line) + [EOS]

            if len(ids) < 4 or len(ids) > cfg.max_seq_len:
                skipped += 1
                continue

            x = torch.tensor([ids[:-1]], dtype=torch.long).to(device)
            y = torch.tensor([ids[1:]],  dtype=torch.long).to(device)

            logits, loss, _ = model(x, y)

            n_tok         = y.shape[1]
            total_loss   += loss.item() * n_tok
            total_tokens += n_tok

    avg_loss   = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    print(f"Lines evaluated:  {n_lines - skipped:,}")
    print(f"Tokens evaluated: {total_tokens:,}")
    print(f"Average loss:     {avg_loss:.4f}")
    print(f"Perplexity:       {perplexity:.2f}")
    return perplexity

# ── 2. Sampling utils ─────────────────────────────────────────────────────────

def top_p_filter(logits, p=0.92):
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs     = torch.softmax(sorted_logits, dim=-1)
    cum_probs = torch.cumsum(probs, dim=-1)

    # shift right — keep token that pushes over threshold
    remove = (cum_probs - probs) > p
    sorted_logits[remove] = float('-inf')

    # always keep top token — prevents all-filtered case
    sorted_logits[:, 0] = sorted_logits[:, 0].clamp(min=-1e9)

    result = torch.zeros_like(logits).scatter(1, sorted_idx, sorted_logits)
    return result

def apply_rep_penalty(logits, generated, penalty=1.2):
    for tok in set(generated):
        if logits[0, tok] > 0:
            logits[0, tok] /= penalty
        else:
            logits[0, tok] *= penalty
    return logits

def generate_tokens(prompt, max_new=100, temperature=0.7,
                    top_p=0.92, rep_penalty=1.2, min_new=20):
    ids = [BOS] + sp.encode(prompt)
    x   = torch.tensor([ids], dtype=torch.long).to(device)
    generated = []

    with torch.no_grad():
        for _ in range(max_new):
            logits, _, _ = model(x[:, -cfg.max_seq_len:])
            logits = logits[:, -1, :]

            # rep penalty
            logits = apply_rep_penalty(logits, generated, rep_penalty)

            # temperature
            logits = logits / temperature

            # block EOS until min length (before top-p so the kept token isn't EOS)
            if len(generated) < min_new:
                logits[0, EOS] = float('-inf')

            # top-p
            logits = top_p_filter(logits, top_p)

            probs = torch.softmax(logits, dim=-1)
            if not torch.isfinite(probs).all() or probs.sum() <= 0:
                tok = logits.argmax(dim=-1, keepdim=True)
            else:
                tok = torch.multinomial(probs, num_samples=1)

            if tok.item() == EOS:
                break

            generated.append(tok.item())
            x = torch.cat([x, tok], dim=1)

    return generated

# ── 3. Metrics ────────────────────────────────────────────────────────────────

def repetition_rate(tokens, window=32):
    if len(tokens) < 2:
        return 0.0
    repeats = 0
    for i in range(1, len(tokens)):
        context = tokens[max(0, i - window):i]
        if tokens[i] in context:
            repeats += 1
    return repeats / (len(tokens) - 1)

def distinct_n(tokens, n):
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams)

# ── 4. Evaluate generation ────────────────────────────────────────────────────

def evaluate_generation(prompts, n_runs=5):
    print(f"\nEvaluating generation ({len(prompts)} prompts × {n_runs} runs)...")

    all_rep_rates = []
    all_distinct1 = []
    all_distinct2 = []
    all_lengths   = []

    for prompt in prompts:
        for _ in range(n_runs):
            tokens = generate_tokens(prompt, max_new=100,
                                     temperature=0.7, top_p=0.92,
                                     rep_penalty=1.2, min_new=20)
            if not tokens:
                continue
            all_rep_rates.append(repetition_rate(tokens))
            all_distinct1.append(distinct_n(tokens, 1))
            all_distinct2.append(distinct_n(tokens, 2))
            all_lengths.append(len(tokens))

    print(f"Avg length:        {sum(all_lengths)/len(all_lengths):.1f} tokens")
    print(f"Repetition rate:   {sum(all_rep_rates)/len(all_rep_rates):.3f}  (lower=better)")
    print(f"Distinct-1:        {sum(all_distinct1)/len(all_distinct1):.3f}  (higher=better)")
    print(f"Distinct-2:        {sum(all_distinct2)/len(all_distinct2):.3f}  (higher=better)")

# ── 5. Run ────────────────────────────────────────────────────────────────────

TEST_PROMPTS = [
    "অসম হৈছে",
    "মই ভাবিছো",
    "ভাৰতৰ",
    "বিজ্ঞান আৰু",
    "অসমীয়া ভাষাৰ",
    "গুৱাহাটী চহৰ",
    "ল'ৰাটোৱে",
    "আজি বতৰ",
]

compute_perplexity("data/processed/clean_as.txt", n_lines=5000)
evaluate_generation(TEST_PROMPTS, n_runs=5)

print("\n── Generation samples ──")
SAMPLE_PROMPTS = [
    "অসম হৈছে",
    "মই ভাবিছো",
    "অসমীয়া ভাষাৰ",
    "গুৱাহাটী চহৰ",
]

for prompt in SAMPLE_PROMPTS:
    tokens = generate_tokens(prompt, max_new=150,
                             temperature=0.7, top_p=0.92,
                             rep_penalty=1.2, min_new=20)
    text = sp.decode(tokens)
    print(f"\nPrompt:   {prompt}")
    print(f"Length:   {len(tokens)} tokens")
    print(f"Rep rate: {repetition_rate(tokens):.3f}")
    print(f"Output:   {text}")

print("\nEvaluation complete.")