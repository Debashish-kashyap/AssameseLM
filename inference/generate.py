import torch
import sentencepiece as spm
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import AssameseGPT, ModelConfig

# ── Load ──────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sp = spm.SentencePieceProcessor()
sp.load("data/tokenizer/assamese_bpe.model")

cfg   = ModelConfig()
model = AssameseGPT(cfg).to(device)

checkpoint = torch.load("training/checkpoints_84m_v3/best.pt", map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()
print("Model loaded.\n")

# ── Sampling utils ────────────────────────────────────────────────────────────

def top_k_filter(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_val = values[:, -1].unsqueeze(-1)
    return logits.masked_fill(logits < min_val, float('-inf'))

def top_p_filter(logits, p):
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cum_probs - torch.softmax(sorted_logits, dim=-1) > p
    sorted_logits[remove] = float('-inf')
    logits = torch.zeros_like(logits).scatter(1, sorted_idx, sorted_logits)
    return logits

def apply_repetition_penalty(logits, generated_ids, penalty):
    if penalty == 1.0:
        return logits
    for token_id in set(generated_ids):
        if logits[0, token_id] > 0:
            logits[0, token_id] /= penalty
        else:
            logits[0, token_id] *= penalty
    return logits

# ── Generate ──────────────────────────────────────────────────────────────────

def generate(
    prompt,
    max_new_tokens=100,
    temperature=0.7,
    top_k=0,
    top_p=0.9,
    rep_penalty=1.3,
):
    BOS = sp.piece_to_id("[BOS]")
    EOS = sp.piece_to_id("[EOS]")

    ids = [BOS] + sp.encode(prompt)
    x   = torch.tensor([ids], dtype=torch.long).to(device)
    generated = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            x_crop = x[:, -cfg.max_seq_len:]

            logits, _, _ = model(x_crop)        # was: logits, _ = model(x_crop) — 3-tuple fix
            logits = logits[:, -1, :]

            logits = apply_repetition_penalty(logits, generated, rep_penalty)
            logits = logits / temperature

            if top_p < 1.0:
                logits = top_p_filter(logits, top_p)
            elif top_k > 0:
                logits = top_k_filter(logits, top_k)

            probs    = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)

            if next_tok.item() == EOS:
                break

            generated.append(next_tok.item())
            x = torch.cat([x, next_tok], dim=1)

    return sp.decode(generated)

# ── Test different settings ───────────────────────────────────────────────────

prompts = ["অসম হৈছে", "মই ভাবিছো", "ভাৰতৰ"]

print("=" * 60)
print("SETTING 1: Focused (low temp, strong rep penalty)")
print("=" * 60)
for p in prompts:
    out = generate(p, temperature=0.6, top_k=40, top_p=0.9, rep_penalty=1.4)
    print(f"Prompt: {p}")
    print(f"Output: {out}\n")

print("=" * 60)
print("SETTING 2: Balanced (default)")
print("=" * 60)
for p in prompts:
    out = generate(p, temperature=0.8, top_k=50, top_p=0.9, rep_penalty=1.3)
    print(f"Prompt: {p}")
    print(f"Output: {out}\n")

print("=" * 60)
print("SETTING 3: Creative (high temp, weak rep penalty)")
print("=" * 60)
for p in prompts:
    out = generate(p, temperature=1.1, top_k=80, top_p=0.95, rep_penalty=1.1)
    print(f"Prompt: {p}")
    print(f"Output: {out}\n")