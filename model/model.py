# model/model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Config ─────────────────────────────────────────────────────────────────

class ModelConfig:
    vocab_size     = 16000
    embed_dim      = 640
    n_heads        = 10      # Q heads
    n_kv_heads     = 2       # GQA — K/V heads shared across Q-head groups
    n_layers       = 15
    ffn_hidden_dim = 1728    # SwiGLU hidden dim
    max_seq_len    = 512
    dropout        = 0.1
    rope_base      = 10000.0

# ── RMSNorm ──────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight

# ── RoPE ─────────────────────────────────────────────────────────────────────

def precompute_rope(dim, max_seq_len, base=10000.0, device=None):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rope(x, cos, sin):
    T = x.shape[-2]
    cos, sin = cos[:T], sin[:T]
    x1, x2 = x[..., ::2], x[..., 1::2]
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)

# ── GQA helper ───────────────────────────────────────────────────────────────

def repeat_kv(x, n_rep):
    B, n_kv, T, hd = x.shape
    if n_rep == 1:
        return x
    return x[:, :, None, :, :].expand(B, n_kv, n_rep, T, hd).reshape(B, n_kv * n_rep, T, hd)

# ── Attention ────────────────────────────────────────────────────────────────

class GQAAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.embed_dim % cfg.n_heads == 0
        assert cfg.n_heads % cfg.n_kv_heads == 0
        self.n_heads    = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep      = cfg.n_heads // cfg.n_kv_heads
        self.head_dim   = cfg.embed_dim // cfg.n_heads
        self.dropout    = cfg.dropout

        self.wq = nn.Linear(cfg.embed_dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.embed_dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.embed_dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.embed_dim, bias=False)

    def forward(self, x, cos, sin, return_weights=False):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        weights = None
        if return_weights:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))
            weights = torch.softmax(scores, dim=-1)
            out = weights @ v
            weights = weights.detach()
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.dropout if self.training else 0.0
            )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out), weights

# ── FeedForward (SwiGLU) ─────────────────────────────────────────────────────

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up   = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))

# ── Decoder Block ────────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.embed_dim)
        self.attn  = GQAAttention(cfg)
        self.norm2 = RMSNorm(cfg.embed_dim)
        self.ff    = SwiGLU(cfg.embed_dim, cfg.ffn_hidden_dim, cfg.dropout)

    def forward(self, x, cos, sin, return_weights=False):
        attn_out, w = self.attn(self.norm1(x), cos, sin, return_weights)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x, w

# ── Full Model ───────────────────────────────────────────────────────────────

class AssameseGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.drop      = nn.Dropout(cfg.dropout)
        self.blocks    = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm      = RMSNorm(cfg.embed_dim)
        self.lm_head   = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

        head_dim = cfg.embed_dim // cfg.n_heads
        cos, sin = precompute_rope(head_dim, cfg.max_seq_len, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, return_weights=False):
        x = self.drop(self.token_emb(idx))

        all_weights = [] if return_weights else None
        for block in self.blocks:
            x, w = block(x, self.rope_cos, self.rope_sin, return_weights)
            if return_weights:
                all_weights.append(w)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
                ignore_index=0
            )
        return logits, loss, all_weights

    def count_params(self):
        return sum(p.numel() for p in self.parameters())