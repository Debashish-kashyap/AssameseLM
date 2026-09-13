import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import sentencepiece as spm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageDraw, ImageFont

# Point this at a font that supports Bengali/Assamese script, or Assamese
# glyphs will render as boxes. Install with:
#   sudo apt install fonts-noto-bengali   (Linux)
# or download Noto Sans Bengali from Google Fonts (Windows/Mac) and point here.
BENGALI_FONT_PATH = "D:/assamese_lm/fonts/NotoSansBengali-Regular.ttf"
_bn_font = fm.FontProperties(fname=BENGALI_FONT_PATH) if os.path.exists(BENGALI_FONT_PATH) else None


def _render_label_image(text, font_path, font_size=32, color=(0, 0, 0, 255)):
    """
    Render a token label to a transparent PNG using PIL with the 'raqm'
    layout engine, which does proper complex-script shaping (conjuncts,
    vowel signs). matplotlib's own text renderer does NOT shape Indic
    scripts correctly and will show broken/placeholder glyphs instead.
    """
    try:
        font = ImageFont.truetype(font_path, font_size, layout_engine=ImageFont.Layout.RAQM)
    except (AttributeError, OSError):
        # older Pillow without raqm support, or missing font — fall back
        # to basic layout engine (conjuncts may still render incorrectly)
        font = ImageFont.truetype(font_path, font_size)

    # measure text first on a throwaway canvas
    tmp = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(tmp)
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0] + 4, bbox[3] - bbox[1] + 4

    img = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    draw.text((2 - bbox[0], 2 - bbox[1]), text, font=font, fill=color)
    return img

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

BOS = sp.piece_to_id("[BOS]")

# ── Get attention weights for a sentence ─────────────────────────────────────

def get_attention(sentence):
    ids    = [BOS] + sp.encode(sentence)
    tokens = ["[BOS]"] + sp.encode_as_pieces(sentence)
    x      = torch.tensor([ids], dtype=torch.long).to(device)

    with torch.no_grad():
        # return_weights=True required — default is False, all_weights
        # would otherwise come back None and break everything downstream
        logits, _, all_weights = model(x, return_weights=True)

    # all_weights: n_layers × (1, n_heads, T, T)
    # (with GQA, K/V are repeated up to n_heads before the manual
    # attention path runs, so weight shape is unchanged from full MHA —
    # no changes needed to the layer/head indexing below)
    return tokens, all_weights

def print_attention_matrix(tokens, weights, layer, head):
    """Print attention as ASCII heatmap."""
    W = weights[layer][0, head].cpu().float()  # (T, T)
    T = len(tokens)

    print(f"\nLayer {layer+1}, Head {head+1}")
    print(f"{'':>12}", end="")
    for tok in tokens:
        print(f"{tok[:6]:>8}", end="")
    print()

    for i, row_tok in enumerate(tokens):
        print(f"{row_tok[:12]:>12}", end="")
        for j in range(T):
            val = W[i, j].item()
            # ASCII intensity: . ░ ▒ ▓ █
            if val < 0.05:   char = "·"
            elif val < 0.15: char = "░"
            elif val < 0.30: char = "▒"
            elif val < 0.50: char = "▓"
            else:            char = "█"
            print(f"{char:>8}", end="")
        print()

def save_attention_heatmap(tokens, weights, layer, head, save_path, title=None):
    """Save a single layer/head attention matrix as a PNG heatmap."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # strip the SentencePiece '▁' prefix marker before rendering
    display_tokens = [t.replace("▁", "") for t in tokens]

    W = weights[layer][0, head].cpu().float().numpy()  # (T, T)
    n = len(display_tokens)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(W, cmap="viridis", vmin=0, vmax=1)

    # annotate each cell with its value
    for i in range(n):
        for j in range(n):
            val = W[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                     color=color, fontsize=8)

    if os.path.exists(BENGALI_FONT_PATH):
        # use properly-shaped PIL-rendered images for the tick labels,
        # since matplotlib can't shape Assamese conjuncts correctly
        ax.set_xticks([])
        ax.set_yticks([])
        for i, tok in enumerate(display_tokens):
            label_img = _render_label_image(tok, BENGALI_FONT_PATH, font_size=26)
            # y-axis labels (left side)
            imagebox = OffsetImage(label_img, zoom=0.5)
            ab = AnnotationBbox(imagebox, (0, i), xybox=(-38, 0),
                                 xycoords=("data", "data"), boxcoords="offset points",
                                 box_alignment=(1, 0.5), frameon=False)
            ax.add_artist(ab)
            # x-axis labels (bottom, horizontal — reliable across all lengths)
            imagebox_x = OffsetImage(label_img, zoom=0.5)
            ab_x = AnnotationBbox(imagebox_x, (i, n - 1), xybox=(0, -32),
                                   xycoords=("data", "data"), boxcoords="offset points",
                                   box_alignment=(0.5, 1), frameon=False)
            ax.add_artist(ab_x)
        fig.subplots_adjust(left=0.26, bottom=0.22)
    else:
        # no Bengali font available — fall back to matplotlib's native
        # text (will render conjuncts incorrectly, but at least runs)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(display_tokens, rotation=45, ha="right", fontsize=12)
        ax.set_yticklabels(display_tokens, fontsize=12)

    display_title = (title or f"Layer {layer+1}, Head {head+1}").replace("▁", "")
    title_kwargs = {"fontproperties": _bn_font} if _bn_font else {}
    ax.set_title(display_title, fontsize=14, **title_kwargs)
    fig.colorbar(im, ax=ax, label="Attention weight")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap → {save_path}")


def top_attended_tokens(tokens, weights, layer, head, query_idx, exclude_bos=False, top_n=5):
    """
    For a given token, show what it attends to most.

    exclude_bos: drop [BOS] from the ranking before renormalizing the
    remaining scores to sum to 1. Deep layers (roughly layer 7+) tend to
    dump most attention mass on [BOS] as a low-cost "attention sink" once
    the useful information has already been extracted by earlier layers —
    this is normal transformer behavior, not specific to this model, but
    it buries the secondary (actually informative) structure underneath
    it. Renormalizing after removing BOS surfaces that structure.
    """
    W = weights[layer][0, head].cpu().float()
    row = W[query_idx].clone()  # attention from query_idx to all others

    bos_idx = tokens.index("[BOS]") if "[BOS]" in tokens else None
    bos_score = row[bos_idx].item() if bos_idx is not None else None

    if exclude_bos and bos_idx is not None:
        row[bos_idx] = 0.0
        remaining = row.sum().item()
        if remaining > 1e-8:
            row = row / remaining  # renormalize so shown scores still sum to 1

    ranked = sorted(enumerate(row.tolist()), key=lambda x: x[1], reverse=True)
    label = f"(layer {layer+1}, head {head+1})"
    if exclude_bos and bos_idx is not None:
        label += f" [BOS excluded, was {bos_score:.3f} of original mass]"
    print(f"\nToken '{tokens[query_idx]}' {label} attends to:")
    for idx, score in ranked[:top_n]:
        if exclude_bos and idx == bos_idx:
            continue
        bar = "█" * int(score * 30)
        print(f"  {tokens[idx]:>15} → {score:.3f} {bar}")

# ── Run visualization ─────────────────────────────────────────────────────────

sentences = [
    "অসম এখন ধুনীয়া ৰাজ্য",
    "মই অসমীয়া ভাষা শিকিছো",
]

for sent in sentences:
    print(f"\n{'='*60}")
    print(f"Sentence: {sent}")
    tokens, all_weights = get_attention(sent)
    print(f"Tokens:   {tokens}")

    # show attention matrix for layer 1 head 1
    print_attention_matrix(tokens, all_weights, layer=0, head=0)

    # show what last token attends to, across all layers
    # early layers (0-2) still show real BOS competition, so leave them as-is;
    # layer 3+ tends to sink onto BOS, so exclude it there to see the real signal
    print(f"\n── What last token '{tokens[-1]}' attends to ──")
    for layer in range(cfg.n_layers):
        for head in range(cfg.n_heads):
            top_attended_tokens(tokens, all_weights, layer, head, query_idx=-1,
                                 exclude_bos=(layer >= 3))

    # show layer-by-layer summary for one token
    print(f"\n── Layer-by-layer attention pattern ──")
    for layer in range(cfg.n_layers):
        W   = all_weights[layer][0].mean(0).cpu()  # average over heads
        row = W[-1]  # last token
        top_idx = row.argmax().item()
        print(f"  Layer {layer+1}: '{tokens[-1]}' → most attends to '{tokens[top_idx]}' ({row[top_idx]:.3f})")

    # export the clearest specialization example (layer 4, head 3) as PNG
    sent_idx = sentences.index(sent)
    save_attention_heatmap(
        tokens, all_weights, layer=3, head=2,
        save_path=f"outputs/attention_sentence{sent_idx+1}_L4H3.png",
        title=f"Layer 4, Head 3 — {sent}"
    )