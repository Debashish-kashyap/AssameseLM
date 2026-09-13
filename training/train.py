import torch
from torch.utils.data import DataLoader
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import AssameseGPT, ModelConfig
from training.dataset import AssameseDataset

# ── Config ──────────────────────────────────────────────────────────────────
TOKENIZER  = "data/tokenizer/assamese_bpe.model"
DATA_FILE  = "data/processed/clean_as.txt"
SAVE_DIR   = "training/checkpoints"
MAX_LINES  = 500_000   # start with 500k lines, full run later
SEQ_LEN    = 512
BATCH_SIZE = 16        # fits in 4GB VRAM at seq_len=512
LR         = 3e-4
EPOCHS     = 1
LOG_EVERY  = 100       # print loss every N steps
SAVE_EVERY = 1000      # save checkpoint every N steps

os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Data ─────────────────────────────────────────────────────────────────────
dataset = AssameseDataset(DATA_FILE, TOKENIZER, seq_len=SEQ_LEN, max_lines=MAX_LINES)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# ── Model ─────────────────────────────────────────────────────────────────────
cfg   = ModelConfig()
model = AssameseGPT(cfg).to(device)
print(f"Parameters: {model.count_params():,}")

# ── Optimizer ─────────────────────────────────────────────────────────────────
# AdamW: Adam + weight decay. Standard for LM training.
# weight_decay penalizes large weights → regularization → less overfitting
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)

# ── LR Scheduler ──────────────────────────────────────────────────────────────
# Cosine annealing: LR starts at LR, decays to 0 following cosine curve
# Why: sharp early updates, gentle fine-tuning later
total_steps = EPOCHS * len(loader)
scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

# ── Training Loop ─────────────────────────────────────────────────────────────
model.train()
step       = 0
total_loss = 0

for epoch in range(EPOCHS):
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # forward
        logits, loss, _ = model(x, y)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping: prevents exploding gradients
        # if grad norm > 1.0, scale all grads down proportionally
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        step += 1

        if step % LOG_EVERY == 0:
            avg_loss = total_loss / LOG_EVERY
            lr_now   = scheduler.get_last_lr()[0]
            print(f"Step {step:6d} | Loss: {avg_loss:.4f} | LR: {lr_now:.2e}")
            total_loss = 0

        if step % SAVE_EVERY == 0:
            path = f"{SAVE_DIR}/ckpt_step{step}.pt"
            torch.save({
                "step":       step,
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "scheduler":  scheduler.state_dict(),
            }, path)
            print(f"Saved: {path}")

# final save
torch.save({
    "step":      step,
    "model":     model.state_dict(),
    "optimizer": optimizer.state_dict(),
}, f"{SAVE_DIR}/final.pt")
print("Training complete. Model saved.")