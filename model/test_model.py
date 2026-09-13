import torch
from model.model import AssameseGPT, ModelConfig

cfg   = ModelConfig()
model = AssameseGPT(cfg).cuda()

print(f"Parameters: {model.count_params():,}")

# dummy forward pass
batch  = torch.randint(0, cfg.vocab_size, (2, 128)).cuda()  # 2 sequences, 128 tokens
target = torch.randint(0, cfg.vocab_size, (2, 128)).cuda()

logits, loss = model(batch, target)

print(f"Input shape:  {batch.shape}")
print(f"Output shape: {logits.shape}")   # should be (2, 128, 16000)
print(f"Loss:         {loss.item():.4f}")  # should be ~9.67 (log 16000)
print("Forward pass OK")