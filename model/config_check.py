# model/config_check.py

class TestConfig:
    vocab_size  = 16000
    embed_dim   = 640
    n_heads     = 10
    n_layers    = 15
    ff_dim      = 2560
    max_seq_len = 512
    dropout     = 0.1

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import AssameseGPT

model = AssameseGPT(TestConfig())
actual = model.count_params()

# our theoretical estimate
theoretical = (TestConfig.vocab_size * TestConfig.embed_dim + 
              TestConfig.n_layers * 12 * TestConfig.embed_dim**2)

print(f"Theoretical estimate: {theoretical:,}")
print(f"Actual params:        {actual:,}")
print(f"Difference:           {actual - theoretical:,}")
print(f"Error %:              {100*(actual-theoretical)/theoretical:.2f}%")