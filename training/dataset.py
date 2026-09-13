import torch
from torch.utils.data import Dataset
import sentencepiece as spm
import random

class AssameseDataset(Dataset):
    def __init__(self, filepath, tokenizer_path, seq_len=512, max_lines=None):
        self.seq_len = seq_len
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_path)

        BOS = self.sp.piece_to_id("[BOS]")  # 2
        EOS = self.sp.piece_to_id("[EOS]")  # 3

        print("Tokenizing corpus...")
        all_ids = []

        with open(filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_lines and i >= max_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                ids = self.sp.encode(line)
                # wrap each sentence with BOS/EOS
                all_ids.extend([BOS] + ids + [EOS])

        self.data = torch.tensor(all_ids, dtype=torch.long)
        print(f"Total tokens: {len(self.data):,}")

        # number of complete sequences
        self.n_seq = (len(self.data) - 1) // seq_len
        print(f"Total sequences: {self.n_seq:,}")

    def __len__(self):
        return self.n_seq

    def __getitem__(self, idx):
        start = idx * self.seq_len
        # input: tokens[start : start+seq_len]
        # target: tokens[start+1 : start+seq_len+1]  (shifted by 1)
        x = self.data[start     : start + self.seq_len]
        y = self.data[start + 1 : start + self.seq_len + 1]
        return x, y