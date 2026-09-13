import sentencepiece as spm
import os

os.makedirs("data/tokenizer", exist_ok=True)

spm.SentencePieceTrainer.train(
    input="data/processed/clean_as.txt",
    model_prefix="data/tokenizer/assamese_bpe",
    vocab_size=16000,
    model_type="bpe",
    character_coverage=0.9999,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    pad_piece="[PAD]",
    unk_piece="[UNK]",
    bos_piece="[BOS]",
    eos_piece="[EOS]",
    input_sentence_size=2000000,
    shuffle_input_sentence=True,
    num_threads=8,
)

print("Tokenizer trained!")
print("Files: data/tokenizer/assamese_bpe.model")
print("       data/tokenizer/assamese_bpe.vocab")