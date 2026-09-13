import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("data/tokenizer/assamese_bpe.model")

test_sentences = [
    "মই অসমীয়া ভাষা শিকিছো",
    "ভাৰতৰ উত্তৰ-পূৱ অঞ্চলত অসম অৱস্থিত",
    "শিকিছো শিকিছিলো শিকাওক শিকাব",   # same root, different suffixes
    "Hello this is English",
]

for sent in test_sentences:
    tokens = sp.encode_as_pieces(sent)
    ids    = sp.encode_as_ids(sent)
    print(f"\nInput : {sent}")
    print(f"Tokens: {tokens}")
    print(f"IDs   : {ids}")
    print(f"Count : {len(tokens)} tokens")