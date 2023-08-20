import sys
import torch
import torch.nn as nn
from src.transformer import *


def translate(src, transformer, vocab, tokenizer):
    src_tokens = tokenizer[SRC_LANGUAGE](src)
    tgt_tokens = ["<BOS>"]

    src_vectors = torch.tensor(([BOS_IDX] + vocab[SRC_LANGUAGE](src_tokens) + [EOS_IDX] + [0] * (max_seq_len - len(src_tokens)))[:max_seq_len], dtype=torch.long, device=device).unsqueeze(0)
    
    for i in range(max_seq_len):
        tgt_vectors = torch.tensor((vocab[TGT_LANGUAGE](tgt_tokens) + [0] * (max_seq_len - len(tgt_tokens)))[:max_seq_len], dtype=torch.long, device=device).unsqueeze(0)
        output = transformer(src_vectors, tgt_vectors)
        idx = torch.argmax(nn.functional.softmax(output, dim=2)[0][i]).item()
        tgt_tokens.append(vocab[TGT_LANGUAGE].lookup_token(idx))

        if idx == EOS_IDX:
            break

    return " ".join(tgt_tokens).replace("<BOS>", "").replace("<EOS>", "").replace("<PAD>", "").strip()

if __name__ == "__main__":
    if len(sys.argv) > 2:
        print('Usage: python translate.py "Hello, my name is John."')
        sys.exit(1)

    SRC_LANGUAGE = "en"
    TGT_LANGUAGE = "es"
    BOS_IDX = 2
    EOS_IDX = 3
    max_seq_len = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = {}
    vocab = {}

    tokenizer[SRC_LANGUAGE] = torch.load("./output/tokenizer-english")
    tokenizer[TGT_LANGUAGE] = torch.load("./output/tokenizer-spanish")
    vocab[SRC_LANGUAGE] = torch.load("./output/vocab-english")
    vocab[TGT_LANGUAGE] = torch.load("./output/vocab-spanish")
    transformer = torch.load("./output/transformer_model").to(device)

    s = sys.argv[1]
    print(translate(s, transformer, vocab, tokenizer))