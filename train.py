from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import torch.nn as nn

EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-4
MAX_SEQ = 64
OVERFIT_EPOCHS = 100

ds = load_dataset("Helsinki-NLP/opus_books", "en-pt")
train_raw = ds["train"].select(range(1000))

tok = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased", do_lower_case=False)
pad_id = tok.pad_token_id
start_id = tok.cls_token_id
eos_id = tok.sep_token_id

def tokeniza_par(ex):
    src = tok.encode(ex["en"], add_special_tokens=False)[:MAX_SEQ]
    tgt = [start_id] + tok.encode(ex["pt"], add_special_tokens=False)[:MAX_SEQ - 2] + [eos_id]
    return src, tgt

print("total:", len(train_raw))
print("exemplo:", train_raw[0]["translation"]["en"])
print("traducao:", train_raw[0]["translation"]["pt"])

s, t = tokeniza_par(train_raw[0]["translation"])
print("tokens src:", s[:10])
print("tokens tgt:", t[:10])
