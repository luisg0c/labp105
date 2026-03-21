from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import torch.nn as nn

EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-4
MAX_SEQ = 64
OVERFIT_EPOCHS = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def pad_batch(seqs):
    ml = max(len(s) for s in seqs)
    return [s + [pad_id] * (ml - len(s)) for s in seqs]

src_all, tgt_all = [], []
for ex in train_raw:
    s, t = tokeniza_par(ex["translation"])
    src_all.append(s)
    tgt_all.append(t)

src_padded = torch.tensor(pad_batch(src_all)).to(device)
tgt_padded = torch.tensor(pad_batch(tgt_all)).to(device)

print(f"dados: {len(train_raw)} pares, src {src_padded.shape}, tgt {tgt_padded.shape}")
print("device:", device)

from model import Transformer, D_MODEL, N_HEADS, N_LAYERS, D_FF, MAX_SEQ as MODEL_MAX_SEQ

vocab_size = tok.vocab_size
model = Transformer(vocab_size, D_MODEL, N_HEADS, N_LAYERS, D_FF, MODEL_MAX_SEQ).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

if __name__ == "__main__":
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nvocab: {vocab_size}  params: {n_params}")
    print(f"batches por epoca: {len(src_padded) // BATCH_SIZE}")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        n_batches = 0
        for i in range(0, len(src_padded), BATCH_SIZE):
            batch_src = src_padded[i:i+BATCH_SIZE]
            batch_tgt = tgt_padded[i:i+BATCH_SIZE]

            tgt_in = batch_tgt[:, :-1]
            tgt_out = batch_tgt[:, 1:]

            optimizer.zero_grad()
            logits = model(batch_src, tgt_in)
            loss = criterion(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg = total_loss / max(n_batches, 1)
        print(f"  epoca {epoch+1:>2}/{EPOCHS}  loss: {avg:.4f}")
