from datasets import load_dataset
import torch
import torch.nn as nn

EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-4
MAX_SEQ = 64
OVERFIT_EPOCHS = 100

ds = load_dataset("Helsinki-NLP/opus_books", "en-pt")
train_raw = ds["train"].select(range(1000))

print("total:", len(train_raw))
print("exemplo:", train_raw[0]["translation"]["en"])
print("traducao:", train_raw[0]["translation"]["pt"])
