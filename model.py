# Attention(Q, K, V) = softmax(QK^T / sqrt(d_k) + M) V

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(7)

D_MODEL = 128
D_FF = 512
N_HEADS = 4
N_LAYERS = 2
MAX_SEQ = 64


def create_causal_mask(seq_len):
    return torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch = q.size(0)
        q = self.wq(q).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.wk(k).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.wv(v).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores + mask
        weights = torch.softmax(scores, dim=-1)
        att = weights @ v

        att = att.transpose(1, 2).contiguous().view(batch, -1, self.n_heads * self.d_k)
        return self.wo(att)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w2(F.relu(self.w1(x)))
