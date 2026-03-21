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


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        att = self.attn(x, x, x)
        x = self.norm1(x + att)
        ff = self.ff(x)
        x = self.norm2(x + ff)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, y, enc_out, mask=None):
        att = self.self_attn(y, y, y, mask)
        y = self.norm1(y + att)
        cross = self.cross_attn(y, enc_out, enc_out)
        y = self.norm2(y + cross)
        ff = self.ff(y)
        y = self.norm3(y + ff)
        return y


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq, d_model)
        self.enc_blocks = nn.ModuleList([EncoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.dec_blocks = nn.ModuleList([DecoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        enc_out = self.encode(src)
        mask = create_causal_mask(tgt.size(1)).to(tgt.device)
        x = self.emb(tgt) + self.pos_emb(torch.arange(tgt.size(1), device=tgt.device))
        for block in self.dec_blocks:
            x = block(x, enc_out, mask)
        return self.proj(x)

    def encode(self, src):
        x = self.emb(src) + self.pos_emb(torch.arange(src.size(1), device=src.device))
        for block in self.enc_blocks:
            x = block(x)
        return x

    def decode(self, tgt, enc_out, mask):
        x = self.emb(tgt) + self.pos_emb(torch.arange(tgt.size(1), device=tgt.device))
        for block in self.dec_blocks:
            x = block(x, enc_out, mask)
        return self.proj(x)
