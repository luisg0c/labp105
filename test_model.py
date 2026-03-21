import torch
from model import (create_causal_mask, MultiHeadAttention, EncoderBlock,
                   DecoderBlock, Transformer, D_MODEL, N_HEADS, D_FF, N_LAYERS, MAX_SEQ)

torch.manual_seed(3)


def test_mascara_formato():
    mask = create_causal_mask(5)
    assert mask[0][0] == 0, "diagonal devia ser 0"
    assert mask[0][1] == float('-inf'), "acima devia ser -inf"
    assert mask[3][3] == 0, "diagonal errada"
    print("ok: mascara formato correto")


def test_encoder_shape():
    enc = EncoderBlock(D_MODEL, N_HEADS, D_FF)
    x = torch.randn(2, 10, D_MODEL)
    out = enc(x)
    assert out.shape == (2, 10, D_MODEL), f"shape errado: {out.shape}"
    print("ok: encoder mantem shape")


def test_decoder_shape():
    dec = DecoderBlock(D_MODEL, N_HEADS, D_FF)
    y = torch.randn(2, 8, D_MODEL)
    enc_out = torch.randn(2, 10, D_MODEL)
    mask = create_causal_mask(8)
    out = dec(y, enc_out, mask)
    assert out.shape == (2, 8, D_MODEL), f"shape errado: {out.shape}"
    print("ok: decoder mantem shape")


def test_transformer_forward():
    vocab = 1000
    model = Transformer(vocab, D_MODEL, N_HEADS, N_LAYERS, D_FF, MAX_SEQ)
    src = torch.randint(0, vocab, (2, 10))
    tgt = torch.randint(0, vocab, (2, 8))
    out = model(src, tgt)
    assert out.shape == (2, 8, vocab), f"shape errado: {out.shape}"
    print("ok: transformer forward shape correto")


if __name__ == "__main__":
    test_mascara_formato()
    test_encoder_shape()
    test_decoder_shape()
    test_transformer_forward()
    print("\ntodos os testes passaram")
