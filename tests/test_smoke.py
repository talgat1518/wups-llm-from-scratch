import torch

from model import ModelConfig, WupsGPT
from tokenizer import ByteTokenizer


def test_tokenizer_roundtrip_ascii_and_utf8():
    tok = ByteTokenizer()
    text = "AI / ИИ / ЖИ"
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    assert tok.vocab_size == 256


def test_model_forward_and_generation():
    torch.manual_seed(1)
    cfg = ModelConfig(block_size=16, d_model=32, n_layers=1, n_heads=4)
    model = WupsGPT(cfg)
    x = torch.randint(0, 256, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, 256)
    assert loss is not None and torch.isfinite(loss)
    generated = model.generate(x[:1, :4], max_new_tokens=3, top_k=10)
    assert generated.shape == (1, 7)
