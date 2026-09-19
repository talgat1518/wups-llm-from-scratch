"""A tiny byte-level tokenizer with no pretrained vocabulary.

Every UTF-8 byte maps directly to one token ID in [0, 255]. This keeps the
prototype fully self-contained and works for Russian, Kazakh, English and any
other UTF-8 text.
"""
from __future__ import annotations


class ByteTokenizer:
    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int]) -> str:
        data = bytes(int(i) % 256 for i in token_ids)
        return data.decode("utf-8", errors="replace")
