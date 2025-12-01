"""Utility components for tokenizing raw text for language modeling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass
class CharTokenizer:
    """A simple character-level tokenizer.

    This tokenizer is intentionally tiny so that it can run in CPU-only
    environments that do not have the dependencies required for modern
    subword tokenizers. The goal for this repository is to showcase
    distributed training mechanics rather than tokenizer sophistication.
    """

    stoi: Dict[str, int]
    itos: List[str]

    @classmethod
    def build(cls, texts: Iterable[str]) -> "CharTokenizer":
        """Construct a tokenizer from an iterable of raw text snippets."""
        vocab = sorted({ch for text in texts for ch in text})
        stoi = {ch: idx for idx, ch in enumerate(vocab)}
        itos = vocab
        return cls(stoi=stoi, itos=itos)

    def encode(self, text: str) -> List[int]:
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)
