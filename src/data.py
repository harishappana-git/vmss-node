"""Dataset helpers for language modeling."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset

from .tokenizer import CharTokenizer


@dataclass
class TextConfig:
    data_path: Path
    block_size: int = 64


class CharDataset(Dataset):
    """A tiny dataset that slices a text stream into token blocks."""

    def __init__(self, config: TextConfig, tokenizer: CharTokenizer) -> None:
        self.config = config
        self.tokenizer = tokenizer
        raw_text = Path(config.data_path).read_text(encoding="utf-8")
        self.tokens: List[int] = tokenizer.encode(raw_text)
        if len(self.tokens) < config.block_size + 1:
            raise ValueError(
                "Dataset is too small for the configured block size."
            )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.tokens) - self.config.block_size

    def __getitem__(self, idx: int):  # type: ignore[override]
        chunk = self.tokens[idx : idx + self.config.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def build_tokenizer(config: TextConfig) -> CharTokenizer:
    text = Path(config.data_path).read_text(encoding="utf-8")
    return CharTokenizer.build([text])
