"""Transformer language model used for the training demo."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    vocab_size: int
    n_embd: int = 128
    n_layer: int = 2
    n_head: int = 4
    block_size: int = 64
    dropout: float = 0.1


class TransformerLanguageModel(nn.Module):
    """A lightweight GPT-style model for educational purposes."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, config.block_size, config.n_embd)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.n_embd,
            nhead=config.n_head,
            dim_feedforward=config.n_embd * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.n_layer)
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t = idx.size()
        if t > self.config.block_size:
            raise ValueError("Sequence length exceeds model context window")
        tok_emb = self.token_embedding(idx)
        pos_emb = self.pos_embedding[:, :t, :]
        x = tok_emb + pos_emb
        x = self.transformer(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, x
