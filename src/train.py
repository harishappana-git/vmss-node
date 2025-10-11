"""Entry point for distributed language-model training."""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from .data import CharDataset, TextConfig, build_tokenizer
from .model import ModelConfig, TransformerLanguageModel


@dataclass
class TrainingConfig:
    data_path: Path
    block_size: int = 64
    batch_size: int = 32
    micro_batch_size: int = 8
    max_steps: int = 200
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 20
    log_every: int = 10
    checkpoint_dir: Path = Path("checkpoints")
    world_size: int = 1


class Trainer:
    def __init__(self, config: TrainingConfig, rank: int, world_size: int) -> None:
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(
            "cuda", rank
        ) if torch.cuda.is_available() else torch.device("cpu")
        self.ddp = world_size > 1

        text_config = TextConfig(data_path=config.data_path, block_size=config.block_size)
        tokenizer = build_tokenizer(text_config)
        dataset = CharDataset(text_config, tokenizer)

        sampler = (
            DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
            if self.ddp
            else RandomSampler(dataset)
        )
        self.loader = DataLoader(
            dataset,
            batch_size=config.micro_batch_size,
            sampler=sampler,
            drop_last=True,
        )

        model_config = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=config.block_size,
        )
        model = TransformerLanguageModel(model_config).to(self.device)
        if self.ddp:
            model = DDP(model, device_ids=[rank] if self.device.type == "cuda" else None)
        self.model = model

        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optim,
            lr_lambda=self._lr_lambda,
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.micro_batch_size = config.micro_batch_size

    def _lr_lambda(self, step: int) -> float:
        if step < self.config.warmup_steps:
            return float(step + 1) / float(self.config.warmup_steps)
        return 1.0

    def run(self) -> None:
        self.model.train()
        global_step = 0
        if self.config.batch_size % self.config.micro_batch_size != 0:
            raise ValueError(
                "batch_size must be divisible by micro_batch_size for gradient accumulation"
            )
        grad_accum = self.config.batch_size // self.config.micro_batch_size
        for step in range(self.config.max_steps):
            epoch_loss = 0.0
            for micro_step, (x, y) in enumerate(self.loader):
                x = x.to(self.device)
                y = y.to(self.device)

                logits, _ = self.model(x)
                loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / grad_accum
                loss.backward()

                epoch_loss += loss.item()
                if (micro_step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optim.step()
                    self.scheduler.step()
                    self.optim.zero_grad()
                    global_step += 1

                    if global_step % self.config.log_every == 0 and self.rank == 0:
                        ppl = math.exp(epoch_loss * grad_accum / (micro_step + 1))
                        print(
                            f"step={global_step:04d} loss={epoch_loss:.4f} ppl={ppl:.2f}",
                            flush=True,
                        )

                if global_step >= self.config.max_steps:
                    break
            if global_step >= self.config.max_steps:
                break
        if self.rank == 0:
            self._save_checkpoint()

    def _save_checkpoint(self) -> None:
        self.config.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        model = self.model.module if isinstance(self.model, DDP) else self.model
        ckpt_path = self.config.checkpoint_dir / "model.pt"
        torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")


def setup_process(rank: int, world_size: int, config: TrainingConfig) -> None:
    if world_size > 1:
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
    trainer = Trainer(config, rank, world_size)
    trainer.run()
    if world_size > 1:
        dist.destroy_process_group()


def launch_training(config: TrainingConfig) -> None:
    world_size = config.world_size
    if world_size > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        mp.spawn(
            setup_process,
            args=(world_size, config),
            nprocs=world_size,
            join=True,
        )
    else:
        setup_process(rank=0, world_size=1, config=config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("src/data/sample.txt"),
        help="Path to the training corpus",
    )
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
    )
    parser.add_argument("--world-size", type=int, default=1)
    return parser.parse_args()


def main(args: Optional[argparse.Namespace] = None) -> None:
    parsed = parse_args() if args is None else args
    config = TrainingConfig(
        data_path=parsed.data_path,
        block_size=parsed.block_size,
        batch_size=parsed.batch_size,
        micro_batch_size=parsed.micro_batch_size,
        max_steps=parsed.max_steps,
        learning_rate=parsed.learning_rate,
        weight_decay=parsed.weight_decay,
        warmup_steps=parsed.warmup_steps,
        log_every=parsed.log_every,
        checkpoint_dir=parsed.checkpoint_dir,
        world_size=parsed.world_size,
    )
    launch_training(config)


if __name__ == "__main__":
    main()
