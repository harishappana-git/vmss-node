# Distributed LLM Training Demo

This repository demonstrates how to train a lightweight transformer-based
language model with data parallelism using [PyTorch Distributed Data Parallel
(DDP)](https://pytorch.org/docs/stable/nn.parallel.html#distributeddataparallel).
The goal is to provide an approachable template that highlights the mechanics
of splitting a workload across multiple workers while keeping the model small
and easy to reason about.

## Project Layout

```
src/
  data.py          # Dataset and tokenizer glue code
  data/sample.txt  # Tiny sample corpus used for quick experiments
  model.py         # Transformer language model definition
  tokenizer.py     # Minimal character-level tokenizer
  train.py         # Training entry point with optional DDP support
```

## Installation

1. Create a virtual environment (recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
   ```

   The demo only depends on PyTorch. Add the CUDA build if you plan to run on
   GPUs.

## Running Single-Process Training

The quickest way to experiment is to run on a single process (and optionally a
single GPU):

```bash
python -m src.train --data-path src/data/sample.txt --max-steps 20
```

This command trains the model for 20 optimizer steps on the bundled sample
corpus. Training logs show the loss and the derived perplexity. A checkpoint is
saved to `checkpoints/model.pt` after the run finishes.

## Distributed Data Parallel Training

Data parallelism replicates the model across multiple workers and splits each
batch of data between them. During the backward pass, gradients from each worker
are synchronised before the optimizer step. PyTorch DDP handles gradient
synchronisation efficiently by overlapping communication and computation.

To launch training across `N` processes on a single machine, set `--world-size`
to `N`:

```bash
python -m src.train --world-size 2 --max-steps 50
```

Behind the scenes the script performs the following steps:

1. **Process group initialisation** – the `launch_training` helper spawns
   `world_size` processes and configures the PyTorch distributed backend (`gloo`
   for CPU, `nccl` for GPU).
2. **Distributed sampling** – each worker gets a different shard of the data via
   `DistributedSampler`, ensuring that together they cover the full dataset
   without overlap.
3. **Model replication** – every worker owns a copy of the model wrapped in
   `DistributedDataParallel`, which automatically synchronises gradients.
4. **Gradient accumulation** – the trainer supports gradient accumulation via
   the `--batch-size` and `--micro-batch-size` flags so that the effective batch
   size can exceed what fits in memory per worker.
5. **Checkpointing** – only rank 0 writes checkpoints to disk to avoid file
   contention.

When running on multiple machines you can override `MASTER_ADDR` and
`MASTER_PORT` in the environment before invoking the script so that the workers
know how to reach the rendezvous server.

## Customising Training

Key arguments exposed by `src/train.py`:

- `--block-size`: maximum sequence length (context window).
- `--batch-size`: global batch size across gradient accumulation steps.
- `--micro-batch-size`: per-iteration micro-batch size before accumulation.
- `--max-steps`: number of optimizer steps to run.
- `--learning-rate`, `--weight-decay`, `--warmup-steps`: optimizer schedule.
- `--checkpoint-dir`: output directory for checkpoints.
- `--world-size`: number of processes to spawn for data parallel training.

Feel free to replace `src/data/sample.txt` with your own corpus. If you change
the context window (`--block-size`), ensure your dataset is longer than that
length.

## How It Works

The training loop follows the classic recipe for language models: the dataset
provides `(input, target)` token pairs, the transformer predicts the next token,
and cross entropy loss guides the optimisation. The interesting part lies in the
DDP orchestration:

- During each forward pass, every worker processes a different slice of the
  batch. Their gradients are aggregated automatically after `loss.backward()`.
- Because the optimiser only steps when gradients from all workers are ready,
  the model stays synchronised throughout training.
- Gradient accumulation allows each worker to process several micro-batches
  before synchronisation, which keeps the example runnable on machines with
  limited memory.

This setup mirrors the data parallel approach used to scale modern LLMs to many
GPUs or nodes. The same primitives can be extended with more sophisticated
optimisation strategies, checkpoint sharding, or pipeline/tensor parallelism as
projects grow.
