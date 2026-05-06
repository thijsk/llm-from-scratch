# Train Your Own LLM From Scratch in .NET

A hands-on workshop where you write every piece of a GPT training pipeline yourself, understanding what each component does and why.

This is a .NET / TorchSharp port of Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) workshop. The goal is the same: build a working language model from scratch on your laptop in a single session, understanding each step instead of relying on frameworks.

This project strips nanoGPT down to the essentials and implements it in C# with TorchSharp. It trains a ~10M param GPT model on Shakespeare that runs on CPU, GPU, or Apple Silicon — completing the full pipeline in under an hour.

## What You'll Build

A working GPT model trained from scratch, capable of generating Shakespeare-like text. You'll write:

- **Tokenizer** — turning text into numbers the model can process
- **Model architecture** — the transformer: embeddings, attention, feed-forward layers
- **Training loop** — forward pass, loss, backprop, optimizer, learning rate scheduling
- **Text generation** — sampling from your trained model

## Prerequisites

- Any laptop or desktop (Mac, Linux, or Windows)
- .NET 8 SDK or newer
- Comfort reading C# code (you don't need ML experience)

Training uses NVIDIA GPU (CUDA), Apple Silicon GPU (MPS), or CPU automatically.

## Quick Start

```bash
dotnet run --project src/LlmFromScratch -- train --output-dir artifacts
```

Generate text from the saved final checkpoint:

```bash
dotnet run --project src/LlmFromScratch -- generate --checkpoint artifacts/checkpoints/final --prompt "To be or not"
```

Analyze loss curves and export a CSV:

```bash
dotnet run --project src/LlmFromScratch -- analyze --loss-log artifacts/loss_log.json
```

The default dataset path resolves automatically to `data/shakespeare.txt` when you run from the repository root.

To use CUDA on Windows, pass `UseCuda=true`:

```bash
dotnet run --project src/LlmFromScratch -p:UseCuda=true -- train --output-dir artifacts-cuda
```

## How It Works

Work through the docs in order. Each part walks you through writing a piece of the pipeline, explaining what each component does and why. By the end, you'll have a working GPT that you wrote yourself.

| Part | What You'll Learn | Concepts |
|------|-------------------|----------|
| [Part 1: Tokenization](docs/01-tokenization.md) | Character-level encoding in C# | Character encoding, vocabulary size, why BPE fails on small data |
| [Part 2: The Transformer](docs/02-the-transformer.md) | GPT architecture in TorchSharp | Embeddings, self-attention, layer norm, MLP blocks, weight tying |
| [Part 3: The Training Loop](docs/03-training-loop.md) | Complete training pipeline | Loss functions, AdamW, gradient clipping, LR scheduling, validation |
| [Part 4: Text Generation](docs/04-text-generation.md) | Inference and sampling | Temperature, top-k, autoregressive decoding, reproducibility |
| [Part 5: Putting It All Together](docs/05-putting-it-together.md) | Train on real data, experiment | Loss curves, model scaling, hyperparameter tuning |
| [Part 6: Competition](docs/06-competition.md) | Train the best AI poet | Find datasets, scale up, submit your best poem |

## Project Structure

```
.
├── data/
│   └── shakespeare.txt
├── docs/
│   ├── 01-tokenization.md
│   ├── 02-the-transformer.md
│   ├── 03-training-loop.md
│   ├── 04-text-generation.md
│   ├── 05-putting-it-together.md
│   └── 06-competition.md
└── src/
    └── LlmFromScratch/
        ├── CharacterTokenizer.cs
        ├── GptConfig.cs
        ├── GptModel.cs
        ├── LlmFromScratch.csproj
        ├── LossLogAnalyzer.cs
        ├── Program.cs
        ├── TextDataset.cs
        └── Training.cs
```

## Implementation Notes

- **Architecture**: The GPT architecture is identical to nanoGPT and the PyTorch versions — embeddings, multi-head self-attention, MLP blocks, residual connections, and cross-entropy training.
- **Checkpoints**: Saved as `model.bin` (TorchSharp format) plus `metadata.json` for reproducibility.
- **CLI**: The project exposes `train`, `generate`, and `analyze` subcommands for a complete workflow.
- **Hardware**: Runs on CPU by default, automatically detects and uses CUDA or Apple Silicon GPU when available.
- **Tokenizer**: Character-level by default, designed for small datasets like Shakespeare (~65 tokens).
