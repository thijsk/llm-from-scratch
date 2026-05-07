# Train Your Own LLM From Scratch (.NET)

A hands-on workshop where you write every piece of a GPT training pipeline yourself, understanding what each component does and why.

This repository is a .NET/TorchSharp fork of [angelos-p/llm-from-scratch](https://github.com/angelos-p/llm-from-scratch). Full credit for the original workshop structure, narrative, and Python implementation goes to [@angelos-p](https://github.com/angelos-p).

Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) is a key inspiration for this project. It shows how a working language model can be built in a few hundred lines of PyTorch and helped motivate the original workshop.

This .NET version aims to give that same experience in C#, keeping the project focused on essentials and scaling it to a ~10M parameter model that trains on a laptop in under an hour.

## What You'll Build

A working GPT model trained from scratch on your laptop, capable of generating Shakespeare-like text. You'll write:

- **Tokenizer** — turning text into numbers the model can process
- **Model architecture** — the transformer: embeddings, attention, feed-forward layers
- **Training loop** — forward pass, loss, backprop, optimizer, learning rate scheduling
- **Text generation** — sampling from your trained model

## Prerequisites

- Any laptop or desktop (Mac, Linux, or Windows)
- .NET 8 SDK+
- Comfort reading C# code (you don't need ML experience)

Training uses NVIDIA GPU (CUDA) or CPU.

## Getting Started

### Local (recommended)

Clone this repo and run from the repository root:

```bash
dotnet restore
dotnet build src/LlmFromScratch/LlmFromScratch.csproj -c Release
dotnet run --project src/LlmFromScratch -- train --output-dir artifacts
```

Generate text from the saved final checkpoint:

```bash
dotnet run --project src/LlmFromScratch -- generate --checkpoint artifacts/checkpoints/final --prompt "To be or not"
```

Analyze loss logs and export CSV/summary JSON:

```bash
dotnet run --project src/LlmFromScratch -- analyze --loss-log artifacts/loss_log.json
```

Windows CUDA build (if CUDA-enabled LibTorch is configured):

```bash
dotnet run --project src/LlmFromScratch -p:UseCuda=true -- train --output-dir artifacts-cuda
```

---

Work through the docs in order. Each part walks you through writing a piece of the pipeline, explaining what each component does and why. By the end, you'll have a working `CharacterTokenizer.cs`, `GptModel.cs`, `Training.cs`, and `Program.cs` that you wrote yourself.

| Part | What You'll Write | Concepts |
|------|-------------------|----------|
| [Part 1: Tokenization](docs/01-tokenization.md) | Character-level tokenizer | Character encoding, vocabulary size, why BPE fails on small data |
| [Part 2: The Transformer](docs/02-the-transformer.md) | Full GPT model architecture | Embeddings, self-attention, layer norm, MLP blocks |
| [Part 3: The Training Loop](docs/03-training-loop.md) | Complete training pipeline | Loss functions, AdamW, gradient clipping, LR scheduling |
| [Part 4: Text Generation](docs/04-text-generation.md) | Inference and sampling | Temperature, top-k, autoregressive decoding |
| [Part 5: Putting It All Together](docs/05-putting-it-together.md) | Train on real data, experiment | Loss curves, scaling experiments, next steps |
| [Part 6: Competition](docs/06-competition.md) | Train the best AI poet | Find datasets, scale up, submit your best poem |

## Architecture: GPT at a Glance

```
Input Text
    |
    v
+-----------------+
|   Tokenizer     |  "hello" -> [20, 43, 50, 50, 53]  (character-level)
+--------+--------+
         v
+-----------------+
|  Token Embed +  |  token IDs -> vectors (n_embd dimensions)
|  Position Embed |  + positional information
+--------+--------+
         v
+-----------------+
|  Transformer    |  x n_layer
|  Block:         |
|  +------------+ |
|  | LayerNorm  | |
|  | Self-Attn  | |  n_head parallel attention heads
|  | + Residual | |
|  +------------+ |
|  | LayerNorm  | |
|  | MLP (FFN)  | |  expand 4x, GELU, project back
|  | + Residual | |
|  +------------+ |
+--------+--------+
         v
+-----------------+
|   LayerNorm     |
|   Linear -> logits|  vocab_size outputs (next-token probabilities)
+-----------------+
```

## Model Configs for This Workshop

| Config | Params | n_layer | n_head | n_embd | Approx Train Time |
|--------|--------|---------|--------|--------|-------------------|
| Tiny | ~0.5M | 2 | 2 | 128 | ~5 min |
| Small | ~4M | 4 | 4 | 256 | ~20 min |
| **Medium (default)** | **~10M** | **6** | **6** | **384** | **~45 min** |

All configs use character-level tokenization (vocab_size ~= 65) and block_size = 256.

## Tokenization: Characters vs BPE

This workshop uses **character-level** tokenization on Shakespeare. BPE tokenization (GPT-2's ~50k vocab) doesn't work well on small datasets, where most token bigrams are too rare for the model to learn robust patterns from.

| Tokenizer | Vocab Size | Dataset Size Needed |
|-----------|-----------|-------------------|
| **Character-level** | ~65 | Small (Shakespeare, ~1MB) |
| **BPE** | ~50,000 | Large (TinyStories+, 100MB+) |

Part 5 covers scaling up to larger corpora and tokenization strategy trade-offs.

## Key References

- [angelos-p/llm-from-scratch](https://github.com/angelos-p/llm-from-scratch) — Original workshop repository this .NET fork is based on
- [nanoGPT](https://github.com/karpathy/nanoGPT) — Minimal GPT training in ~300 lines of PyTorch
- [build-nanogpt video lecture](https://github.com/karpathy/build-nanogpt) — 4-hour video building GPT-2 from an empty file
- [Karpathy's microgpt](http://karpathy.github.io/2026/02/12/microgpt/) — A full GPT in 200 lines of pure Python, no dependencies
- [nanochat](https://github.com/karpathy/nanochat) — Full ChatGPT clone training pipeline
- [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) — The original transformer paper
- [GPT-2 paper (2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — Language models as unsupervised learners
- [TinyStories paper](https://arxiv.org/abs/2305.07759) — Why small models trained on curated data punch above their weight
