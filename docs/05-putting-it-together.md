# Part 5: Putting It All Together

Time to wire everything up, train on real data, and prepare for the competition.

## Project Structure

By now you should have:

```
dotnet/
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
        ├── Program.cs
        ├── TextDataset.cs
        └── Training.cs
```

The Shakespeare dataset is included in the repo at `data/shakespeare.txt` — no download needed.

## Step 1: Train

From the `dotnet` folder:

```bash
dotnet run --project src/LlmFromScratch -- train --output-dir artifacts
```

The default config trains a 6L/6H/384D model (~10M params) on Shakespeare for 5000 steps with batch_size=64. On a modern laptop, expect roughly 30-60 minutes depending on your hardware. You'll see:

- Validation loss + generated samples every 100 steps
- Checkpoints every 1000 steps
- Final checkpoint + loss log at the end

## Step 2: Generate

```bash
dotnet run --project src/LlmFromScratch -- generate --checkpoint artifacts/checkpoints/final --prompt "To be or not"
```

This loads a checkpoint and generates text. You can pass any checkpoint file as an argument.

## Step 3: Experiment

Once the basic pipeline works, try these to build intuition before the competition:

### Model Size vs. Quality

Train three models on the same data and compare output quality:

| Config | Params | n_layer | n_head | n_embd | Expected Loss |
|--------|--------|---------|--------|--------|---------------|
| Tiny | ~0.5M | 2 | 2 | 128 | ~2.0 |
| Small | ~4M | 4 | 4 | 256 | ~1.5 |
| Medium | ~10M | 6 | 6 | 384 | ~1.2 |

Modify the CLI flags:

```bash
# tiny — fast, good for testing ideas
dotnet run --project src/LlmFromScratch -- train --output-dir artifacts-tiny --n-layer 2 --n-head 2 --n-embd 128 --max-steps 2000

# medium — default, good baseline
dotnet run --project src/LlmFromScratch -- train --output-dir artifacts-medium --n-layer 6 --n-head 6 --n-embd 384

# large — needs more data to justify
dotnet run --project src/LlmFromScratch -- train --output-dir artifacts-large --n-layer 12 --n-head 12 --n-embd 768
```

### Context Length

Train with `--block-size 128` vs `--block-size 512`. Longer context lets the model capture full stanzas and rhyme schemes, but uses more memory (reduce batch_size to compensate).

### Learning Rate

Try `3e-4` (conservative), `1e-3` (default), `3e-3` (aggressive). The right LR depends on your model size and data.

## Monitoring Training

The training loop saves `loss_log.json`. You can analyze it with the built-in command:

```bash
dotnet run --project src/LlmFromScratch -- analyze --loss-log artifacts/loss_log.json
```

This exports `loss_log.csv` with columns: `step`, `train_loss`, `val_loss`.

### What to Look For

- **Train loss not decreasing**: Learning rate too low, or a bug
- **Train loss decreasing, val loss increasing**: Overfitting — more data or smaller model
- **Loss spikes**: Reduce learning rate or check gradient clipping
- **Loss plateaus**: Model has learned what it can. More data or bigger model

## Further Reading

- [Karpathy's microgpt](http://karpathy.github.io/2026/02/12/microgpt/) — A full GPT in 200 lines of pure Python
- [build-nanogpt video lecture](https://github.com/karpathy/build-nanogpt) — 4-hour video building GPT-2 from an empty file
- [nanochat](https://github.com/karpathy/nanochat) — Full ChatGPT clone training pipeline
- [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) — The original transformer paper
- [GPT-2 paper (2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — Language models as unsupervised learners
- [TinyStories paper](https://arxiv.org/abs/2305.07759) — Small models trained on curated data
- [Chinchilla (2022)](https://arxiv.org/abs/2203.15556) — Optimal scaling of data vs. parameters

## Next: [Part 6 — Competition: Best AI Poet →](06-competition.md)