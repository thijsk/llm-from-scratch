# Part 3: The Training Loop

You have a model. Now you need to teach it language. The training loop is where the model actually learns, and every decision here affects whether your model converges or diverges into nonsense.

## The Training Objective

GPT is trained with **next-token prediction**: given tokens `[t0, t1, ..., tn]`, predict `[t1, t2, ..., tn+1]`. The loss function is cross-entropy between the model's predicted probability distribution and the actual next token.

This is a self-supervised task — the labels come from the data itself. Every piece of text is simultaneously input and target, just shifted by one position.

## Implementation: `Training.cs`

Training logic lives in `src/LlmFromScratch/Training.cs` and handles:

## Step 1: Data Loading (Character-Level)

The dataset is loaded in `TextDataset.cs`:

```csharp
var dataset = TextDataset.Load(dataPath, blockSize);
var trainBatch = dataset.GetTrainBatch(batchSize);
var valBatch = dataset.GetValidationBatch(batchSize);
```

Each batch:
- Samples `batch_size` random starting positions
- `x`: characters from position `i` to `i + block_size` (input)
- `y`: characters from position `i+1` to `i + block_size + 1` (target — shifted by one)

The function also provides `stoi`/`itos` mappings for text generation.

## Step 2: Device Setup

The project auto-detects the device:

```csharp
public static Device GetDevice()
    => torch.cuda.is_available() ? torch.CUDA : torch.CPU;
```

On a MacBook with Apple Silicon, MPS gives roughly 2-3x speedup over CPU (when available).

## Step 3: Learning Rate Schedule

```csharp
public static double Get(int step, int warmupSteps, int maxSteps, double maxLr, double minLr)
{
    if (step < warmupSteps) {
        return maxLr * (step + 1) / warmupSteps;
    }

    if (step >= maxSteps) {
        return minLr;
    }

    var progress = (step - warmupSteps) / (double)(maxSteps - warmupSteps);
    return minLr + 0.5 * (maxLr - minLr) * (1 + Math.Cos(Math.PI * progress));
}
```

Two phases:

1. **Warmup** (first ~100 steps): Ramp the learning rate from near-zero up to `max_lr`. This gives the optimizer time to calibrate its moment estimates before making large updates.

2. **Cosine decay** (remaining steps): Smoothly decrease the learning rate. Large updates early (explore), small updates late (refine).

## Step 4: The Optimizer

```csharp
using var optimizer = torch.optim.AdamW(model.parameters(), lr: options.MaxLr, weight_decay: 0.01);
```

For character-level training on Shakespeare, plain `AdamW` with `lr=1e-3` and light weight decay works well. The full GPT-2 recipe (separate decay groups, betas=(0.9, 0.95), weight_decay=0.1) is designed for large-scale BPE training and is overkill here.

## Step 5: The Full Training Loop

The training loop is implemented in the `Trainer` class. Key components:

- Forward pass: `logits = model(x)`, loss = `cross_entropy(logits, y)`
- Backward pass: `loss.backward()`
- Gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters().ToArray(), 1.0)`
- Optimizer step: `optimizer.step()`

Every 100 steps:
- Compute validation loss on 20 held-out batches
- Generate a sample from the current model
- Log both to stdout and to `loss_log.json`

Every 1000 steps:
- Save a checkpoint with the model weights, config, and tokenizer

This means you can watch the model learn in real-time without waiting for training to finish.

## Step 6: Entry Point

The CLI entry point in `Program.cs` exposes the `train` command:

```bash
dotnet run --project src/LlmFromScratch -- train --output-dir artifacts
```

Useful overrides:

```bash
dotnet run --project src/LlmFromScratch -- train \
  --output-dir artifacts --max-steps 2000 --batch-size 32 \
  --n-layer 4 --n-head 4 --n-embd 256
```

## Checkpoint Format

Each checkpoint directory contains:

- `model.bin` — TorchSharp weights and buffers
- `metadata.json` — config, vocabulary, and step number

So a final checkpoint looks like:

```text
artifacts/
└── checkpoints/
    └── final/
        ├── metadata.json
        └── model.bin
```

The trainer also writes `loss_log.json` at the output directory root.

## Monitoring Training

The training loop saves `loss_log.json`. You can analyze it with the built-in command:

```bash
dotnet run --project src/LlmFromScratch -- analyze --loss-log artifacts/loss_log.json
```

This writes:

- `loss_summary.json` with best validation step/loss and overfitting gap
- `loss_log.csv` for plotting in Excel, Python, or any charting tool

The CSV has columns: `step`, `train_loss`, `val_loss` (if validation ran at that step).

### What to Look For

- **Train loss not decreasing**: Learning rate too low, or a bug
- **Train loss decreasing, val loss increasing**: Overfitting — more data or smaller model
- **Loss spikes**: Reduce learning rate or check gradient clipping
- **Loss plateaus**: Model has learned what it can. More data or bigger model

## Key Takeaways

- Next-token prediction is self-supervised: labels come from shifting the text by one position
- Warmup + cosine decay is a standard LR schedule
- Validation loss tells you if the model is overfitting
- Checkpoints let you compare outputs at different training stages
- Loss curves guide hyperparameter tuning

## Next: [Part 4 — Text Generation →](04-text-generation.md)