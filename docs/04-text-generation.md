# Part 4: Text Generation

Your model is trained. Now let's make it write. Text generation with a GPT is **autoregressive**: generate one token at a time, append it to the input, and repeat.

Generation is handled by `TextGenerator.Generate()` in `Training.cs` and exposed through the `generate` CLI command.

## The Naive Approach: Greedy Decoding

Always pick the most probable next token.

```csharp
Tensor nextToken = logits.argmax(dim: -1, keepdim: true);
```

This is deterministic — the same prompt always produces the same output. It tends to be repetitive and boring because the highest-probability continuation reinforces itself.

## Temperature

Scale the logits before applying softmax. Higher temperature = more random, lower = more deterministic.

```csharp
logits = logits / temperature;
```

The math: softmax computes `exp(logit_i) / sum(exp(logit_j))`. Dividing all logits by temperature changes the distribution:
- **T = 1.0**: Normal probabilities
- **T → 0**: Approaches greedy (argmax)
- **T > 1.0**: Flattens the distribution, giving rare tokens more chance
- **T = 0.7-0.9**: The typical sweet spot for coherent but varied text

## Top-k Sampling

Only consider the k most probable tokens. Set everything else to `-inf`.

```csharp
(values, _) = torch.topk(logits, topK, dim: -1);
threshold = values.select(1, topK - 1).unsqueeze(1);
logits = logits.masked_fill(logits < threshold, double.NegativeInfinity);
```

This prevents the model from sampling extremely unlikely tokens. With a character-level model (vocab=65), `top_k=40` is reasonable — it still considers most characters but excludes the very unlikely ones.

## The Full Generate Function

```csharp
public static string Generate(
    Gpt model, 
    string prompt, 
    CharacterTokenizer tokenizer,
    int maxNewTokens = 200,
    double temperature = 0.8,
    int topK = 40,
    int? seed = null)
{
    if (seed.HasValue)
        torch.manual_seed(seed.Value);

    var device = model.weights.device;
    var tokens = prompt.Select(c => tokenizer.Encode(c.ToString())[0]).ToArray();
    var idx = torch.tensor(new long[][] { tokens }, dtype: ScalarType.Int64, device: device);

    model.eval();
    for (int i = 0; i < maxNewTokens; i++)
    {
        var idxCond = idx.narrow(1, Math.Max(0, idx.shape[1] - model.Config.BlockSize), 
                                    Math.Min(idx.shape[1], model.Config.BlockSize));
        var logits = model.call(idxCond);
        var nextLogits = logits.select(1, logits.shape[1] - 1) / temperature;

        if (topK > 0)
        {
            (var values, _) = torch.topk(nextLogits, topK, dim: -1);
            var threshold = values.select(1, topK - 1).unsqueeze(1);
            nextLogits = nextLogits.masked_fill(nextLogits < threshold, double.NegativeInfinity);
        }

        var probs = torch.softmax(nextLogits, dim: -1);
        var nextToken = torch.multinomial(probs, 1);
        idx = torch.cat(new[] { idx, nextToken }, dim: 1);
    }

    var ids = idx[0].to(torch.CPU).to(ScalarType.Int64).data<long>().ToArray();
    return tokenizer.Decode(ids);
}
```

The pipeline for each token:
1. Run the model on the current sequence → get logits for next position
2. Apply temperature scaling
3. Filter with top-k (remove very unlikely tokens)
4. Convert to probabilities with softmax
5. Sample from the distribution with `multinomial`
6. Append the sampled token and repeat

`model.eval()` disables dropout and batch norm — we don't need training-mode behavior for inference and it saves computation.

The function takes `stoi`/`itos` mappings from the training data — these define how characters map to token IDs and back.

## Reproducibility with Seeds

Generation involves random sampling (`torch.multinomial`), so the same prompt produces different output each time. To get reproducible results, set a seed before generating:

```bash
dotnet run --project src/LlmFromScratch -- generate \
  --checkpoint artifacts/checkpoints/final \
  --prompt "To be or not" \
  --seed 42
```

Same seed + same checkpoint + same arguments will always produce the same output.

## Try Different Settings

```csharp
checkpoint = ...  // load checkpoint

// deterministic, repetitive
generate(model, "To be or not to be", tokenizer, temperature: 0.1);

// balanced
generate(model, "To be or not to be", tokenizer, temperature: 0.8);

// creative, potentially incoherent
generate(model, "To be or not to be", tokenizer, temperature: 1.5);
```

## What to Expect

Here are real samples from a training run (6L/6H/384D on Shakespeare):

### Step 200 (val loss ~3.5) — Random characters
```
To be or notis p ce mei odorethleedetire'ilethed ye m arkesothir fnon b tigb'i.
```

### Step 1000 (val loss 1.64) — Words and structure emerging
```
To be or nothing are good men,
The profent of little, our actory.

CORIOLANUS:
Is it now of your many death?
```

### Step 2400 (val loss ~1.60) — Peak quality, plausible Shakespeare
```
To be or not to be some of you shall know
That everlature by Romeo: what news,
Which you had knock'd my part to speak
```

Note: the best output is around step 1500-2500. After that, the model overfits and starts regurgitating memorized training data.

## Key Takeaways

- Autoregressive generation: predict one token, append, repeat
- Greedy decoding is deterministic and repetitive
- Temperature controls randomness (0.7-0.9 is usually good)
- Top-k removes extremely unlikely tokens
- With character-level models, generate samples during training to watch the model learn

## Command-Line Interface

Generate from a trained checkpoint directory:

```bash
dotnet run --project src/LlmFromScratch -- generate --checkpoint artifacts/checkpoints/final --prompt "To be or not"
```

With explicit sampling controls:

```bash
dotnet run --project src/LlmFromScratch -- generate \
  --checkpoint artifacts/checkpoints/final \
  --prompt "To be or not" \
  --max-new-tokens 200 \
  --temperature 0.8 \
  --top-k 40 \
  --seed 42
```

## Next: [Part 5 — Putting It All Together →](05-putting-it-together.md)