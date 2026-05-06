# Part 2: The Transformer

This is the core of the workshop. You'll write the full GPT model architecture from scratch in TorchSharp.

## The Big Picture

A GPT is an **autoregressive language model**: given a sequence of tokens, it predicts the next one. Stack this prediction in a loop and you get text generation.

The architecture is a stack of identical **transformer blocks**, each containing:
1. **Multi-head self-attention** — lets each token look at all previous tokens
2. **Feed-forward network (MLP)** — processes each position independently
3. **Residual connections** — add the input back to the output of each sub-layer
4. **Layer normalization** — stabilizes training

## The Model: `GptModel.cs`

The GPT model lives in `src/LlmFromScratch/GptModel.cs` and contains all the components you'll learn about here:

## Configuration

The configuration type is `GptConfig`:

```csharp
public sealed class GptConfig
{
    public int VocabSize { get; set; } = 65;       // character-level: 65 unique chars in Shakespeare
    public int BlockSize { get; set; } = 256;      // max sequence length (context window)
    public int NLayer { get; set; } = 6;           // number of transformer blocks
    public int NHead { get; set; } = 6;            // number of attention heads
    public int NEmbd { get; set; } = 384;          // embedding dimension
}
```

`VocabSize` comes from the tokenizer (65 characters for Shakespeare). `BlockSize` is the maximum number of tokens the model can see at once. `NEmbd` is the width of the model — every hidden state is a vector of this size.

During training, `VocabSize` is updated from the tokenizer before the model is created.

## GPT Module

The top-level model is `Gpt`, which derives from TorchSharp's `Module<Tensor, Tensor>`:

```csharp
public sealed class Gpt : Module<Tensor, Tensor>
{
    private readonly Embedding _tokenEmbedding;
    private readonly Embedding _positionEmbedding;
    private readonly Sequential _blocks;
    private readonly LayerNorm _finalNorm;
    private readonly Linear _lmHead;
}
```

The model uses:

- `Embedding(config.VocabSize, config.NEmbd)` for token embeddings
- `Embedding(config.BlockSize, config.NEmbd)` for positional embeddings
- `Sequential(...)` to hold the transformer blocks
- `Linear(config.NEmbd, config.VocabSize, false)` for logits

Two embedding tables:
- **Token embedding** (`_tokenEmbedding`): maps each token ID to a learned vector. Size: `[65, 384]`
- **Position embedding** (`_positionEmbedding`): maps each position (0 to 255) to a learned vector. Size: `[256, 384]`

**Weight tying**: the same matrix that maps tokens → embeddings is reused (transposed) to map embeddings → logits at the output:

```csharp
_lmHead.weight = _tokenEmbedding.weight!;
```

This reduces parameters and improves training — the model's input and output representations of tokens are forced to be consistent. With our small vocab of 65 this saves very little, but it's standard practice and matters a lot with large vocabularies.

## Forward Pass

The forward pass is the usual GPT flow:

```csharp
using var positions = torch.arange(time, dtype: ScalarType.Int64).unsqueeze(0).to(CurrentDevice);
using var tokenEmbeddings = _tokenEmbedding.call(idx);
using var positionEmbeddings = _positionEmbedding.call(positions);
using var x0 = tokenEmbeddings + positionEmbeddings.expand(batchSize, time, Config.NEmbd);
using var x1 = _blocks.call(x0);
using var x2 = _finalNorm.call(x1);
return _lmHead.call(x2);
```

So the shape flow is:

- input token IDs: `(B, T)`
- embeddings: `(B, T, n_embd)`
- transformer stack: `(B, T, n_embd)`
- logits: `(B, T, vocab_size)`

The position embedding is added to the token embedding — this is how the model knows word order. Without it, "the dog bit the man" and "the man bit the dog" would look identical.

## Self-Attention

This is the mechanism that lets each token attend to (look at) every previous token in the sequence.

```csharp
class CausalSelfAttention : Module<Tensor, Tensor>
{
    private readonly Linear _cAttn;
    private readonly Linear _cProj;
    private readonly int _nHead;
    private readonly int _nEmbd;

    public CausalSelfAttention(GptConfig config)
    {
        _cAttn = Linear(config.NEmbd, 3 * config.NEmbd);  // Q, K, V projections
        _cProj = Linear(config.NEmbd, config.NEmbd);       // output projection
        _nHead = config.NHead;
        _nEmbd = config.NEmbd;
    }

    public override Tensor forward(Tensor x)
    {
        var (batchSize, time, embedDim) = x.shape;
        using var qkv = _cAttn.call(x);
        var parts = qkv.split(_nEmbd, dim: 2);
        using var q = parts[0].view(batchSize, time, _nHead, embedDim / _nHead).transpose(1, 2);
        using var k = parts[1].view(batchSize, time, _nHead, embedDim / _nHead).transpose(1, 2);
        using var v = parts[2].view(batchSize, time, _nHead, embedDim / _nHead).transpose(1, 2);

        // attention with causal mask (each token can only attend to previous tokens)
        using var y = functional.scaled_dot_product_attention(q, k, v, is_casual: true);

        using var concatenated = y.transpose(1, 2).contiguous().view(batchSize, time, embedDim);
        return _cProj.call(concatenated);
    }
}
```

Breaking this down:

1. **Q, K, V projections**: A single linear layer projects the input into three matrices — Query, Key, and Value. Each is `(B, T, n_embd)`.

2. **Multi-head reshape**: We split the embedding dimension into `n_head` separate heads, each with `head_dim = n_embd / n_head = 64` dimensions. This lets the model attend to different aspects of the input in parallel.

3. **Scaled dot-product attention**: For each query position, compute a similarity score against all key positions, mask out future positions (causal), apply softmax, and use the result to weight the values. The math: `softmax(QK^T / sqrt(head_dim)) @ V`

4. **Causal masking** (`is_casual=true`): Position `i` can only attend to positions `0..i`. This prevents the model from "cheating" by looking at future tokens during training. This is what makes GPT **autoregressive**.

5. **Output projection**: Concatenate all heads and project back to `n_embd` dimensions.

### Why Multi-Head?

With 6 heads of 64 dimensions each (instead of one head of 384 dimensions), the model can simultaneously track different relationships: one head might track which vowels follow consonants, another might track line-break patterns, another might focus on recent context.

## MLP Block

```csharp
class MLP : Module<Tensor, Tensor>
{
    private readonly Linear _cFc;
    private readonly GELU _gelu;
    private readonly Linear _cProj;

    public MLP(GptConfig config)
    {
        _cFc = Linear(config.NEmbd, 4 * config.NEmbd);      // project up: 384 → 1536
        _gelu = GELU(approximate: 'tanh');
        _cProj = Linear(4 * config.NEmbd, config.NEmbd);    // project back down: 1536 → 384
    }

    public override Tensor forward(Tensor x)
    {
        using var expanded = _cFc.call(x);
        using var activated = _gelu.call(expanded);
        return _cProj.call(activated);
    }
}
```

The MLP is applied independently to each position. It expands the representation to 4x the embedding dimension, applies a non-linearity (GELU), and projects back down. This is where the model does most of its "thinking" — the attention gathers information, the MLP processes it.

**Why GELU instead of ReLU?** GELU (Gaussian Error Linear Unit) is smoother than ReLU. It doesn't have a hard cutoff at zero, which helps gradient flow. GPT-2 uses the `tanh` approximation for speed.

## Transformer Block

```csharp
class Block : Module<Tensor, Tensor>
{
    private readonly LayerNorm _ln1;
    private readonly CausalSelfAttention _attn;
    private readonly LayerNorm _ln2;
    private readonly MLP _mlp;

    public Block(GptConfig config)
    {
        _ln1 = LayerNorm(config.NEmbd);
        _attn = new CausalSelfAttention(config);
        _ln2 = LayerNorm(config.NEmbd);
        _mlp = new MLP(config);
    }

    public override Tensor forward(Tensor x)
    {
        using var attnNorm = _ln1.call(x);
        using var attnOut = _attn.forward(attnNorm);
        using var x1 = x + attnOut;  // residual connection

        using var mlpNorm = _ln2.call(x1);
        using var mlpOut = _mlp.forward(mlpNorm);
        return x1 + mlpOut;  // residual connection
    }
}
```

Two key design choices:

1. **Pre-norm** (LayerNorm before attention/MLP, not after): This stabilizes training by normalizing inputs to each sub-layer. The original transformer paper used post-norm, but pre-norm is now standard.

2. **Residual connections** (`x = x + sublayer(x)`): The input is added back to the output. This lets gradients flow directly through the network during backpropagation, making deep networks trainable. Without residuals, a 6-layer network would be much harder to train.

## Parameter Count

Where do the parameters live?
- Token embeddings: `65 × 384 = 25K` (tiny with char-level vocab — shared with lm_head)
- Position embeddings: `256 × 384 = 98K`
- Per transformer block: `~1.8M` (attention: 4 × 384² = 590K, MLP: 2 × 384 × 1536 = 1.2M, norms: negligible)
- 6 blocks: `~10.6M`
- Total: `~10.8M`

Notice how almost all the parameters are in the transformer blocks, not the embeddings. With GPT-2's 50k vocab, the embedding table would be 50,257 × 384 = 19.3M — nearly double the entire model. This is why vocab size matters.

## Key Takeaways

- A GPT is a stack of identical transformer blocks
- Each block: LayerNorm → Self-Attention → Residual → LayerNorm → MLP → Residual
- Self-attention lets tokens look at all previous tokens (causal masking prevents looking ahead)
- Multi-head attention runs multiple attention patterns in parallel
- Residual connections and layer norm make deep networks trainable
- Weight tying between input embeddings and output projection reduces parameters

## Next: [Part 3 — The Training Loop →](03-training-loop.md)