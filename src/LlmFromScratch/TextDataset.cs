using TorchSharp;
using static TorchSharp.torch;

namespace LlmFromScratch;

public sealed class TextDataset
{
    private readonly long[] _tokens;
    private readonly int _trainLength;
    private readonly Random _random;

    private TextDataset(long[] tokens, CharacterTokenizer tokenizer, int seed)
    {
        _tokens = tokens;
        Tokenizer = tokenizer;
        _trainLength = (int)(tokens.Length * 0.9);
        _random = new Random(seed);
    }

    public CharacterTokenizer Tokenizer { get; }

    public int Length => _tokens.Length;

    public static TextDataset Load(string path, int seed, bool asciiOnly)
    {
        var text = File.ReadAllText(path);

        if (asciiOnly)
        {
            text = new string(text.Where(IsAllowedAscii).ToArray());
            if (text.Length == 0)
            {
                throw new InvalidOperationException("ASCII filtering removed all characters from the dataset.");
            }
        }

        var tokenizer = CharacterTokenizer.FromText(text);
        var tokens = tokenizer.Encode(text);
        return new TextDataset(tokens, tokenizer, seed);
    }

    private static bool IsAllowedAscii(char character)
        => character == '\n'
            || character == '\r'
            || character == '\t'
            || (character >= ' ' && character <= '~');

    public (Tensor X, Tensor Y) GetTrainBatch(int blockSize, int batchSize, Device device)
        => GetBatch(0, _trainLength, blockSize, batchSize, device);

    public (Tensor X, Tensor Y) GetValidationBatch(int blockSize, int batchSize, Device device)
        => GetBatch(_trainLength, _tokens.Length, blockSize, batchSize, device);

    private (Tensor X, Tensor Y) GetBatch(int start, int end, int blockSize, int batchSize, Device device)
    {
        var maxStart = end - start - blockSize - 1;
        if (maxStart <= 0)
        {
            throw new InvalidOperationException("Dataset split is smaller than the configured block size.");
        }

        var xRows = new Tensor[batchSize];
        var yRows = new Tensor[batchSize];

        for (var batch = 0; batch < batchSize; batch++)
        {
            var offset = start + _random.Next(maxStart);
            xRows[batch] = tensor(_tokens.AsSpan(offset, blockSize).ToArray(), dtype: ScalarType.Int64);
            yRows[batch] = tensor(_tokens.AsSpan(offset + 1, blockSize).ToArray(), dtype: ScalarType.Int64);
        }

        var x = torch.stack(xRows).to(device);
        var y = torch.stack(yRows).to(device);

        foreach (var row in xRows)
        {
            row.Dispose();
        }

        foreach (var row in yRows)
        {
            row.Dispose();
        }

        return (x, y);
    }
}