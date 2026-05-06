using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace LlmFromScratch;

public sealed class Gpt : Module<Tensor, Tensor>
{
    private readonly Embedding _tokenEmbedding;
    private readonly Embedding _positionEmbedding;
    private readonly Sequential _blocks;
    private readonly LayerNorm _finalNorm;
    private readonly Linear _lmHead;

    public Gpt(GptConfig config) : base(nameof(Gpt))
    {
        Config = config;
        _tokenEmbedding = Embedding(config.VocabSize, config.NEmbd);
        _positionEmbedding = Embedding(config.BlockSize, config.NEmbd);

        var namedBlocks = new List<(string name, Module<Tensor, Tensor> submodule)>();
        for (var layer = 0; layer < config.NLayer; layer++)
        {
            namedBlocks.Add(($"block_{layer}", new Block(config)));
        }

        _blocks = Sequential(namedBlocks);
        _finalNorm = LayerNorm(new long[] { config.NEmbd });
        _lmHead = Linear(config.NEmbd, config.VocabSize, false);
        _lmHead.weight = _tokenEmbedding.weight!;

        init.normal_(_tokenEmbedding.weight!, mean: 0.0, std: 0.02);
        init.normal_(_positionEmbedding.weight!, mean: 0.0, std: 0.02);

        RegisterComponents();
    }

    public GptConfig Config { get; }

    public Device CurrentDevice { get; private set; } = CPU;

    public override Tensor forward(Tensor idx)
    {
        var batchSize = idx.shape[0];
        var time = idx.shape[1];
        if (time > Config.BlockSize)
        {
            throw new ArgumentException($"Sequence length {time} exceeds block size {Config.BlockSize}.");
        }

        using var positions = torch.arange(time, dtype: ScalarType.Int64).unsqueeze(0).to(CurrentDevice);
        using var tokenEmbeddings = _tokenEmbedding.call(idx);
        using var positionEmbeddings = _positionEmbedding.call(positions);
        using var x0 = tokenEmbeddings + positionEmbeddings.expand(batchSize, time, Config.NEmbd);
        using var x1 = _blocks.call(x0);
        using var x2 = _finalNorm.call(x1);
        return _lmHead.call(x2);
    }

    protected override torch.nn.Module _to(DeviceType deviceType, int deviceIndex, bool nonBlocking)
    {
        CurrentDevice = new Device(deviceType, deviceIndex);
        return base._to(deviceType, deviceIndex, nonBlocking);
    }

    private sealed class Block : Module<Tensor, Tensor>
    {
        private readonly LayerNorm _ln1;
        private readonly CausalSelfAttention _attention;
        private readonly LayerNorm _ln2;
        private readonly Mlp _mlp;

        public Block(GptConfig config) : base(nameof(Block))
        {
            _ln1 = LayerNorm(new long[] { config.NEmbd });
            _attention = new CausalSelfAttention(config);
            _ln2 = LayerNorm(new long[] { config.NEmbd });
            _mlp = new Mlp(config);
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            using var a = _attention.call(_ln1.call(x));
            using var h = x + a;
            using var m = _mlp.call(_ln2.call(h));
            return h + m;
        }
    }

    private sealed class CausalSelfAttention : Module<Tensor, Tensor>
    {
        private readonly Linear _cAttn;
        private readonly Linear _cProj;
        private readonly int _nHead;
        private readonly int _nEmbd;

        public CausalSelfAttention(GptConfig config) : base(nameof(CausalSelfAttention))
        {
            if (config.NEmbd % config.NHead != 0)
            {
                throw new ArgumentException("Embedding size must be divisible by the number of heads.");
            }

            _cAttn = Linear(config.NEmbd, 3 * config.NEmbd);
            _cProj = Linear(config.NEmbd, config.NEmbd);
            _nHead = config.NHead;
            _nEmbd = config.NEmbd;

            init.normal_(_cAttn.weight!, mean: 0.0, std: 0.02);
            init.zeros_(_cAttn.bias!);
            init.normal_(_cProj.weight!, mean: 0.0, std: 0.02);
            init.zeros_(_cProj.bias!);

            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            var batchSize = x.shape[0];
            var time = x.shape[1];
            var channels = x.shape[2];
            var headDim = channels / _nHead;

            using var qkv = _cAttn.call(x);
            var parts = qkv.split(_nEmbd, dim: 2);
            using var q = parts[0].view(batchSize, time, _nHead, headDim).transpose(1, 2);
            using var k = parts[1].view(batchSize, time, _nHead, headDim).transpose(1, 2);
            using var v = parts[2].view(batchSize, time, _nHead, headDim).transpose(1, 2);
            using var y = functional.scaled_dot_product_attention(q, k, v, is_casual: true);
            using var merged = y.transpose(1, 2).contiguous().view(batchSize, time, channels);
            return _cProj.call(merged);
        }
    }

    private sealed class Mlp : Module<Tensor, Tensor>
    {
        private readonly Linear _cFc;
        private readonly Module<Tensor, Tensor> _gelu;
        private readonly Linear _cProj;

        public Mlp(GptConfig config) : base(nameof(Mlp))
        {
            _cFc = Linear(config.NEmbd, 4 * config.NEmbd);
            _gelu = GELU();
            _cProj = Linear(4 * config.NEmbd, config.NEmbd);

            init.normal_(_cFc.weight!, mean: 0.0, std: 0.02);
            init.zeros_(_cFc.bias!);
            init.normal_(_cProj.weight!, mean: 0.0, std: 0.02);
            init.zeros_(_cProj.bias!);

            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            using var expanded = _cFc.call(x);
            using var activated = _gelu.call(expanded);
            return _cProj.call(activated);
        }
    }
}