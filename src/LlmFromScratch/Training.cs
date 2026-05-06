using System.Text.Json;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace LlmFromScratch;

public static class Trainer
{
    public static void Train(TrainOptions options)
    {
        torch.random.manual_seed(options.Seed);
        Directory.CreateDirectory(options.OutputDir);

        if (options.NumThreads > 0)
        {
            torch.set_num_threads(options.NumThreads);
        }

        var device = DeviceSelector.GetDevice();
        Console.WriteLine($"Using device: {device.type}, threads: {torch.get_num_threads()}");

        var dataset = TextDataset.Load(options.DataPath, options.Seed, options.AsciiOnly);
        options.Config.VocabSize = dataset.Tokenizer.VocabSize;

        Console.WriteLine($"Dataset: {dataset.Length:N0} chars, vocab size: {dataset.Tokenizer.VocabSize}");
        if (options.AsciiOnly)
        {
            Console.WriteLine("Preprocessing: ASCII-only (non-ASCII chars removed)");
        }

        using var model = new Gpt(options.Config);
        model.to(device);

        var parameterCount = model.parameters().Sum(parameter => parameter.shape.Aggregate(1L, (count, size) => count * size));
        Console.WriteLine($"Model: {options.Config.NLayer}L/{options.Config.NHead}H/{options.Config.NEmbd}D, {parameterCount / 1e6:F1}M params");

        using var optimizer = torch.optim.AdamW(model.parameters(), lr: options.MaxLr, weight_decay: 0.01);
        using var lossFn = CrossEntropyLoss();

        var log = new LossLog();
        double latestValidationLoss = double.NaN;
        var tokensPerStep = options.BatchSize * options.Config.BlockSize * options.GradAccumSteps;
        var overallTimer = System.Diagnostics.Stopwatch.StartNew();
        var windowTimer = System.Diagnostics.Stopwatch.StartNew();
        var windowSteps = 0;

        for (var step = 0; step < options.MaxSteps; step++)
        {
            if (step % options.ValidationEvery == 0)
            {
                latestValidationLoss = Evaluate(model, dataset, lossFn, options.Config.BlockSize, options.BatchSize, device);
                Console.WriteLine($"Step {step,5} | val loss: {latestValidationLoss:F4}");
            }

            var learningRate = LearningRateSchedule.Get(step, options.WarmupSteps, options.MaxSteps, options.MaxLr, options.MinLr);
            foreach (var group in optimizer.ParamGroups)
            {
                group.LearningRate = learningRate;
            }

            optimizer.zero_grad();
            double trainLoss = 0;
            for (var accumStep = 0; accumStep < options.GradAccumSteps; accumStep++)
            {
                using (torch.NewDisposeScope())
                {
                    using var batch = BatchScope.Create(dataset.GetTrainBatch(options.Config.BlockSize, options.BatchSize, device));
                    using var logits = model.call(batch.X);
                    using var loss = lossFn.call(logits.view(-1, options.Config.VocabSize), batch.Y.view(-1));
                    using var scaledLoss = loss / options.GradAccumSteps;

                    scaledLoss.backward();
                    trainLoss += loss.to(torch.CPU).item<float>() / options.GradAccumSteps;
                }
            }

            torch.nn.utils.clip_grad_norm_(model.parameters().ToArray(), 1.0);
            optimizer.step();

            windowSteps++;
            log.Steps.Add(step);
            log.Train.Add(trainLoss);
            if (step % options.ValidationEvery == 0)
            {
                log.Val.Add(latestValidationLoss);
            }

            if (step % 50 == 0)
            {
                var windowSecs = windowTimer.Elapsed.TotalSeconds;
                var tokensPerSec = windowSecs > 0 ? (long)(windowSteps * tokensPerStep / windowSecs) : 0;
                var stepsRemaining = options.MaxSteps - step - 1;
                var avgStepSecs = overallTimer.Elapsed.TotalSeconds / (step + 1);
                var eta = TimeSpan.FromSeconds(stepsRemaining * avgStepSecs);
                var etaStr = eta.TotalHours >= 1
                    ? $"{(int)eta.TotalHours}h{eta.Minutes:D2}m"
                    : eta.TotalMinutes >= 1
                        ? $"{(int)eta.TotalMinutes}m{eta.Seconds:D2}s"
                        : $"{eta.Seconds}s";
                Console.WriteLine($"Step {step,5} | train loss: {trainLoss:F4} | lr: {learningRate:E2} | {tokensPerSec:N0} tok/s | ETA: {etaStr}");
                windowTimer.Restart();
                windowSteps = 0;
            }

            if (step > 0 && step % options.SampleEvery == 0)
            {
                var sample = TextGenerator.Generate(model, dataset.Tokenizer, options.SamplePrompt, 100, options.SampleTemperature, options.SampleTopK);
                Console.WriteLine();
                Console.WriteLine($"--- Step {step} sample ---");
                Console.WriteLine(sample);
                Console.WriteLine("---");
                Console.WriteLine();
            }

            if (step > 0 && step % options.CheckpointEvery == 0)
            {
                var checkpointDir = Path.Combine(options.OutputDir, "checkpoints", $"step_{step:D5}");
                CheckpointIO.Save(checkpointDir, step, model, options.Config, dataset.Tokenizer);
            }

            if (step % 50 == 0)
            {
                GC.Collect();
            }
        }

        var finalDir = Path.Combine(options.OutputDir, "checkpoints", "final");
        CheckpointIO.Save(finalDir, options.MaxSteps, model, options.Config, dataset.Tokenizer);

        var logPath = Path.Combine(options.OutputDir, "loss_log.json");
        File.WriteAllText(logPath, JsonSerializer.Serialize(log, JsonSettings.Options));
    }

    private static double Evaluate(Gpt model, TextDataset dataset, Loss<Tensor, Tensor, Tensor> lossFn, int blockSize, int batchSize, Device device)
    {
        model.eval();
        var losses = new List<double>();

        using (torch.no_grad())
        using (torch.NewDisposeScope())
        {
            for (var index = 0; index < 20; index++)
            {
                using var batch = BatchScope.Create(dataset.GetValidationBatch(blockSize, batchSize, device));
                using var logits = model.call(batch.X);
                using var loss = lossFn.call(logits.view(-1, model.Config.VocabSize), batch.Y.view(-1));
                losses.Add(loss.to(torch.CPU).item<float>());
            }
        }

        model.train();
        return losses.Average();
    }
}

public static class CheckpointIO
{
    public static void Save(string checkpointDir, int step, Gpt model, GptConfig config, CharacterTokenizer tokenizer)
    {
        Directory.CreateDirectory(checkpointDir);
        var metadata = new CheckpointMetadata
        {
            Step = step,
            Config = config,
            Vocabulary = tokenizer.Vocabulary
        };

        model.save(Path.Combine(checkpointDir, "model.bin"));
        File.WriteAllText(Path.Combine(checkpointDir, "metadata.json"), JsonSerializer.Serialize(metadata, JsonSettings.Options));
    }

    public static LoadedCheckpoint Load(string checkpointDir)
    {
        var metadataPath = Path.Combine(checkpointDir, "metadata.json");
        var modelPath = Path.Combine(checkpointDir, "model.bin");

        if (!File.Exists(metadataPath) || !File.Exists(modelPath))
        {
            throw new FileNotFoundException($"Checkpoint directory '{checkpointDir}' does not contain both metadata.json and model.bin.");
        }

        var metadata = JsonSerializer.Deserialize<CheckpointMetadata>(File.ReadAllText(metadataPath), JsonSettings.Options)
            ?? throw new InvalidOperationException("Checkpoint metadata could not be deserialized.");

        var tokenizer = CharacterTokenizer.FromVocabulary(metadata.Vocabulary);
        var model = new Gpt(metadata.Config);
        model.load(modelPath);
        model.to(DeviceSelector.GetDevice());
        model.eval();

        return new LoadedCheckpoint(model, tokenizer, metadata.Step);
    }
}

public sealed class LoadedCheckpoint : IDisposable
{
    public LoadedCheckpoint(Gpt model, CharacterTokenizer tokenizer, int step)
    {
        Model = model;
        Tokenizer = tokenizer;
        Step = step;
    }

    public Gpt Model { get; }

    public CharacterTokenizer Tokenizer { get; }

    public int Step { get; }

    public void Dispose()
    {
        Model.Dispose();
    }
}

public static class TextGenerator
{
    public static string Generate(Gpt model, CharacterTokenizer tokenizer, string prompt, int maxNewTokens, double temperature, int topK)
    {
        if (temperature <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(temperature), "Temperature must be greater than zero.");
        }

        var promptTokens = tokenizer.Encode(prompt);
        if (promptTokens.Length == 0)
        {
            throw new ArgumentException("Prompt does not contain any characters from the tokenizer vocabulary.", nameof(prompt));
        }

        using var seedTokens = tensor(promptTokens, dtype: ScalarType.Int64).unsqueeze(0).to(model.CurrentDevice);
        var idx = seedTokens;

        using (torch.no_grad())
        {
            model.eval();

            for (var tokenIndex = 0; tokenIndex < maxNewTokens; tokenIndex++)
            {
                var start = Math.Max(0, (int)idx.shape[1] - model.Config.BlockSize);
                using var idxCond = idx.narrow(1, start, idx.shape[1] - start);
                using var logits = model.call(idxCond);
                using var nextLogits = logits.select(1, logits.shape[1] - 1) / temperature;

                Tensor filteredLogits = nextLogits;
                Tensor? values = null;
                Tensor? threshold = null;
                if (topK > 0 && topK < model.Config.VocabSize)
                {
                    (values, _) = torch.topk(nextLogits, topK, dim: -1);
                    threshold = values.select(1, topK - 1).unsqueeze(1);
                    filteredLogits = nextLogits.masked_fill(nextLogits < threshold, double.NegativeInfinity);
                }

                using var probs = torch.softmax(filteredLogits, dim: -1);
                using var nextToken = torch.multinomial(probs, 1);
                var updated = torch.cat(new[] { idx, nextToken }, dim: 1);

                if (!ReferenceEquals(filteredLogits, nextLogits))
                {
                    filteredLogits.Dispose();
                    values?.Dispose();
                    threshold?.Dispose();
                }

                if (!ReferenceEquals(idx, seedTokens))
                {
                    idx.Dispose();
                }

                idx = updated;
            }
        }

        try
        {
            return tokenizer.Decode(idx.to(torch.CPU).data<long>().ToArray());
        }
        finally
        {
            if (!ReferenceEquals(idx, seedTokens))
            {
                idx.Dispose();
            }
        }
    }
}

public static class DeviceSelector
{
    public static Device GetDevice()
        => torch.cuda.is_available() ? torch.CUDA : torch.CPU;
}

public static class LearningRateSchedule
{
    public static double Get(int step, int warmupSteps, int maxSteps, double maxLr, double minLr)
    {
        if (step < warmupSteps)
        {
            return maxLr * (step + 1) / warmupSteps;
        }

        if (step >= maxSteps)
        {
            return minLr;
        }

        var progress = (step - warmupSteps) / (double)(maxSteps - warmupSteps);
        return minLr + 0.5 * (maxLr - minLr) * (1 + Math.Cos(Math.PI * progress));
    }
}

public sealed class BatchScope : IDisposable
{
    private BatchScope(Tensor x, Tensor y)
    {
        X = x;
        Y = y;
    }

    public Tensor X { get; }

    public Tensor Y { get; }

    public static BatchScope Create((Tensor X, Tensor Y) batch)
        => new(batch.X, batch.Y);

    public void Dispose()
    {
        X.Dispose();
        Y.Dispose();
    }
}

public static class JsonSettings
{
    public static JsonSerializerOptions Options { get; } = new()
    {
        WriteIndented = true
    };
}