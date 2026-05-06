namespace LlmFromScratch;

public sealed class GptConfig
{
    public int VocabSize { get; set; } = 65;
    public int BlockSize { get; set; } = 256;
    public int NLayer { get; set; } = 6;
    public int NHead { get; set; } = 6;
    public int NEmbd { get; set; } = 384;
}

public sealed class TrainOptions
{
    public required string DataPath { get; init; }
    public required string OutputDir { get; init; }
    public required string SamplePrompt { get; init; }
    public required GptConfig Config { get; init; }
    public int MaxSteps { get; init; }
    public int BatchSize { get; init; }
    public int ValidationEvery { get; init; }
    public int CheckpointEvery { get; init; }
    public int SampleEvery { get; init; }
    public double MaxLr { get; init; }
    public double MinLr { get; init; }
    public int WarmupSteps { get; init; }
    public int Seed { get; init; }
    public double SampleTemperature { get; init; }
    public int SampleTopK { get; init; }
    public bool AsciiOnly { get; init; }
    public int NumThreads { get; init; }
    public int GradAccumSteps { get; init; }
}

public sealed class CheckpointMetadata
{
    public required GptConfig Config { get; init; }
    public required string Vocabulary { get; init; }
    public int Step { get; init; }
}

public sealed class LossLog
{
    public List<int> Steps { get; set; } = new();
    public List<double> Train { get; set; } = new();
    public List<double> Val { get; set; } = new();
    public int GradAccumSteps { get; init; }
}