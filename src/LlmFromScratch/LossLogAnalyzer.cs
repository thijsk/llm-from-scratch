using System.Globalization;
using System.Text;
using System.Text.Json;

namespace LlmFromScratch;

public sealed class LossSummary
{
    public int TrainPoints { get; init; }
    public int ValidationPoints { get; init; }
    public double FinalTrainLoss { get; init; }
    public double BestValidationLoss { get; init; }
    public int BestValidationStep { get; init; }
    public double OverfittingGap { get; init; }
}

public static class LossLogAnalyzer
{
    public static LossSummary Analyze(string lossLogPath, string summaryOutPath, string csvOutPath)
    {
        if (!File.Exists(lossLogPath))
        {
            throw new FileNotFoundException($"Loss log not found: {lossLogPath}");
        }

        var log = JsonSerializer.Deserialize<LossLog>(File.ReadAllText(lossLogPath), JsonSettings.Options)
            ?? throw new InvalidOperationException("Could not deserialize loss log.");

        if (log.Steps.Count == 0 || log.Train.Count == 0)
        {
            throw new InvalidOperationException("Loss log is empty.");
        }

        var validationEvery = InferValidationInterval(log.Steps, log.Val.Count);
        var (bestValidationLoss, bestValidationStep) = FindBestValidation(log.Steps, log.Val, validationEvery);

        var summary = new LossSummary
        {
            TrainPoints = log.Train.Count,
            ValidationPoints = log.Val.Count,
            FinalTrainLoss = log.Train[^1],
            BestValidationLoss = bestValidationLoss,
            BestValidationStep = bestValidationStep,
            OverfittingGap = double.IsNaN(bestValidationLoss) ? double.NaN : log.Train[^1] - bestValidationLoss
        };

        var summaryDirectory = Path.GetDirectoryName(summaryOutPath);
        if (!string.IsNullOrWhiteSpace(summaryDirectory))
        {
            Directory.CreateDirectory(summaryDirectory);
        }

        var csvDirectory = Path.GetDirectoryName(csvOutPath);
        if (!string.IsNullOrWhiteSpace(csvDirectory))
        {
            Directory.CreateDirectory(csvDirectory);
        }

        File.WriteAllText(summaryOutPath, JsonSerializer.Serialize(summary, JsonSettings.Options));
        File.WriteAllText(csvOutPath, ToCsv(log, validationEvery));

        return summary;
    }

    private static int InferValidationInterval(IReadOnlyList<int> steps, int validationPoints)
    {
        if (validationPoints <= 1)
        {
            return 1;
        }

        var maxStep = steps[^1];
        return Math.Max(1, maxStep / (validationPoints - 1));
    }

    private static (double loss, int step) FindBestValidation(IReadOnlyList<int> steps, IReadOnlyList<double> validationLosses, int validationEvery)
    {
        if (validationLosses.Count == 0)
        {
            return (double.NaN, -1);
        }

        var bestLoss = double.PositiveInfinity;
        var bestStep = -1;

        for (var index = 0; index < validationLosses.Count; index++)
        {
            var loss = validationLosses[index];
            if (loss < bestLoss)
            {
                bestLoss = loss;
                bestStep = index * validationEvery;
            }
        }

        if (bestStep > steps[^1])
        {
            bestStep = steps[^1];
        }

        return (bestLoss, bestStep);
    }

    private static string ToCsv(LossLog log, int validationEvery)
    {
        var csv = new StringBuilder();
        csv.AppendLine("step,train_loss,val_loss");

        for (var index = 0; index < log.Steps.Count; index++)
        {
            var step = log.Steps[index];
            var trainLoss = log.Train[index];
            var valLoss = string.Empty;

            if (validationEvery > 0 && step % validationEvery == 0)
            {
                var validationIndex = step / validationEvery;
                if (validationIndex >= 0 && validationIndex < log.Val.Count)
                {
                    valLoss = log.Val[validationIndex].ToString(CultureInfo.InvariantCulture);
                }
            }

            csv.Append(step.ToString(CultureInfo.InvariantCulture));
            csv.Append(',');
            csv.Append(trainLoss.ToString(CultureInfo.InvariantCulture));
            csv.Append(',');
            csv.AppendLine(valLoss);
        }

        return csv.ToString();
    }
}