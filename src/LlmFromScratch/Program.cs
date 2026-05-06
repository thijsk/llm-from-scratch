using System.Globalization;
using TorchSharp;

namespace LlmFromScratch;

internal static class Program
{
	private static int Main(string[] args)
	{
		if (args.Length == 0)
		{
			PrintHelp();
			return 1;
		}

		try
		{
			var command = args[0].ToLowerInvariant();
			var options = CommandLine.Parse(args.Skip(1).ToArray());

			return command switch
			{
				"train" => RunTrain(options),
				"generate" => RunGenerate(options),
				"analyze" => RunAnalyze(options),
				"help" or "--help" or "-h" => PrintHelpAndExit(),
				_ => UnknownCommand(command)
			};
		}
		catch (Exception exception)
		{
			Console.Error.WriteLine(exception.Message);
			return 1;
		}
	}

	private static int RunTrain(Dictionary<string, string> options)
	{
		var trainOptions = new TrainOptions
		{
			DataPath = CommandLine.GetString(options, "data", PathResolver.GetDefaultDataPath()),
			OutputDir = CommandLine.GetString(options, "output-dir", "artifacts"),
			SamplePrompt = CommandLine.GetString(options, "sample-prompt", "To be or not"),
			MaxSteps = CommandLine.GetInt(options, "max-steps", 5000),
			BatchSize = CommandLine.GetInt(options, "batch-size", 64),
			ValidationEvery = CommandLine.GetInt(options, "validation-every", 100),
			CheckpointEvery = CommandLine.GetInt(options, "checkpoint-every", 1000),
			SampleEvery = CommandLine.GetInt(options, "sample-every", 100),
			MaxLr = CommandLine.GetDouble(options, "max-lr", 1e-3),
			MinLr = CommandLine.GetDouble(options, "min-lr", 1e-4),
			WarmupSteps = CommandLine.GetInt(options, "warmup-steps", 100),
			Seed = CommandLine.GetInt(options, "seed", 42),
			SampleTemperature = CommandLine.GetDouble(options, "temperature", 0.8),
			SampleTopK = CommandLine.GetInt(options, "top-k", 40),
			AsciiOnly = CommandLine.GetBool(options, "ascii-only", false),
			NumThreads = CommandLine.GetInt(options, "num-threads", 0),
			GradAccumSteps = CommandLine.GetInt(options, "grad-accum-steps", 1),
			Config = new GptConfig
			{
				BlockSize = CommandLine.GetInt(options, "block-size", 256),
				NLayer = CommandLine.GetInt(options, "n-layer", 6),
				NHead = CommandLine.GetInt(options, "n-head", 6),
				NEmbd = CommandLine.GetInt(options, "n-embd", 384)
			}
		};

		Trainer.Train(trainOptions);
		return 0;
	}

	private static int RunGenerate(Dictionary<string, string> options)
	{
		var checkpointDir = CommandLine.GetRequired(options, "checkpoint");
		var prompt = CommandLine.GetString(options, "prompt", "To be or not");
		var maxNewTokens = CommandLine.GetInt(options, "max-new-tokens", 200);
		var temperature = CommandLine.GetDouble(options, "temperature", 0.8);
		var topK = CommandLine.GetInt(options, "top-k", 40);
		var seed = CommandLine.GetInt(options, "seed", 42);

		torch.random.manual_seed(seed);

		using var loaded = CheckpointIO.Load(checkpointDir);
		var output = TextGenerator.Generate(loaded.Model, loaded.Tokenizer, prompt, maxNewTokens, temperature, topK);
		Console.WriteLine(output);
		return 0;
	}

	private static int RunAnalyze(Dictionary<string, string> options)
	{
		var lossLogPath = CommandLine.GetString(options, "loss-log", Path.Combine("artifacts", "loss_log.json"));
		var summaryOutPath = CommandLine.GetString(options, "summary-out", Path.Combine("artifacts", "loss_summary.json"));
		var csvOutPath = CommandLine.GetString(options, "csv-out", Path.Combine("artifacts", "loss_log.csv"));

		var summary = LossLogAnalyzer.Analyze(lossLogPath, summaryOutPath, csvOutPath);
		Console.WriteLine($"Loss log: {lossLogPath}");
		Console.WriteLine($"CSV export: {csvOutPath}");
		Console.WriteLine($"Summary: {summaryOutPath}");
		Console.WriteLine($"Train points: {summary.TrainPoints}");
		Console.WriteLine($"Validation points: {summary.ValidationPoints}");
		Console.WriteLine($"Best validation: {summary.BestValidationLoss:F4} at step {summary.BestValidationStep}");
		Console.WriteLine($"Final train: {summary.FinalTrainLoss:F4}");
		Console.WriteLine($"Overfitting estimate: {summary.OverfittingGap:F4}");
		return 0;
	}

	private static int UnknownCommand(string command)
	{
		Console.Error.WriteLine($"Unknown command '{command}'.");
		PrintHelp();
		return 1;
	}

	private static int PrintHelpAndExit()
	{
		PrintHelp();
		return 0;
	}

	private static void PrintHelp()
	{
		Console.WriteLine("LlmFromScratch (.NET + TorchSharp)");
		Console.WriteLine();
		Console.WriteLine("Commands:");
		Console.WriteLine("  train     Train a character-level GPT on the mirrored dataset.");
		Console.WriteLine("  generate  Generate text from a saved checkpoint directory.");
		Console.WriteLine("  analyze   Summarize loss_log.json and export CSV/summary JSON.");
		Console.WriteLine();
		Console.WriteLine("Train options:");
		Console.WriteLine("  --ascii-only true|false  Remove non-ASCII characters before tokenization (default: false)");
		Console.WriteLine("  --num-threads N          Number of CPU threads for LibTorch (0 = let LibTorch decide, default: 0)");
		Console.WriteLine();
		Console.WriteLine("Examples:");
		Console.WriteLine("  dotnet run -- train --output-dir artifacts");
		Console.WriteLine("  dotnet run -- train --output-dir artifacts-eng --data data/english.txt --ascii-only true");
		Console.WriteLine("  dotnet run -- generate --checkpoint artifacts/checkpoints/final --prompt \"To be or not\"");
		Console.WriteLine("  dotnet run -- analyze --loss-log artifacts/loss_log.json");
	}
}

internal static class PathResolver
{
	public static string GetDefaultDataPath()
	{
		var candidates = new[]
		{
			Path.Combine(Environment.CurrentDirectory, "data", "shakespeare.txt"),
			Path.Combine(Environment.CurrentDirectory, "..", "data", "shakespeare.txt"),
			Path.Combine(Environment.CurrentDirectory, "..", "..", "data", "shakespeare.txt"),
			Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "data", "shakespeare.txt"),
			Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "data", "shakespeare.txt")
		};

		foreach (var candidate in candidates.Select(Path.GetFullPath))
		{
			if (File.Exists(candidate))
			{
				return candidate;
			}
		}

		return Path.GetFullPath(candidates[0]);
	}
}

internal static class CommandLine
{
	public static Dictionary<string, string> Parse(string[] args)
	{
		var result = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

		for (var index = 0; index < args.Length; index++)
		{
			var token = args[index];
			if (!token.StartsWith("--", StringComparison.Ordinal))
			{
				throw new ArgumentException($"Expected an option starting with '--', found '{token}'.");
			}

			var key = token[2..];
			if (index + 1 >= args.Length || args[index + 1].StartsWith("--", StringComparison.Ordinal))
			{
				result[key] = "true";
				continue;
			}

			result[key] = args[++index];
		}

		return result;
	}

	public static string GetRequired(Dictionary<string, string> options, string key)
	{
		if (!options.TryGetValue(key, out var value) || string.IsNullOrWhiteSpace(value))
		{
			throw new ArgumentException($"Missing required option '--{key}'.");
		}

		return value;
	}

	public static string GetString(Dictionary<string, string> options, string key, string defaultValue)
		=> options.TryGetValue(key, out var value) ? value : defaultValue;

	public static int GetInt(Dictionary<string, string> options, string key, int defaultValue)
		=> options.TryGetValue(key, out var value)
			? int.Parse(value, CultureInfo.InvariantCulture)
			: defaultValue;

	public static double GetDouble(Dictionary<string, string> options, string key, double defaultValue)
		=> options.TryGetValue(key, out var value)
			? double.Parse(value, CultureInfo.InvariantCulture)
			: defaultValue;

	public static bool GetBool(Dictionary<string, string> options, string key, bool defaultValue)
	{
		if (!options.TryGetValue(key, out var value))
		{
			return defaultValue;
		}

		if (bool.TryParse(value, out var parsed))
		{
			return parsed;
		}

		if (value == "1")
		{
			return true;
		}

		if (value == "0")
		{
			return false;
		}

		throw new ArgumentException($"Option '--{key}' must be a boolean (true/false). Received '{value}'.");
	}
}
