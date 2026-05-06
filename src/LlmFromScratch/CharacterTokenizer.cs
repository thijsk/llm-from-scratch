namespace LlmFromScratch;

public sealed class CharacterTokenizer
{
    private readonly Dictionary<char, long> _stoi;
    private readonly Dictionary<long, char> _itos;

    private CharacterTokenizer(string vocabulary)
    {
        Vocabulary = vocabulary;
        _stoi = vocabulary.Select((character, index) => (character, index))
            .ToDictionary(entry => entry.character, entry => (long)entry.index);
        _itos = vocabulary.Select((character, index) => (character, index))
            .ToDictionary(entry => (long)entry.index, entry => entry.character);
    }

    public string Vocabulary { get; }

    public int VocabSize => Vocabulary.Length;

    public static CharacterTokenizer FromText(string text)
    {
        var vocabulary = new string(text.ToHashSet().OrderBy(character => character).ToArray());
        return new CharacterTokenizer(vocabulary);
    }

    public static CharacterTokenizer FromVocabulary(string vocabulary)
        => new(vocabulary);

    public long[] Encode(string text)
        => text.Where(_stoi.ContainsKey).Select(character => _stoi[character]).ToArray();

    public string Decode(IEnumerable<long> tokens)
        => new(tokens.Select(token => _itos[token]).ToArray());
}