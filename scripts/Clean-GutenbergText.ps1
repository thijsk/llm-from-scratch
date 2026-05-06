param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath,

    [Parameter(Mandatory = $true)]
    [string]$OutputPath,

    [string]$StartAfterPattern,

    [string]$EndBeforePattern,

    [switch]$RemoveNotesSections,

    [switch]$AsciiOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-TextSlice {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Text,

        [Parameter(Mandatory = $true)]
        [string]$Pattern,

        [ValidateSet('Start', 'End')]
        [string]$Mode
    )

    $match = [regex]::Match($Text, $Pattern, [System.Text.RegularExpressions.RegexOptions]::IgnoreCase -bor [System.Text.RegularExpressions.RegexOptions]::Multiline)
    if (-not $match.Success) {
        return $Text
    }

    if ($Mode -eq 'Start') {
        return $Text.Substring($match.Index + $match.Length)
    }

    return $Text.Substring(0, $match.Index)
}

function Convert-ToAscii {
    param([Parameter(Mandatory = $true)][string]$Text)

    $normalized = $Text
    $normalized = $normalized.Replace([char]0x2018, "'")
    $normalized = $normalized.Replace([char]0x2019, "'")
    $normalized = $normalized.Replace([char]0x201C, '"')
    $normalized = $normalized.Replace([char]0x201D, '"')
    $normalized = $normalized.Replace([char]0x2013, '-')
    $normalized = $normalized.Replace([char]0x2014, '-')
    $normalized = $normalized.Replace([char]0x2026, '...')
    $normalized = $normalized.Replace([char]0x00A0, ' ')

    $builder = [System.Text.StringBuilder]::new($normalized.Length)
    foreach ($character in $normalized.ToCharArray()) {
        if (($character -ge ' ' -and $character -le '~') -or $character -eq "`r" -or $character -eq "`n" -or $character -eq "`t") {
            [void]$builder.Append($character)
        }
    }

    return $builder.ToString()
}

function Remove-NoteSections {
    param([Parameter(Mandatory = $true)][string]$Text)

    $noteHeadingPattern = "^(FOOTNOTES|LINENOTES|TRANSCRIBER'S NOTES|NOTES)\s*:?\s*$"
    $sectionStartPattern = "^[A-Z][A-Z0-9 ,\.\-'_\(\)\[\]&;:]{3,}$"

    $lines = $Text -split "`n"
    $output = [System.Collections.Generic.List[string]]::new($lines.Length)
    $skip = $false

    foreach ($line in $lines) {
        $trimmed = $line.Trim()

        if (-not $skip -and $trimmed -match $noteHeadingPattern) {
            $skip = $true
            continue
        }

        if ($skip) {
            if ($trimmed -match $noteHeadingPattern) {
                continue
            }

            if ($trimmed -cmatch $sectionStartPattern) {
                $skip = $false
                [void]$output.Add($line)
            }

            continue
        }

        [void]$output.Add($line)
    }

    return [string]::Join("`n", $output)
}

$resolvedInputPath = (Resolve-Path -LiteralPath $InputPath).Path
$outputDirectory = Split-Path -Parent $OutputPath
if (-not [string]::IsNullOrWhiteSpace($outputDirectory)) {
    New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
}

$text = [System.IO.File]::ReadAllText($resolvedInputPath)
$text = $text.TrimStart([char]0xFEFF)
$text = $text -replace "`r`n", "`n"
$text = $text -replace "`r", "`n"

$startPatterns = @(
    '^\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*$',
    '^START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*$'
)

foreach ($pattern in $startPatterns) {
    $updated = Get-TextSlice -Text $text -Pattern $pattern -Mode Start
    if ($updated.Length -ne $text.Length) {
        $text = $updated
        break
    }
}

$endPatterns = @(
    '^\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*$',
    '^END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*$'
)

foreach ($pattern in $endPatterns) {
    $updated = Get-TextSlice -Text $text -Pattern $pattern -Mode End
    if ($updated.Length -ne $text.Length) {
        $text = $updated
        break
    }
}

if (-not [string]::IsNullOrWhiteSpace($StartAfterPattern)) {
    $text = Get-TextSlice -Text $text -Pattern $StartAfterPattern -Mode Start
}

if (-not [string]::IsNullOrWhiteSpace($EndBeforePattern)) {
    $text = Get-TextSlice -Text $text -Pattern $EndBeforePattern -Mode End
}

if ($RemoveNotesSections) {
    $text = Remove-NoteSections -Text $text
    $text = [regex]::Replace($text, '(?m)^\s*\[\d+(?::\d+)?(?:-\d+)?\].*$\n?', '')
    $text = [regex]::Replace($text, '(?m)^\s*Title\].*$\n?', '')
    $text = [regex]::Replace($text, '(?m)^.*\bMS\..*$\n?', '')
}

$lines = $text -split "`n"
$trimmedLines = foreach ($line in $lines) {
    $normalizedLine = $line.TrimEnd()
    if ($normalizedLine -match '^[\p{Zs}\t]*\d{1,2}[\p{Zs}\t]*$') {
        ''
    }
    else {
        $normalizedLine
    }
}

$text = [string]::Join("`n", $trimmedLines)
$text = [regex]::Replace($text, '(?m)(.+?)[\t ]{2,}\d{1,5}$', '$1')
$text = $text.Trim()
$text = [regex]::Replace($text, "`n{3,}", "`n`n")

if ($AsciiOnly) {
    $text = Convert-ToAscii -Text $text
    $text = [regex]::Replace($text, "`n{3,}", "`n`n")
    $text = $text.Trim()
}

$text = $text -replace "`n", "`r`n"

$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllText($OutputPath, $text, $utf8NoBom)

$inputChars = [System.IO.File]::ReadAllText($resolvedInputPath).Length
$outputChars = $text.Length
Write-Host "Cleaned Gutenberg text written to $OutputPath"
Write-Host ("Characters: {0:N0} -> {1:N0}" -f $inputChars, $outputChars)
if ($AsciiOnly) {
    Write-Host "Mode: ASCII-only"
}
if ($RemoveNotesSections) {
    Write-Host "Removed sections: FOOTNOTES/LINENOTES/NOTES"
}
if (-not [string]::IsNullOrWhiteSpace($StartAfterPattern)) {
    Write-Host "Trimmed after pattern: $StartAfterPattern"
}
if (-not [string]::IsNullOrWhiteSpace($EndBeforePattern)) {
    Write-Host "Trimmed before pattern: $EndBeforePattern"
}