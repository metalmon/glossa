# rtf2docx — reconvert an RTF to a clean .docx via Microsoft Word (COM).
#
# Standalone data-prep tool, kept OUT of the glossa core: some RTFs (e.g. GOSTs
# exported from other systems) reach us as .doc/.docx that lost their table and
# section structure, and pure-Rust RTF crates only give flat text without tables.
# Word re-authors the RTF faithfully — tables and styles intact — so glossa's
# office_oxide extractor sees real tables again.
#
# Usage:   pwsh tools/rtf2docx/rtf2docx.ps1 <input.rtf> [output.docx]
#          (output defaults to the input path with a .docx extension)
param(
    [Parameter(Mandatory = $true)][string]$Source,
    [string]$Output
)
$ErrorActionPreference = 'Stop'
$src = (Resolve-Path $Source).Path
if (-not $Output) { $Output = [System.IO.Path]::ChangeExtension($src, '.docx') }
$out = [System.IO.Path]::GetFullPath($Output)

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0   # wdAlertsNone
try {
    $doc = $word.Documents.Open($src, $false, $true)   # ConfirmConversions=$false, ReadOnly=$true
    $doc.SaveAs2($out, 16)   # 16 = wdFormatDocumentDefault (.docx)
    $doc.Close($false)
    Write-Output "rtf2docx: $src -> $out"
} finally {
    $word.Quit()
}
