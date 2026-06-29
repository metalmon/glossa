#Requires -RunAsAdministrator
<#
.SYNOPSIS
  Install glossa kb from a GitHub Release and register a Windows Service (MCP streamable-http).

.PARAMETER Version
  Release version without 'v' (e.g. 0.1.0).

.PARAMETER CorpusPath
  Folder containing documents; .glossa will be created here.

.PARAMETER Profile
  MCP profile: reader, editor, or full.

.PARAMETER Bind
  HTTP bind address (default 127.0.0.1:8080).

.PARAMETER InstallDir
  Root install directory (default C:\Program Files\glossa).

.PARAMETER ServiceName
  Windows service name (default glossa-mcp).

.PARAMETER AllowedHost
  Value for --allowed-host (default localhost).
#>
param(
    [Parameter(Mandatory = $true)]
    [string] $Version,

    [Parameter(Mandatory = $true)]
    [string] $CorpusPath,

    [ValidateSet("reader", "editor", "full")]
    [string] $Profile = "reader",

    [string] $Bind = "127.0.0.1:8080",

    [string] $InstallDir = "$env:ProgramFiles\glossa",

    [string] $ServiceName = "glossa-mcp",

    [string] $AllowedHost = "localhost"
)

$ErrorActionPreference = "Stop"
$Repo = "metalmon/glossa"
$Target = "x86_64-pc-windows-msvc"
$Stem = "glossa-$Version-$Target"
$ZipName = "$Stem.zip"
$Url = "https://github.com/$Repo/releases/download/v$Version/$ZipName"
$ExtractDir = Join-Path $InstallDir $Stem
$KbExe = Join-Path $ExtractDir "kb.exe"

Write-Host "Downloading $Url ..."
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
$ZipPath = Join-Path $env:TEMP $ZipName
Invoke-WebRequest -Uri $Url -OutFile $ZipPath -UseBasicParsing

Write-Host "Extracting to $ExtractDir ..."
if (Test-Path $ExtractDir) { Remove-Item -Recurse -Force $ExtractDir }
Expand-Archive -Path $ZipPath -DestinationPath $InstallDir -Force

if (-not (Test-Path $KbExe)) {
    throw "kb.exe not found at $KbExe"
}

Write-Host "Creating corpus directory $CorpusPath ..."
New-Item -ItemType Directory -Force -Path $CorpusPath | Out-Null

$Manifest = Join-Path $CorpusPath ".glossa\manifest.json"
if (-not (Test-Path $Manifest)) {
    Write-Host "Running initial index ..."
    & $KbExe index $CorpusPath
    if ($LASTEXITCODE -ne 0) { throw "kb index failed with exit code $LASTEXITCODE" }
}

# sc.exe requires a space after binPath= and start=
$BinPath = "`"$KbExe`" mcp `"$CorpusPath`" --profile $Profile --transport streamable-http --bind $Bind --allowed-host $AllowedHost --windows-service"

$existing = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Stopping and removing existing service $ServiceName ..."
    sc.exe stop $ServiceName | Out-Null
    Start-Sleep -Seconds 2
    sc.exe delete $ServiceName | Out-Null
    Start-Sleep -Seconds 1
}

Write-Host "Creating service $ServiceName ..."
sc.exe create $ServiceName binPath= $BinPath start= auto | Out-Null
if ($LASTEXITCODE -ne 0) { throw "sc.exe create failed" }
sc.exe description $ServiceName "glossa MCP ($Profile) on $CorpusPath" | Out-Null

Write-Host "Starting service ..."
sc.exe start $ServiceName | Out-Null
if ($LASTEXITCODE -ne 0) { throw "sc.exe start failed — check Event Viewer and corpus permissions" }

Write-Host ""
Write-Host "Installed."
Write-Host "  Binary:   $KbExe"
Write-Host "  Corpus:   $CorpusPath"
Write-Host "  MCP URL:  http://$Bind/mcp"
Write-Host "  Health:   curl http://$Bind/health"
Write-Host ""
Write-Host "Connect agents: docs/connect-to-agents.md"
Write-Host "Stop:    sc.exe stop $ServiceName"
Write-Host "Remove:  sc.exe stop $ServiceName; sc.exe delete $ServiceName"
