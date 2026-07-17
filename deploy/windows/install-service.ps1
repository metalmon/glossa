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
  Value for --allowed-host. Defaults to the IP/hostname from -Bind (e.g. 127.0.0.1:8080 → 127.0.0.1).
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

    [string] $AllowedHost
)

$ErrorActionPreference = "Stop"
$Repo = "metalmon/glossa"
$Target = "x86_64-pc-windows-msvc"
$Stem = "glossa-$Version-$Target"
$ZipName = "$Stem.zip"
$Url = "https://github.com/$Repo/releases/download/v$Version/$ZipName"
$ExtractDir = Join-Path $InstallDir $Stem
$KbExe = Join-Path $ExtractDir "kb.exe"

if (-not $AllowedHost) {
    if ($Bind -match '^([^:]+)') {
        $AllowedHost = $Matches[1]
    } else {
        $AllowedHost = "127.0.0.1"
    }
}

function Remove-GlossaService {
    param([string] $Name)
    $existing = Get-Service -Name $Name -ErrorAction SilentlyContinue
    if (-not $existing) { return }
    Write-Host "Stopping and removing existing service $Name ..."
    if ($existing.Status -ne "Stopped") {
        Stop-Service -Name $Name -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
    }
    & sc.exe delete $Name | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "sc.exe delete $Name failed (exit $LASTEXITCODE)" }
    Start-Sleep -Seconds 1
}

function Get-GlossaServiceDiagnostics {
    param([string] $Name)
    $query = & sc.exe query $Name 2>&1 | Out-String
    $config = & sc.exe qc $Name 2>&1 | Out-String
    @"
sc.exe query $Name :
$query
sc.exe qc $Name :
$config
"@
}

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

$BinaryPathName = "`"$KbExe`" mcp `"$CorpusPath`" --profile $Profile --transport streamable-http --bind $Bind --allowed-host $AllowedHost --windows-service --service-name $ServiceName"

Remove-GlossaService -Name $ServiceName

Write-Host "Creating service $ServiceName ..."
New-Service `
    -Name $ServiceName `
    -BinaryPathName $BinaryPathName `
    -DisplayName $ServiceName `
    -Description "glossa MCP ($Profile) on $CorpusPath" `
    -StartupType Automatic | Out-Null

Write-Host "Starting service ..."
try {
    Start-Service -Name $ServiceName -ErrorAction Stop
} catch {
    $diag = Get-GlossaServiceDiagnostics -Name $ServiceName
    throw @"
Start-Service failed: $($_.Exception.Message)

$diag
Check Event Viewer (Windows Logs → Application) and corpus permissions for LocalSystem.
"@
}

$deadline = (Get-Date).AddSeconds(30)
while ((Get-Service -Name $ServiceName).Status -ne "Running") {
    if ((Get-Date) -gt $deadline) {
        $diag = Get-GlossaServiceDiagnostics -Name $ServiceName
        throw "Service $ServiceName did not reach Running within 30s.`n`n$diag"
    }
    Start-Sleep -Milliseconds 500
}

Write-Host ""
Write-Host "Installed."
Write-Host "  Binary:   $KbExe"
Write-Host "  Corpus:   $CorpusPath"
Write-Host "  MCP URL:  http://$Bind/mcp"
Write-Host "  Health:   curl http://$Bind/health"
Write-Host ""
Write-Host "Connect agents: docs/connect-to-agents.md"
Write-Host "Stop:    Stop-Service $ServiceName"
Write-Host "Remove:  Stop-Service $ServiceName; sc.exe delete $ServiceName"
