#Requires -Version 5.1
<#
.SYNOPSIS
  Start glossa MCP (streamable-http) against a KB and print a ready-to-paste Cursor mcpServers JSON.

.EXAMPLE
  .\scripts\start-mcp-http.ps1
  .\scripts\start-mcp-http.ps1 -KbRoot C:\path\to\my-corpus -Bind 127.0.0.1:8080 -Profile editor
#>
[CmdletBinding()]
param(
    [string]$KbRoot = "",
    [string]$Bind = "127.0.0.1:8080",
    [ValidateSet("reader", "editor", "full")]
    [string]$Profile = "editor",
    [string]$KbExe = "",
    [string]$ServerName = "glossa-kb",
    [string[]]$AllowedHost = @("localhost", "127.0.0.1")
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $KbRoot) {
    $KbRoot = Join-Path $repoRoot "kb-docs"
}
$KbRoot = (Resolve-Path -LiteralPath $KbRoot).Path

if (-not $KbExe) {
    $candidates = @(
        (Join-Path $repoRoot "target\release\kb.exe"),
        (Join-Path $repoRoot "target\debug\kb.exe")
    )
    foreach ($c in $candidates) {
        if (Test-Path -LiteralPath $c) { $KbExe = $c; break }
    }
}
if (-not $KbExe) {
    $cmd = Get-Command kb.exe -ErrorAction SilentlyContinue
    if ($cmd) { $KbExe = $cmd.Source }
}
if (-not $KbExe -or -not (Test-Path -LiteralPath $KbExe)) {
    throw "kb.exe not found. Build release (`cargo build --release -p glossa --bin kb`) or pass -KbExe."
}
$KbExe = (Resolve-Path -LiteralPath $KbExe).Path

# http://host:port/mcp from --bind
if ($Bind -match '^\[.+\]:') {
    $mcpUrl = "http://${Bind}/mcp"
} elseif ($Bind -match '^\d+\.\d+\.\d+\.\d+:\d+$' -or $Bind -match '^localhost:\d+$') {
    $mcpUrl = "http://${Bind}/mcp"
} elseif ($Bind -match '^[^:]+:\d+$') {
    $mcpUrl = "http://${Bind}/mcp"
} else {
    $mcpUrl = "http://${Bind}/mcp"
}

$hostArgs = @()
foreach ($h in $AllowedHost) {
    $hostArgs += @("--allowed-host", $h)
}

$mcpJson = [ordered]@{
    mcpServers = [ordered]@{
        $ServerName = [ordered]@{
            url = $mcpUrl
        }
    }
} | ConvertTo-Json -Depth 5

Write-Host ""
Write-Host "KB:      $KbRoot"
Write-Host "kb.exe:  $KbExe"
Write-Host "bind:    $Bind"
Write-Host "profile: $Profile"
Write-Host "endpoint:$mcpUrl"
Write-Host ""
Write-Host "======== copy-paste into Cursor MCP config (.cursor/mcp.json or Settings → MCP) ========" -ForegroundColor Cyan
Write-Host $mcpJson
Write-Host "======================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Health:  http://$Bind/health"
Write-Host "Starting MCP (Ctrl+C to stop)..."
Write-Host ""

& $KbExe mcp $KbRoot `
    --profile $Profile `
    --transport streamable-http `
    --bind $Bind `
    @hostArgs
