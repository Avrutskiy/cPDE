# Run in an x64 Visual Studio developer PowerShell with CMake on PATH.
[CmdletBinding()]
param(
    [ValidateSet('cpu', 'cuda', 'both')][string]$Backend = 'cuda',
    [string]$BuildDirectory,
    [ValidateRange(1, 256)][int]$Jobs = 2,
    [string[]]$CMakeArgs = @()
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
if (-not $BuildDirectory) { $BuildDirectory = Join-Path $root "build-$Backend" }
$configure = @('-S', $root, '-B', $BuildDirectory, "-DCPDE_BACKEND=$Backend", '-DCMAKE_BUILD_TYPE=Release') + $CMakeArgs
& cmake @configure
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& cmake --build $BuildDirectory --config Release --parallel $Jobs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Built $Backend backend in $BuildDirectory (Visual Studio binaries are in Release)."
