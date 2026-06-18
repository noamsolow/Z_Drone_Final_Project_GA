param(
    [string]$Config = "",
    [switch]$Quiet
)

$studyRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $studyRoot "..\..\..")).Path
$pythonExe = Join-Path $repoRoot ".venv-depthpro\Scripts\python.exe"
$scriptPath = Join-Path $studyRoot "run_extract_features.py"

if (-not (Test-Path $pythonExe)) {
    throw "Depth Pro environment not found at $pythonExe"
}

if ([string]::IsNullOrWhiteSpace($Config)) {
    $Config = Join-Path $studyRoot "config.yaml"
}

$arguments = @($scriptPath, "--config", $Config)
if ($Quiet) {
    $arguments += "--quiet"
}

& $pythonExe @arguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
