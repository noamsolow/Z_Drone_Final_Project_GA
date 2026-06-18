param(
    [string]$Config = ""
)

$studyRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $studyRoot "..\..\..")).Path
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
$scriptPath = Join-Path $studyRoot "merge_clean_and_recovery.py"

if (-not (Test-Path $pythonExe)) {
    throw "Project environment not found at $pythonExe"
}

if ([string]::IsNullOrWhiteSpace($Config)) {
    $Config = Join-Path $studyRoot "config.yaml"
}

& $pythonExe $scriptPath --config $Config
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
