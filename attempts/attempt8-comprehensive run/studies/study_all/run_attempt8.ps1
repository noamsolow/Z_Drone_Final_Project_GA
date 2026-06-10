param(
    [string]$PythonExe = ".\.venv\Scripts\python.exe"
)

$scriptPath = "attempts/attempt8-comprehensive run/studies/study_all/run_attempt8.py"
$configPath = "attempts/attempt8-comprehensive run/studies/study_all/config.yaml"

Write-Host "[attempt8] Starting comprehensive rerun"
Write-Host "[attempt8] Python: $PythonExe"
Write-Host "[attempt8] Script: $scriptPath"
Write-Host "[attempt8] Config: $configPath"

& $PythonExe $scriptPath --config $configPath
$exitCode = $LASTEXITCODE

if ($exitCode -ne 0) {
    Write-Host "[attempt8] Run failed with exit code $exitCode"
    exit $exitCode
}

Write-Host "[attempt8] Run completed successfully"
