param(
    [switch]$Quiet
)

$attemptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

$steps = @(
    Join-Path $attemptRoot "study01\run_extract_features.ps1",
    Join-Path $attemptRoot "study02\run_extract_fused_features.ps1",
    Join-Path $attemptRoot "study03\run_extract_noisy_aggregated_features.ps1"
)

foreach ($step in $steps) {
    Write-Host ""
    Write-Host "=== Running $step ==="
    if ($Quiet) {
        & $step -Quiet
    } else {
        & $step
    }
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Write-Host ""
Write-Host "Depth Pro dataset build complete."
