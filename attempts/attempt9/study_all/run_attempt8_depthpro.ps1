param(
    [string]$Config = ""
)

$studyRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $studyRoot "..\..\..")).Path
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
$attempt8Script = Join-Path $repoRoot "attempts\attempt8-comprehensive run\studies\study_all\run_attempt8.py"
$posterPlotScript = Join-Path $repoRoot "attempts\attempt8-comprehensive run\studies\study_all\create_attempt8_poster_plots.py"
$analysisPlotScript = Join-Path $repoRoot "attempts\attempt8-comprehensive run\studies\study_all\create_attempt8_analysis.py"

if (-not (Test-Path $pythonExe)) {
    throw "Project environment not found at $pythonExe"
}

if ([string]::IsNullOrWhiteSpace($Config)) {
    $mergedConfig = Join-Path $studyRoot "config_merged.yaml"
    if (Test-Path $mergedConfig) {
        $Config = $mergedConfig
    } else {
        $Config = Join-Path $studyRoot "config.yaml"
    }
}

$Config = (Resolve-Path $Config).Path

& $pythonExe $attempt8Script --config $Config
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$outputRoot = (& $pythonExe -c "import pathlib, sys, yaml; repo = pathlib.Path(sys.argv[1]); config_path = pathlib.Path(sys.argv[2]); payload = yaml.safe_load(config_path.read_text(encoding='utf-8')); output_root = pathlib.Path(payload['output_root']); print((output_root if output_root.is_absolute() else repo / output_root).resolve())" $repoRoot $Config).Trim()
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "[attempt9/study_all] Creating Attempt 8 poster-style SVG plots in $outputRoot\plots"
& $pythonExe $posterPlotScript --artifacts-root $outputRoot
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "[attempt9/study_all] Creating Attempt 8 analysis-style SVG plots in $outputRoot\plots"
& $pythonExe $analysisPlotScript --artifacts-root $outputRoot --plot-prefix "attempt9_depthpro" --run-label "Attempt 9 Depth Pro"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $pythonExe -c "import json, pathlib, sys; root = pathlib.Path(sys.argv[1]); reports = root / 'reports'; plots_dir = root / 'plots'; figures_dir = reports / 'figures'; svg_plots = sorted(str(path) for path in plots_dir.glob('*.svg')); png_figures = sorted(str(path) for path in figures_dir.glob('*.png')); markdown_reports = sorted(str(path) for path in reports.glob('*.md')); payload = {'plots_dir': str(plots_dir), 'figures_dir': str(figures_dir), 'svg_plot_count': len(svg_plots), 'png_figure_count': len(png_figures), 'markdown_report_count': len(markdown_reports), 'svg_plots': svg_plots, 'png_figures': png_figures, 'markdown_reports': markdown_reports}; out = reports / 'all_plot_manifest.json'; out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8'); print('[attempt9/study_all] all_plot_manifest_json: {}'.format(out)); print('[attempt9/study_all] svg_plot_count: {}'.format(len(svg_plots))); print('[attempt9/study_all] png_figure_count: {}'.format(len(png_figures))); print('[attempt9/study_all] markdown_report_count: {}'.format(len(markdown_reports)))" $outputRoot
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
