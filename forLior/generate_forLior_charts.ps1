$ErrorActionPreference = "Stop"

$OutRoot = $PSScriptRoot
$RepoRoot = Split-Path -Parent $OutRoot

$Dirs = @{
    Attempt9Final = Join-Path $OutRoot "existing_attempt9_final_depthpro"
    Attempt8Synthetic = Join-Path $OutRoot "existing_attempt8_synthetic"
    Attempt8Real = Join-Path $OutRoot "existing_attempt8_real_calibration"
    Generated = Join-Path $OutRoot "generated_summary_charts"
    Data = Join-Path $OutRoot "source_data"
}

foreach ($dir in $Dirs.Values) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

$Manifest = New-Object System.Collections.Generic.List[object]

function Add-Manifest {
    param(
        [string]$Category,
        [string]$File,
        [string]$Source,
        [string]$Notes
    )
    $script:Manifest.Add([pscustomobject]@{
        category = $Category
        file = $File
        source = $Source
        notes = $Notes
    })
}

function Copy-PlotFiles {
    param(
        [string]$SourceDir,
        [string]$DestDir,
        [string]$Category,
        [string]$Notes
    )
    if (-not (Test-Path -LiteralPath $SourceDir)) { return }
    Get-ChildItem -LiteralPath $SourceDir -File |
        Where-Object { $_.Extension -in @(".png", ".svg", ".jpg", ".jpeg") } |
        ForEach-Object {
            $dest = Join-Path $DestDir $_.Name
            Copy-Item -LiteralPath $_.FullName -Destination $dest -Force
            Add-Manifest $Category $dest $_.FullName $Notes
        }
}

function Copy-DataFile {
    param([string]$SourcePath, [string]$DestName)
    if (-not (Test-Path -LiteralPath $SourcePath)) { return }
    $dest = Join-Path $Dirs.Data $DestName
    Copy-Item -LiteralPath $SourcePath -Destination $dest -Force
    Add-Manifest "source_data" $dest $SourcePath "CSV/JSON/MD source used for generated charts"
}

$Attempt9 = Join-Path $RepoRoot "attempts\attempt9\study_all\artifacts_merged_complete"
$Attempt8All = Join-Path $RepoRoot "attempts\attempt8-comprehensive run\studies\study_all\artifacts"
$Attempt8Real = Join-Path $RepoRoot "attempts\attempt8-comprehensive run\studies\study02\artifacts"
$Attempt8RealStudy01 = Join-Path $RepoRoot "attempts\attempt8-comprehensive run\studies\study01\artifacts"

Copy-PlotFiles (Join-Path $Attempt9 "plots") $Dirs.Attempt9Final "existing_attempt9_final_depthpro" "Copied final synthetic DepthPro plots from attempt9"
Copy-PlotFiles (Join-Path $Attempt9 "reports\figures") $Dirs.Attempt9Final "existing_attempt9_final_depthpro" "Copied final synthetic PNG figures from attempt9"
Copy-PlotFiles (Join-Path $Attempt8All "plots") $Dirs.Attempt8Synthetic "existing_attempt8_synthetic" "Copied synthetic result plots from attempt8"
Copy-PlotFiles (Join-Path $Attempt8Real "plots") $Dirs.Attempt8Real "existing_attempt8_real_calibration" "Copied real calibration plots from attempt8 study02"
Copy-PlotFiles (Join-Path $Attempt8RealStudy01 "plots") $Dirs.Attempt8Real "existing_attempt8_real_calibration" "Copied real uncalibrated plots from attempt8 study01"

Copy-DataFile (Join-Path $Attempt9 "reports\model_metrics.csv") "attempt9_model_metrics.csv"
Copy-DataFile (Join-Path $Attempt9 "reports\distance_range_metrics.csv") "attempt9_distance_range_metrics.csv"
Copy-DataFile (Join-Path $Attempt9 "reports\predictions.csv") "attempt9_predictions.csv"
Copy-DataFile (Join-Path $Attempt9 "reports\poster_model_table.csv") "attempt9_poster_model_table.csv"
Copy-DataFile (Join-Path $Attempt8All "reports\final_ensemble_weather_time_mae.csv") "attempt8_final_ensemble_weather_time_mae.csv"
Copy-DataFile (Join-Path $Attempt8Real "reports\before_after_by_distance.csv") "attempt8_real_before_after_by_distance.csv"
Copy-DataFile (Join-Path $Attempt8Real "reports\before_after_by_drone_type.csv") "attempt8_real_before_after_by_drone_type.csv"
Copy-DataFile (Join-Path $Attempt8Real "reports\raw_uncalibrated_metrics.json") "attempt8_real_raw_uncalibrated_metrics.json"
Copy-DataFile (Join-Path $Attempt8Real "reports\final_calibrated_metrics.json") "attempt8_real_final_calibrated_metrics.json"

Add-Type -AssemblyName System.Drawing

$FontTitle = New-Object System.Drawing.Font("Times New Roman", 26, [System.Drawing.FontStyle]::Bold)
$FontAxis = New-Object System.Drawing.Font("Times New Roman", 18, [System.Drawing.FontStyle]::Regular)
$FontTick = New-Object System.Drawing.Font("Times New Roman", 15, [System.Drawing.FontStyle]::Regular)
$FontSmall = New-Object System.Drawing.Font("Times New Roman", 14, [System.Drawing.FontStyle]::Regular)
$FontValue = New-Object System.Drawing.Font("Times New Roman", 14, [System.Drawing.FontStyle]::Bold)
$Black = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(20,20,20))
$White = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::White)
$Dark = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(55,55,55))
$Mid = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(145,145,145))
$Light = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(230,230,230))
$GridPen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(220,220,220)), 1
$AxisPen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(25,25,25)), 2
$LinePen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(25,25,25)), 3
$GrayLinePen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(100,100,100)), 3
$DashedPen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(25,25,25)), 3
$DashedPen.DashStyle = [System.Drawing.Drawing2D.DashStyle]::Dash

function Save-Bitmap {
    param($Bitmap, $Graphics, [string]$Path, [string]$Category, [string]$Source, [string]$Notes)
    $Bitmap.Save($Path, [System.Drawing.Imaging.ImageFormat]::Png)
    $Graphics.Dispose()
    $Bitmap.Dispose()
    Add-Manifest $Category $Path $Source $Notes
}

function New-ChartCanvas {
    param([int]$Width = 1300, [int]$Height = 850)
    $bmp = New-Object System.Drawing.Bitmap $Width, $Height
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
    $g.FillRectangle($White, 0, 0, $Width, $Height)
    [pscustomobject]@{ Bitmap = $bmp; Graphics = $g; Width = $Width; Height = $Height }
}

function Draw-Centered {
    param($Graphics, [string]$Text, $Font, $Brush, [double]$Cx, [double]$Y)
    $size = $Graphics.MeasureString($Text, $Font)
    $Graphics.DrawString($Text, $Font, $Brush, [float]($Cx - $size.Width / 2.0), [float]$Y)
}

function Draw-Right {
    param($Graphics, [string]$Text, $Font, $Brush, [double]$Right, [double]$Y)
    $size = $Graphics.MeasureString($Text, $Font)
    $Graphics.DrawString($Text, $Font, $Brush, [float]($Right - $size.Width), [float]$Y)
}

function X-Map {
    param([double]$X, [double]$Left, [double]$Width, [double]$Min, [double]$Max)
    $Left + (($X - $Min) / ($Max - $Min)) * $Width
}

function Y-Map {
    param([double]$Y, [double]$Top, [double]$Height, [double]$Min, [double]$Max)
    $Top + (1.0 - (($Y - $Min) / ($Max - $Min))) * $Height
}

function Format-Value {
    param([double]$Value, [string]$Format)
    switch ($Format) {
        "percent" { return (($Value * 100.0).ToString("0.0") + "%") }
        "one" { return $Value.ToString("0.0") }
        "two" { return $Value.ToString("0.00") }
        default { return $Value.ToString("0.00") }
    }
}

function Draw-HorizontalBarChart {
    param(
        [object[]]$Rows,
        [string]$Title,
        [string]$ValueLabel,
        [string]$OutPath,
        [string]$Format = "two",
        [double]$MinValue = [double]::NaN,
        [double]$MaxValue = [double]::NaN,
        [string]$Notes = ""
    )
    $canvas = New-ChartCanvas 1400 900
    $g = $canvas.Graphics
    Draw-Centered $g $Title $FontTitle $Black ($canvas.Width/2) 28

    $left = 310; $right = 1280; $top = 110; $bottom = 780
    $plotW = $right - $left; $plotH = $bottom - $top
    if ([double]::IsNaN($MinValue)) { $MinValue = [Math]::Min(0, ($Rows | Measure-Object -Property value -Minimum).Minimum) }
    if ([double]::IsNaN($MaxValue)) { $MaxValue = ($Rows | Measure-Object -Property value -Maximum).Maximum * 1.12 }
    if ($MaxValue -eq $MinValue) { $MaxValue = $MinValue + 1 }
    $zeroX = X-Map 0 $left $plotW $MinValue $MaxValue

    for ($i = 0; $i -le 5; $i++) {
        $tick = $MinValue + (($MaxValue - $MinValue) * $i / 5.0)
        $x = X-Map $tick $left $plotW $MinValue $MaxValue
        $g.DrawLine($GridPen, [int]$x, [int]$top, [int]$x, [int]$bottom)
        Draw-Centered $g (Format-Value $tick $Format) $FontTick $Black $x ($bottom + 12)
    }
    $g.DrawLine($AxisPen, [int]$left, [int]$bottom, [int]$right, [int]$bottom)
    $g.DrawLine($AxisPen, [int]$zeroX, [int]$top, [int]$zeroX, [int]$bottom)

    $barH = [Math]::Min(52, ($plotH / $Rows.Count) * 0.62)
    $gap = ($plotH - ($barH * $Rows.Count)) / [Math]::Max(1, ($Rows.Count - 1))
    for ($i = 0; $i -lt $Rows.Count; $i++) {
        $r = $Rows[$i]
        $y = $top + $i * ($barH + $gap)
        $xVal = X-Map ([double]$r.value) $left $plotW $MinValue $MaxValue
        $x0 = [Math]::Min($zeroX, $xVal)
        $wBar = [Math]::Abs($xVal - $zeroX)
        Draw-Right $g ([string]$r.label) $FontTick $Black ($left - 18) ($y + 11)
        $rect = New-Object System.Drawing.Rectangle ([int]$x0), ([int]$y), ([int]$wBar), ([int]$barH)
        $brush = if ($i -eq 0) { $Dark } else { $Mid }
        $g.FillRectangle($brush, $rect)
        $g.DrawRectangle($AxisPen, $rect)
        $label = Format-Value ([double]$r.value) $Format
        $lx = if ([double]$r.value -ge 0) { $xVal + 42 } else { $xVal - 42 }
        if ($lx -gt ($right - 25)) { $lx = $xVal - 45; $labelBrush = $White } else { $labelBrush = $Black }
        Draw-Centered $g $label $FontValue $labelBrush $lx ($y + 11)
    }
    Draw-Centered $g $ValueLabel $FontAxis $Black (($left+$right)/2) ($bottom + 55)
    if ($Notes -ne "") { Draw-Centered $g $Notes $FontSmall $Black (($left+$right)/2) 830 }
    Save-Bitmap $canvas.Bitmap $g $OutPath "generated_summary_charts" "attempt9 model_metrics.csv" $Title
}

function Draw-GroupedCategoryBarChart {
    param(
        [object[]]$Rows,
        [string]$Title,
        [string]$YLabel,
        [string]$OutPath,
        [object[]]$Series,
        [string]$Notes = ""
    )
    $canvas = New-ChartCanvas 1300 850
    $g = $canvas.Graphics
    Draw-Centered $g $Title $FontTitle $Black ($canvas.Width/2) 28
    $left = 140; $right = 1220; $top = 110; $bottom = 690
    $plotW = $right - $left; $plotH = $bottom - $top
    $maxVal = 0
    foreach ($s in $Series) {
        $m = ($Rows | Measure-Object -Property $s.prop -Maximum).Maximum
        if ($m -gt $maxVal) { $maxVal = $m }
    }
    $yMax = [Math]::Ceiling(($maxVal * 1.2) / 2.0) * 2.0
    if ($yMax -lt 1) { $yMax = 1 }
    for ($i = 0; $i -le 5; $i++) {
        $yt = $yMax * $i / 5.0
        $y = Y-Map $yt $top $plotH 0 $yMax
        $g.DrawLine($GridPen, [int]$left, [int]$y, [int]$right, [int]$y)
        Draw-Right $g ($yt.ToString("0.0")) $FontTick $Black ($left - 14) ($y - 11)
    }
    $g.DrawLine($AxisPen, [int]$left, [int]$top, [int]$left, [int]$bottom)
    $g.DrawLine($AxisPen, [int]$left, [int]$bottom, [int]$right, [int]$bottom)
    $groupW = $plotW / [Math]::Max(1, $Rows.Count)
    $barW = [Math]::Min(95, ($groupW * 0.62) / $Series.Count)
    for ($i = 0; $i -lt $Rows.Count; $i++) {
        $r = $Rows[$i]
        $cx = $left + ($i + 0.5) * $groupW
        Draw-Centered $g ([string]$r.label) $FontTick $Black $cx ($bottom + 12)
        for ($j = 0; $j -lt $Series.Count; $j++) {
            $s = $Series[$j]
            $val = [double]$r.($s.prop)
            $barH = ($val / $yMax) * $plotH
            $x = $cx - (($Series.Count * $barW) / 2.0) + ($j * $barW)
            $y = $bottom - $barH
            $rect = New-Object System.Drawing.Rectangle ([int]$x), ([int]$y), ([int]($barW-4)), ([int]$barH)
            $brush = if ($j -eq 0) { $Dark } else { $Mid }
            $g.FillRectangle($brush, $rect)
            $g.DrawRectangle($AxisPen, $rect)
            Draw-Centered $g ($val.ToString("0.00")) $FontValue $Black ($x + ($barW/2)) ($y - 26)
        }
    }
    Draw-Centered $g $YLabel $FontAxis $Black 58 (($top+$bottom)/2)
    Draw-Centered $g "Condition" $FontAxis $Black (($left+$right)/2) ($bottom + 55)
    $legendX = $right - 250; $legendY = 128
    $g.FillRectangle($White, $legendX-18, $legendY-18, 225, 36 + ($Series.Count*36))
    $g.DrawRectangle($AxisPen, $legendX-18, $legendY-18, 225, 36 + ($Series.Count*36))
    for ($j = 0; $j -lt $Series.Count; $j++) {
        $s = $Series[$j]
        $y = $legendY + ($j*36)
        $brush = if ($j -eq 0) { $Dark } else { $Mid }
        $g.FillRectangle($brush, $legendX, $y-8, 42, 16)
        $g.DrawRectangle($AxisPen, $legendX, $y-8, 42, 16)
        $g.DrawString($s.label, $FontSmall, $Black, $legendX+58, $y-12)
    }
    if ($Notes -ne "") { Draw-Centered $g $Notes $FontSmall $Black (($left+$right)/2) 785 }
    Save-Bitmap $canvas.Bitmap $g $OutPath "generated_summary_charts" "attempt9 predictions.csv" $Title
}

function Draw-LineChart {
    param(
        [object[]]$Rows,
        [string]$Title,
        [string]$YLabel,
        [string]$OutPath,
        [object[]]$Series,
        [double]$YMax = [double]::NaN,
        [string]$Notes = ""
    )
    $canvas = New-ChartCanvas 1400 850
    $g = $canvas.Graphics
    Draw-Centered $g $Title $FontTitle $Black ($canvas.Width/2) 28
    $left = 120; $right = 1300; $top = 105; $bottom = 690
    $plotW = $right - $left; $plotH = $bottom - $top
    $xmin = 0
    $xmax = [Math]::Max(80, (($Rows | Measure-Object -Property x -Maximum).Maximum + 5))
    if ([double]::IsNaN($YMax)) {
        $maxVal = 0
        foreach ($s in $Series) {
            $m = ($Rows | Measure-Object -Property $s.prop -Maximum).Maximum
            if ($m -gt $maxVal) { $maxVal = $m }
        }
        $YMax = [Math]::Ceiling($maxVal * 1.18 / 5.0) * 5
    }
    if ($YMax -lt 1) { $YMax = 1 }
    for ($xt = 0; $xt -le $xmax; $xt += 10) {
        $x = X-Map $xt $left $plotW $xmin $xmax
        $g.DrawLine($GridPen, [int]$x, [int]$top, [int]$x, [int]$bottom)
        Draw-Centered $g ([string]$xt) $FontTick $Black $x ($bottom + 10)
    }
    for ($i = 0; $i -le 5; $i++) {
        $yt = $YMax * $i / 5.0
        $y = Y-Map $yt $top $plotH 0 $YMax
        $g.DrawLine($GridPen, [int]$left, [int]$y, [int]$right, [int]$y)
        Draw-Right $g ($yt.ToString("0.0")) $FontTick $Black ($left - 14) ($y - 11)
    }
    $g.DrawLine($AxisPen, [int]$left, [int]$top, [int]$left, [int]$bottom)
    $g.DrawLine($AxisPen, [int]$left, [int]$bottom, [int]$right, [int]$bottom)
    foreach ($s in $Series) {
        $pts = New-Object System.Collections.Generic.List[System.Drawing.PointF]
        foreach ($r in $Rows) {
            $pts.Add([System.Drawing.PointF]::new([float](X-Map ([double]$r.x) $left $plotW $xmin $xmax), [float](Y-Map ([double]$r.($s.prop)) $top $plotH 0 $YMax)))
        }
        $pen = if ($s.style -eq "dash") { $DashedPen } elseif ($s.style -eq "gray") { $GrayLinePen } else { $LinePen }
        if ($pts.Count -gt 1) { $g.DrawLines($pen, $pts.ToArray()) }
        foreach ($p in $pts) {
            if ($s.style -eq "dash") { $g.FillEllipse($White, $p.X-5, $p.Y-5, 10, 10) } else { $g.FillEllipse($(if ($s.style -eq "gray") { $Mid } else { $Black }), $p.X-5, $p.Y-5, 10, 10) }
            $g.DrawEllipse($AxisPen, $p.X-5, $p.Y-5, 10, 10)
        }
    }
    Draw-Centered $g "True distance (m)" $FontAxis $Black (($left+$right)/2) ($bottom + 52)
    $state = $g.Save()
    $g.TranslateTransform(42, [float](($top+$bottom)/2))
    $g.RotateTransform(-90)
    Draw-Centered $g $YLabel $FontAxis $Black 0 -12
    $g.Restore($state)

    $legendX = $right - 310; $legendY = 122
    $g.FillRectangle($White, $legendX-18, $legendY-18, 285, 36 + ($Series.Count*36))
    $g.DrawRectangle($AxisPen, $legendX-18, $legendY-18, 285, 36 + ($Series.Count*36))
    for ($i = 0; $i -lt $Series.Count; $i++) {
        $s = $Series[$i]
        $y = $legendY + ($i*36)
        $pen = if ($s.style -eq "dash") { $DashedPen } elseif ($s.style -eq "gray") { $GrayLinePen } else { $LinePen }
        $g.DrawLine($pen, $legendX, $y, $legendX+65, $y)
        $g.DrawString($s.label, $FontSmall, $Black, $legendX+82, $y-12)
    }
    if ($Notes -ne "") { Draw-Centered $g $Notes $FontSmall $Black (($left+$right)/2) 785 }
    Save-Bitmap $canvas.Bitmap $g $OutPath "generated_summary_charts" "attempt9/attempt8 CSV data" $Title
}

function Draw-Heatmap {
    param(
        [object[]]$Cells,
        [string[]]$RowOrder,
        [string[]]$ColOrder,
        [string]$Title,
        [string]$OutPath,
        [string]$Format = "two",
        [bool]$HigherIsDarker = $true,
        [string]$Notes = ""
    )
    $canvas = New-ChartCanvas 1250 900
    $g = $canvas.Graphics
    Draw-Centered $g $Title $FontTitle $Black ($canvas.Width/2) 28
    $left = 330; $top = 130
    $cellW = [Math]::Floor(780 / $ColOrder.Count)
    $cellH = [Math]::Floor(600 / $RowOrder.Count)
    $values = @($Cells | ForEach-Object { [double]$_.value })
    $min = ($values | Measure-Object -Minimum).Minimum
    $max = ($values | Measure-Object -Maximum).Maximum
    if ($max -eq $min) { $max = $min + 1 }
    for ($c = 0; $c -lt $ColOrder.Count; $c++) {
        Draw-Centered $g $ColOrder[$c] $FontAxis $Black ($left + $c*$cellW + $cellW/2) 88
    }
    for ($r = 0; $r -lt $RowOrder.Count; $r++) {
        $rowName = $RowOrder[$r]
        Draw-Right $g $rowName $FontTick $Black ($left - 18) ($top + $r*$cellH + $cellH/2 - 12)
        for ($c = 0; $c -lt $ColOrder.Count; $c++) {
            $colName = $ColOrder[$c]
            $cell = $Cells | Where-Object { $_.row -eq $rowName -and $_.col -eq $colName } | Select-Object -First 1
            if ($null -eq $cell) { continue }
            $v = [double]$cell.value
            $t = ($v - $min) / ($max - $min)
            if (-not $HigherIsDarker) { $t = 1 - $t }
            $tone = [int](245 - (205 * [Math]::Pow($t, 0.75)))
            if ($tone -lt 35) { $tone = 35 }
            if ($tone -gt 245) { $tone = 245 }
            $brush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb($tone,$tone,$tone))
            $x = $left + $c*$cellW
            $y = $top + $r*$cellH
            $rect = New-Object System.Drawing.Rectangle ([int]$x), ([int]$y), ([int]$cellW), ([int]$cellH)
            $g.FillRectangle($brush, $rect)
            $g.DrawRectangle($AxisPen, $rect)
            $labelBrush = if ($tone -lt 125) { $White } else { $Black }
            Draw-Centered $g (Format-Value $v $Format) $FontValue $labelBrush ($x+$cellW/2) ($y+$cellH/2-12)
            $brush.Dispose()
        }
    }
    if ($Notes -ne "") { Draw-Centered $g $Notes $FontSmall $Black ($canvas.Width/2) 810 }
    Save-Bitmap $canvas.Bitmap $g $OutPath "generated_summary_charts" "attempt9 metrics/predictions" $Title
}

function Get-R2 {
    param([object[]]$Rows)
    $truth = @($Rows | ForEach-Object { [double]$_.true_distance_m })
    if ($truth.Count -eq 0) { return [double]::NaN }
    $mean = ($truth | Measure-Object -Average).Average
    $ssTot = 0.0
    $ssRes = 0.0
    foreach ($r in $Rows) {
        $t = [double]$r.true_distance_m
        $p = [double]$r.predicted_distance_m
        $ssTot += [Math]::Pow($t - $mean, 2)
        $ssRes += [Math]::Pow($t - $p, 2)
    }
    if ($ssTot -eq 0) { return [double]::NaN }
    1.0 - ($ssRes / $ssTot)
}

function Get-Metrics {
    param([object[]]$Rows, [string]$Label = "")
    $n = $Rows.Count
    $mae = ($Rows | Measure-Object -Property absolute_error_m -Average).Average
    $rmse = [Math]::Sqrt((($Rows | ForEach-Object { [Math]::Pow([double]$_.signed_error_m, 2) }) | Measure-Object -Average).Average)
    $within10 = (@($Rows | Where-Object { [double]$_.absolute_error_m -le 10 }).Count / [double]$n)
    [pscustomobject]@{
        label = $Label
        count = $n
        mae = $mae
        rmse = $rmse
        r2 = Get-R2 $Rows
        within10 = $within10
    }
}

$ModelMetricsPath = Join-Path $Attempt9 "reports\model_metrics.csv"
$DistanceRangePath = Join-Path $Attempt9 "reports\distance_range_metrics.csv"
$PredictionsPath = Join-Path $Attempt9 "reports\predictions.csv"
$RealDistancePath = Join-Path $Attempt8Real "reports\before_after_by_distance.csv"
$RealDronePath = Join-Path $Attempt8Real "reports\before_after_by_drone_type.csv"
$RawMetricsPath = Join-Path $Attempt8Real "reports\raw_uncalibrated_metrics.json"
$CalMetricsPath = Join-Path $Attempt8Real "reports\final_calibrated_metrics.json"

$modelMetrics = Import-Csv -LiteralPath $ModelMetricsPath
$posterTest = @($modelMetrics | Where-Object { $_.split_name -eq "test" -and $_.role -eq "poster" } |
    Sort-Object { [double]$_.poster_rank })
$modelOrder = @($posterTest | Sort-Object { [double]$_.poster_rank } | ForEach-Object { $_.display_name })

Draw-HorizontalBarChart `
    -Rows @($posterTest | Sort-Object { [double]$_.mae } | ForEach-Object { [pscustomobject]@{ label = $_.display_name; value = [double]$_.mae } }) `
    -Title "Held-Out Test MAE by Model" `
    -ValueLabel "MAE (m), lower is better" `
    -OutPath (Join-Path $Dirs.Generated "synthetic_test_mae_by_model.png") `
    -Format "two"

Draw-HorizontalBarChart `
    -Rows @($posterTest | Sort-Object { [double]$_.rmse } | ForEach-Object { [pscustomobject]@{ label = $_.display_name; value = [double]$_.rmse } }) `
    -Title "Held-Out Test RMSE by Model" `
    -ValueLabel "RMSE (m), lower is better" `
    -OutPath (Join-Path $Dirs.Generated "synthetic_test_rmse_by_model.png") `
    -Format "two"

Draw-HorizontalBarChart `
    -Rows @($posterTest | Sort-Object { [double]$_.r2 } -Descending | ForEach-Object { [pscustomobject]@{ label = $_.display_name; value = [double]$_.r2 } }) `
    -Title "Held-Out Test R2 by Model" `
    -ValueLabel "R2, higher is better" `
    -OutPath (Join-Path $Dirs.Generated "synthetic_test_r2_by_model.png") `
    -Format "two" `
    -MinValue -4.5 `
    -MaxValue 1.05

Draw-HorizontalBarChart `
    -Rows @($posterTest | Sort-Object { [double]$_.within_10m_rate } -Descending | ForEach-Object { [pscustomobject]@{ label = $_.display_name; value = [double]$_.within_10m_rate } }) `
    -Title "Test Predictions Within 10 m by Model" `
    -ValueLabel "Fraction within 10 m, higher is better" `
    -OutPath (Join-Path $Dirs.Generated "synthetic_test_within10_by_model.png") `
    -Format "percent" `
    -MinValue 0 `
    -MaxValue 1.0

$distanceRows = @(Import-Csv -LiteralPath $DistanceRangePath | Where-Object { $_.split_name -eq "test" -and $_.role -eq "poster" })
$rangeOrder = @("near", "mid", "far")
foreach ($metric in @("mae", "rmse", "r2", "within_10m_rate")) {
    $cells = @($distanceRows | ForEach-Object {
        [pscustomobject]@{
            row = $_.display_name
            col = $_.distance_range
            value = [double]$_.$metric
        }
    })
    $fmt = if ($metric -eq "within_10m_rate") { "percent" } else { "two" }
    $nice = switch ($metric) {
        "mae" { "MAE" }
        "rmse" { "RMSE" }
        "r2" { "R2" }
        "within_10m_rate" { "Within 10 m" }
    }
    Draw-Heatmap `
        -Cells $cells `
        -RowOrder $modelOrder `
        -ColOrder $rangeOrder `
        -Title ("Synthetic Distance-Band " + $nice + " by Model") `
        -OutPath (Join-Path $Dirs.Generated ("synthetic_distance_band_" + $metric + "_by_model.png")) `
        -Format $fmt `
        -HigherIsDarker ($metric -ne "r2" -and $metric -ne "within_10m_rate") `
        -Notes "near <= 60 m, mid 60-100 m, far > 100 m"
}

$pred = @(Import-Csv -LiteralPath $PredictionsPath | Where-Object { $_.display_name -eq "ensemble" -and $_.split_name -eq "test" })
$exactDistanceMetrics = @($pred | Group-Object true_distance_m | ForEach-Object {
    $m = Get-Metrics $_.Group ([string]$_.Name)
    [pscustomobject]@{
        x = [double]$_.Name
        mae = $m.mae
        rmse = $m.rmse
        within10 = $m.within10
        mean_prediction = ($_.Group | Measure-Object -Property predicted_distance_m -Average).Average
    }
} | Sort-Object x)

Draw-LineChart `
    -Rows $exactDistanceMetrics `
    -Title "Final Ensemble MAE by True Distance" `
    -YLabel "MAE (m)" `
    -OutPath (Join-Path $Dirs.Generated "synthetic_final_ensemble_mae_by_true_distance.png") `
    -Series @([pscustomobject]@{ prop = "mae"; label = "MAE"; style = "solid" })

Draw-LineChart `
    -Rows $exactDistanceMetrics `
    -Title "Final Ensemble RMSE by True Distance" `
    -YLabel "RMSE (m)" `
    -OutPath (Join-Path $Dirs.Generated "synthetic_final_ensemble_rmse_by_true_distance.png") `
    -Series @([pscustomobject]@{ prop = "rmse"; label = "RMSE"; style = "solid" })

Draw-LineChart `
    -Rows $exactDistanceMetrics `
    -Title "Final Ensemble Within-10-m Rate by True Distance" `
    -YLabel "Fraction within 10 m" `
    -OutPath (Join-Path $Dirs.Generated "synthetic_final_ensemble_within10_by_true_distance.png") `
    -Series @([pscustomobject]@{ prop = "within10"; label = "Within 10 m"; style = "solid" }) `
    -YMax 1.0

Draw-LineChart `
    -Rows $exactDistanceMetrics `
    -Title "Mean Predicted Distance vs. True Distance" `
    -YLabel "Distance (m)" `
    -OutPath (Join-Path $Dirs.Generated "synthetic_final_ensemble_mean_prediction_by_true_distance.png") `
    -Series @(
        [pscustomobject]@{ prop = "x"; label = "True distance"; style = "solid" },
        [pscustomobject]@{ prop = "mean_prediction"; label = "Mean prediction"; style = "gray" }
    ) `
    -YMax 155

$weatherMetrics = @($pred | Group-Object weather | ForEach-Object {
    $m = Get-Metrics $_.Group $_.Name
    [pscustomobject]@{ label = $_.Name; mae = $m.mae; rmse = $m.rmse; r2 = $m.r2; within10 = $m.within10 }
})
$timeMetrics = @($pred | Group-Object time_of_day | ForEach-Object {
    $m = Get-Metrics $_.Group $_.Name
    [pscustomobject]@{ label = $_.Name; mae = $m.mae; rmse = $m.rmse; r2 = $m.r2; within10 = $m.within10 }
})
$weatherTimeMetrics = @($pred | Group-Object weather,time_of_day | ForEach-Object {
    $parts = $_.Name -split ", "
    $m = Get-Metrics $_.Group $_.Name
    [pscustomobject]@{ row = $parts[0]; col = $parts[1]; mae = $m.mae; rmse = $m.rmse; r2 = $m.r2; within10 = $m.within10 }
})

foreach ($metric in @("mae", "rmse", "r2", "within10")) {
    $fmt = if ($metric -eq "within10") { "percent" } else { "two" }
    $nice = switch ($metric) {
        "mae" { "MAE" }
        "rmse" { "RMSE" }
        "r2" { "R2" }
        "within10" { "Within 10 m" }
    }
    Draw-Heatmap `
        -Cells @($weatherTimeMetrics | ForEach-Object { [pscustomobject]@{ row = $_.row; col = $_.col; value = [double]$_.$metric } }) `
        -RowOrder @("clear_sky", "light_rain") `
        -ColOrder @("10AM", "8PM") `
        -Title ("Weather-Time " + $nice + " for Final Ensemble") `
        -OutPath (Join-Path $Dirs.Generated ("synthetic_weather_time_" + $metric + ".png")) `
        -Format $fmt `
        -HigherIsDarker ($metric -eq "mae" -or $metric -eq "rmse")
}

Draw-GroupedCategoryBarChart `
    -Rows @($weatherMetrics | ForEach-Object { [pscustomobject]@{ label = $_.label; mae = $_.mae; rmse = $_.rmse } }) `
    -Title "Final Ensemble Error by Weather Condition" `
    -YLabel "Error (m)" `
    -OutPath (Join-Path $Dirs.Generated "synthetic_weather_mae_rmse.png") `
    -Series @(
        [pscustomobject]@{ prop = "mae"; label = "MAE"; style = "solid" },
        [pscustomobject]@{ prop = "rmse"; label = "RMSE"; style = "gray" }
    )

Draw-GroupedCategoryBarChart `
    -Rows @($timeMetrics | ForEach-Object { [pscustomobject]@{ label = $_.label; mae = $_.mae; rmse = $_.rmse } }) `
    -Title "Final Ensemble Error by Time of Day" `
    -YLabel "Error (m)" `
    -OutPath (Join-Path $Dirs.Generated "synthetic_time_of_day_mae_rmse.png") `
    -Series @(
        [pscustomobject]@{ prop = "mae"; label = "MAE"; style = "solid" },
        [pscustomobject]@{ prop = "rmse"; label = "RMSE"; style = "gray" }
    )

$realByDistance = @(Import-Csv -LiteralPath $RealDistancePath | Sort-Object { [double]$_.true_distance_m } | ForEach-Object {
    [pscustomobject]@{
        x = [double]$_.true_distance_m
        raw_mae = [double]$_.raw_mae
        calibrated_mae = [double]$_.calibrated_mae
        raw_within10 = [double]$_.raw_within_10m_rate
        calibrated_within10 = [double]$_.calibrated_within_10m_rate
    }
})

Draw-LineChart `
    -Rows $realByDistance `
    -Title "Real-World MAE Before and After Calibration" `
    -YLabel "MAE (m)" `
    -OutPath (Join-Path $Dirs.Generated "real_calibration_mae_before_after_by_distance.png") `
    -Series @(
        [pscustomobject]@{ prop = "raw_mae"; label = "Before calibration"; style = "dash" },
        [pscustomobject]@{ prop = "calibrated_mae"; label = "After calibration"; style = "gray" }
    )

Draw-LineChart `
    -Rows $realByDistance `
    -Title "Real-World Within-10-m Rate Before and After Calibration" `
    -YLabel "Fraction within 10 m" `
    -OutPath (Join-Path $Dirs.Generated "real_calibration_within10_before_after_by_distance.png") `
    -Series @(
        [pscustomobject]@{ prop = "raw_within10"; label = "Before calibration"; style = "dash" },
        [pscustomobject]@{ prop = "calibrated_within10"; label = "After calibration"; style = "gray" }
    ) `
    -YMax 1.0

$realByDrone = @(Import-Csv -LiteralPath $RealDronePath)
$droneCells = @()
foreach ($r in $realByDrone) {
    $droneCells += [pscustomobject]@{ row = $r.drone_type; col = "before"; value = [double]$r.raw_mae }
    $droneCells += [pscustomobject]@{ row = $r.drone_type; col = "after"; value = [double]$r.calibrated_mae }
}
Draw-Heatmap `
    -Cells $droneCells `
    -RowOrder @("Kongsberg", "Vestfold") `
    -ColOrder @("before", "after") `
    -Title "Real-World MAE by Drone Subset Before/After Calibration" `
    -OutPath (Join-Path $Dirs.Generated "real_calibration_mae_by_drone_subset.png") `
    -Format "two" `
    -HigherIsDarker $true

$rawOverall = Get-Content -LiteralPath $RawMetricsPath | ConvertFrom-Json
$calOverall = Get-Content -LiteralPath $CalMetricsPath | ConvertFrom-Json
$overallRows = @(
    [pscustomobject]@{ label = "Raw real prediction"; mae = [double]$rawOverall.mae; rmse = [double]$rawOverall.rmse; r2 = [double]$rawOverall.r2; within10 = [double]$rawOverall.within_10m_rate },
    [pscustomobject]@{ label = "Calibrated real prediction"; mae = [double]$calOverall.mae; rmse = [double]$calOverall.rmse; r2 = [double]$calOverall.r2; within10 = [double]$calOverall.within_10m_rate }
)

Draw-HorizontalBarChart `
    -Rows @($overallRows | ForEach-Object { [pscustomobject]@{ label = $_.label; value = $_.mae } }) `
    -Title "Real-World Overall MAE Before/After Calibration" `
    -ValueLabel "MAE (m), lower is better" `
    -OutPath (Join-Path $Dirs.Generated "real_calibration_overall_mae.png") `
    -Format "two"

Draw-HorizontalBarChart `
    -Rows @($overallRows | ForEach-Object { [pscustomobject]@{ label = $_.label; value = $_.rmse } }) `
    -Title "Real-World Overall RMSE Before/After Calibration" `
    -ValueLabel "RMSE (m), lower is better" `
    -OutPath (Join-Path $Dirs.Generated "real_calibration_overall_rmse.png") `
    -Format "two"

Draw-HorizontalBarChart `
    -Rows @($overallRows | Sort-Object r2 -Descending | ForEach-Object { [pscustomobject]@{ label = $_.label; value = $_.r2 } }) `
    -Title "Real-World Overall R2 Before/After Calibration" `
    -ValueLabel "R2, higher is better" `
    -OutPath (Join-Path $Dirs.Generated "real_calibration_overall_r2.png") `
    -Format "two" `
    -MinValue -1.2 `
    -MaxValue 1.05

Draw-HorizontalBarChart `
    -Rows @($overallRows | Sort-Object within10 -Descending | ForEach-Object { [pscustomobject]@{ label = $_.label; value = $_.within10 } }) `
    -Title "Real-World Overall Within-10-m Rate Before/After Calibration" `
    -ValueLabel "Fraction within 10 m, higher is better" `
    -OutPath (Join-Path $Dirs.Generated "real_calibration_overall_within10.png") `
    -Format "percent" `
    -MinValue 0 `
    -MaxValue 1.0

$ManifestPath = Join-Path $OutRoot "manifest.csv"
$Manifest | Export-Csv -LiteralPath $ManifestPath -NoTypeInformation

$readme = @"
# forLior final results package

This folder contains only artifacts copied or generated from attempt 8 and newer:

- attempt8-comprehensive run
- attempt9

Folders:

- existing_attempt9_final_depthpro: copied final synthetic DepthPro plots and PNG figures.
- existing_attempt8_synthetic: copied attempt8 synthetic plots, including weather/time and feature pipeline plots.
- existing_attempt8_real_calibration: copied attempt8 real-world calibration plots.
- generated_summary_charts: new black-and-white PNG charts generated from attempt8/attempt9 CSV/JSON reports.
- source_data: selected CSV/JSON/MD files used to generate the charts.

Key generated charts include model MAE/RMSE/R2/within-10-m rankings, distance-band MAE/RMSE/R2/within-10-m heatmaps, final ensemble error by exact distance, weather/time MAE/RMSE/R2 charts, and real-world calibration before/after charts.

See manifest.csv for source paths.
"@
Set-Content -LiteralPath (Join-Path $OutRoot "README.md") -Value $readme
Add-Manifest "documentation" (Join-Path $OutRoot "README.md") "generated" "Package description"

$Manifest | Export-Csv -LiteralPath $ManifestPath -NoTypeInformation
Write-Output "Created forLior package at: $OutRoot"
Write-Output "Generated charts: $((Get-ChildItem -LiteralPath $Dirs.Generated -File -Filter *.png).Count)"
Write-Output "Copied existing attempt8/attempt9 plots: $((Get-ChildItem -LiteralPath $Dirs.Attempt9Final,$Dirs.Attempt8Synthetic,$Dirs.Attempt8Real -File | Where-Object { $_.Extension -in @('.png','.svg','.jpg','.jpeg') }).Count)"
