param(
    [string]$Scenario = "data\raw\tapas_cologne\TAPASCologne-0.32.0\cologne6to8.sumocfg",
    [string]$Output = "data\raw\tapas_cologne\tapas_6to8_fcd.xml",
    [double]$Scale = 0.01,
    [double]$PeriodSeconds = 1.0
)

$sumo = Get-Command sumo -ErrorAction SilentlyContinue
if (-not $sumo) {
    throw "SUMO is not installed or is not available on PATH."
}

& $sumo.Source `
    -c $Scenario `
    --scale $Scale `
    --fcd-output $Output `
    --fcd-output.geo `
    --device.fcd.period $PeriodSeconds `
    --no-step-log true

if ($LASTEXITCODE -ne 0) {
    throw "SUMO exited with code $LASTEXITCODE."
}

Get-Item -LiteralPath $Output | Select-Object FullName, Length, LastWriteTime
