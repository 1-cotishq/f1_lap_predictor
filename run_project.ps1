$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

if (Test-Path ".venv") {
    . .\.venv\Scripts\Activate.ps1
}

python src/f1_lap_predictor.py
