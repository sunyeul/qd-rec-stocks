# Daily data collection, for Windows Task Scheduler (see ingest/README.md).
# Runs from the repository root and writes a log to data\logs\.
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force -Path "data\logs" | Out-Null
$stamp = Get-Date -Format "yyyy-MM-dd_HHmm"
uv run python -m ingest.run_daily *>&1 | Tee-Object -FilePath "data\logs\ingest_$stamp.log"
exit $LASTEXITCODE
