$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
cd $here

if (-not (Test-Path ".\.venv")) {
  python -m venv .venv
}
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip -q
pip install -r requirements.txt

if (-not (Test-Path ".\.env") -and (Test-Path ".\.env.example")) {
  Copy-Item .\.env.example .\.env
  Write-Host "[i] Created proxy/.env from example. Put NVIDIA_API_KEY there before first run."
}

uvicorn app:app --host $env:HOST --port $env:PORT --reload
