# PowerShell runner for Windows
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
cd $here

if (-not (Test-Path ".\.venv")) {
  python -m venv .venv
}
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip -q
pip install -r requirements.txt

# create .env from example if missing
if (-not (Test-Path ".\.env") -and (Test-Path ".\.env.example")) {
  Copy-Item .\.env.example .\.env
  Write-Host "[i] Created client/.env from example. Fill it with your PROXY_URL and PROXY_CLIENT_TOKEN."
}

# Example run (edit paths as needed)
python .\plate_model.py --video ".\videos\test.mp4" --out ".\out.csv" --sample-fps 5 --dedupe-window 2.0
