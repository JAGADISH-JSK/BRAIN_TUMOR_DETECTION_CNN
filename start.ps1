# Starts the backend and the frontend, each in its own PowerShell window.
# Run from the project root:
#   powershell -ExecutionPolicy Bypass -File .\start.ps1
# Optional: change the review threshold, e.g.  -Threshold 0.25
param([string]$Threshold = "0.18")

$root = $PSScriptRoot

$backend = "cd '$root'; .\venv\Scripts\Activate.ps1; `$env:ENTROPY_THRESHOLD = '$Threshold'; python -m backend.app"
$frontend = "cd '$root\frontend'; if (-not (Test-Path node_modules)) { npm install }; npm run dev"

Start-Process powershell -ArgumentList "-NoExit", "-Command", $backend
Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontend

Write-Host "Two windows opened. Wait for the backend window to print 'Ready', then open the link shown in the frontend window (usually http://localhost:5173)."
