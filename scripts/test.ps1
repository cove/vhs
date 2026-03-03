$ErrorActionPreference = "Stop"

$py = Join-Path $PSScriptRoot "..\\.venv\\Scripts\\python.exe"
if (-not (Test-Path $py)) {
  $py = "python"
}

& $py -m pytest @args
exit $LASTEXITCODE
