$ErrorActionPreference = "Stop"

$workspaceRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $workspaceRoot ".venv\Scripts\python.exe"
$pythonCommand = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { "python" }

Push-Location $workspaceRoot
try {
    & $pythonCommand -m exoswarm.cli verify-cache
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & $pythonCommand -m pytest -q
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & $pythonCommand -m exoswarm.cli reproduce
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}

