# Builds the offline Windows bundle: dictation.exe + models\ivrit-turbo + SHA256SUMS + install.ps1.
# Usage: .\make-bundle.ps1 -Exe ..\target\release\dictation.exe -ModelDir <folder with config.json and model.safetensors> -Out .\dist
param(
    [Parameter(Mandatory)][string]$Exe,
    [Parameter(Mandatory)][string]$ModelDir,
    [string]$Out = "$PSScriptRoot\dist",
    [string]$ModelName = "ivrit-turbo"
)
$ErrorActionPreference = "Stop"
$bundle = Join-Path $Out "dictation-windows"
if (Test-Path $bundle) { Remove-Item $bundle -Recurse -Force }
$model = Join-Path $bundle "models\$ModelName"
New-Item -ItemType Directory $model -Force | Out-Null
Copy-Item $Exe (Join-Path $bundle "dictation.exe")
foreach ($f in "config.json", "model.safetensors") { Copy-Item (Join-Path $ModelDir $f) $model }
Copy-Item "$PSScriptRoot\install.ps1" $bundle
Copy-Item "$PSScriptRoot\..\THIRD_PARTY_NOTICES.md" $bundle -ErrorAction SilentlyContinue

$lines = foreach ($f in "config.json", "model.safetensors") {
    $h = (Get-FileHash (Join-Path $model $f) -Algorithm SHA256).Hash.ToLower()
    "$h  $f"
}
# LF endings, no BOM, so `sha256sum -c` and `dictation verify-model` both accept it.
[IO.File]::WriteAllText((Join-Path $model "SHA256SUMS"), (($lines -join "`n") + "`n"), (New-Object Text.UTF8Encoding $false))
Write-Host "bundle ready: $bundle"