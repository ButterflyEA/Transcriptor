# Offline installer. Run from the bundle folder; needs no network.
#   .\install.ps1                  per-user install to %LOCALAPPDATA%\Programs\Dictation
#   .\install.ps1 -Dest D:\Apps\Dictation
param([string]$Dest = "$env:LOCALAPPDATA\Programs\Dictation", [switch]$SkipVerify)
$ErrorActionPreference = "Stop"
$src = $PSScriptRoot
if (-not (Test-Path "$src\dictation.exe")) { throw "dictation.exe not found next to install.ps1" }

New-Item -ItemType Directory $Dest -Force | Out-Null
Copy-Item "$src\dictation.exe" $Dest -Force
if (Test-Path "$src\models") { Copy-Item "$src\models" $Dest -Recurse -Force }
Get-ChildItem $src -Filter "THIRD_PARTY_NOTICES*" | Copy-Item -Destination $Dest -Force

$exe = Join-Path $Dest "dictation.exe"
if (-not $SkipVerify) {
    & $exe verify-model (Join-Path $Dest "models\ivrit-turbo")
    if ($LASTEXITCODE -ne 0) { throw "model files failed checksum verification; copy the bundle again" }
}

$startup = [Environment]::GetFolderPath("Programs")
$lnk = (New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $startup "Dictation.lnk"))
$lnk.TargetPath = $exe
$lnk.Arguments = "run"
$lnk.WorkingDirectory = $Dest
$lnk.Save()
Write-Host "Installed to $Dest. Start it from the Start menu (Dictation) or run: `"$exe`" run"