# Builds the Windows installer wizard (needs Inno Setup 6: winget install JRSoftware.InnoSetup).
# Usage: .\build-installer.ps1 -Exe ..\target\release\dictation.exe -ModelDir <ivrit-turbo folder> [-Version 1.0.0]
param(
    [Parameter(Mandatory)][string]$Exe,
    [Parameter(Mandatory)][string]$ModelDir,
    [string]$Version = "1.0.0",
    [string]$Out = "$PSScriptRoot\dist"
)
$ErrorActionPreference = "Stop"
$iscc = @("$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe", "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe") |
    Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) { throw "Inno Setup 6 not found" }

& "$PSScriptRoot\make-bundle.ps1" -Exe $Exe -ModelDir $ModelDir -Out $Out
$bundle = Join-Path $Out "dictation-windows"
& $iscc "/DBundleDir=$bundle" "/DAppVersion=$Version" "/DOutDir=$Out" "$PSScriptRoot\dictation.iss"
if ($LASTEXITCODE -ne 0) { throw "ISCC failed" }
Write-Host "Installer in $Out (copy dictation-setup-$Version.exe together with its .bin files)"