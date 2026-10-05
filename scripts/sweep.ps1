# Runs the streaming sweep on existing WAV files and saves the output.
# Usage (from d:\Transcriptor):  powershell -ExecutionPolicy Bypass -File .\scripts\sweep.ps1 -Dir <folder with wavs>
param(
    [string]$Dir = ".\clips",
    [string]$Model = "$env:USERPROFILE\.cache\dictation-models\ivrit-turbo",
    [string]$Exe = ".\target\release\dictation.exe"
)

$files = Get-ChildItem -Path $Dir -Filter *.wav
if (-not $files) { Write-Host "No .wav files in $Dir" -ForegroundColor Red; exit 1 }

$results = Join-Path $Dir "results.txt"
"" | Set-Content $results -Encoding utf8
foreach ($f in $files) {
    Write-Host "Sweeping $($f.Name)..." -ForegroundColor Cyan
    Add-Content $results "=== $($f.Name) ===" -Encoding utf8
    & $Exe stream-sweep $f.FullName --model-dir $Model --windows 1,2,3,5 --pad 2>&1 |
        ForEach-Object { Add-Content $results "$_" -Encoding utf8 }
}
Write-Host "Done. Results saved to $results" -ForegroundColor Green
