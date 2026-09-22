$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = (Resolve-Path (Join-Path $repo '.venv\Scripts\python.exe')).Path
$gpuPreferences = 'HKCU:\Software\Microsoft\DirectX\UserGpuPreferences'

# Windows hybrid-GPU policy: 2 means High performance (the NVIDIA adapter).
New-Item -Path $gpuPreferences -Force | Out-Null
New-ItemProperty -Path $gpuPreferences -Name $python -PropertyType String `
    -Value 'GpuPreference=2;' -Force | Out-Null

Write-Host "MuJoCo renderer preference: NVIDIA high performance"
Write-Host "Python: $python"
& $python -u -m simulation.pick_place_demo @args
exit $LASTEXITCODE
