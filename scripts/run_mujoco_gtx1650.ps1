# Run a MuJoCo module on the NVIDIA GPU (Windows hybrid graphics).
#
#   .\scripts\run_mujoco_gtx1650.ps1                                   # simulation.pick_place_demo
#   .\scripts\run_mujoco_gtx1650.ps1 -Module simulation.pick_place.bin_conveyor_task --replay artifacts\bin_conveyor_frames.npz
#
# The venv's Scripts\python.exe is a launcher that starts the base interpreter, which
# is the process that opens the OpenGL window -- so the preference is set for both.
param(
    [string]$Module = 'simulation.pick_place_demo',
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest
)
$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = (Resolve-Path (Join-Path $repo '.venv\Scripts\python.exe')).Path
$base = (& $python -c "import sys; print(sys._base_executable)").Trim()
$gpuPreferences = 'HKCU:\Software\Microsoft\DirectX\UserGpuPreferences'

# Create the key only if missing: New-Item -Force on an existing key would recreate it
# and drop the preferences already stored for other programs.
if (-not (Test-Path $gpuPreferences)) { New-Item -Path $gpuPreferences | Out-Null }
# Windows hybrid-GPU policy: 2 means High performance (the NVIDIA adapter).
foreach ($exe in @($python, $base) | Select-Object -Unique) {
    New-ItemProperty -Path $gpuPreferences -Name $exe -PropertyType String -Value 'GpuPreference=2;' -Force | Out-Null
    Write-Host "GPU preference High performance: $exe"
}

& $python -u -m $Module @Rest
exit $LASTEXITCODE
