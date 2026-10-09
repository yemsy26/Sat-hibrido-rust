param([switch]$SmokeTest)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$taskProject = Split-Path -Parent $PSScriptRoot
$taskMutex = [System.Threading.Mutex]::new($false, 'Local\NeuroRamseyK43Launcher')
$taskMutexHeld = $false
$taskExitCode = 1

function Assert-NoRunningSolver {
    $taskRunning = @(Get-CimInstance Win32_Process -Filter "Name = 'gpu_ramsey43.exe'")
    if ($taskRunning.Count -gt 0) {
        $taskPids = ($taskRunning | ForEach-Object { $_.ProcessId }) -join ', '
        throw "Ya existe un motor GPU activo: PID $taskPids. Cierralo con Ctrl+C en su ventana antes de iniciar otro."
    }
}

function Invoke-LoggedNative {
    param([string]$Executable, [string[]]$Arguments, [string]$LogPath)
    $taskWriter = [System.IO.StreamWriter]::new(
        $LogPath, $false, [System.Text.UTF8Encoding]::new($false))
    $taskWriter.AutoFlush = $true
    $taskOldPreference = $ErrorActionPreference
    try {
        # Windows PowerShell representa stderr nativo como ErrorRecord;
        # los latidos normales de CUDA tambien usan ese canal.
        $ErrorActionPreference = 'Continue'
        & $Executable @Arguments 2>&1 | ForEach-Object {
            $taskLine = $_.ToString()
            $taskWriter.WriteLine($taskLine)
            Write-Host $taskLine
        }
        return $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $taskOldPreference
        $taskWriter.Dispose()
    }
}

try {
    Set-Location -LiteralPath $taskProject
    try { $taskMutexHeld = $taskMutex.WaitOne(0) }
    catch [System.Threading.AbandonedMutexException] { $taskMutexHeld = $true }
    if (-not $taskMutexHeld) {
        throw 'El lanzador ya esta compilando o ejecutando el motor. No se iniciara otra copia.'
    }
    Assert-NoRunningSolver

    if (-not $env:TENSORRT_ROOT) { $env:TENSORRT_ROOT = 'C:\SDK\TensorRT-11.3.0.99' }
    if (-not $env:RAMSEY43_CUDA_ROOT) {
        $env:RAMSEY43_CUDA_ROOT = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6'
    }
    if (-not $env:RAMSEY43_MSVC_ROOT) {
        $env:RAMSEY43_MSVC_ROOT = 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207'
    }
    if (-not $env:RAMSEY43_WINDOWS_SDK_ROOT) {
        $env:RAMSEY43_WINDOWS_SDK_ROOT = 'C:\Program Files (x86)\Windows Kits\10'
    }
    if (-not $env:RAMSEY43_WINDOWS_SDK_VERSION) {
        $env:RAMSEY43_WINDOWS_SDK_VERSION = '10.0.26100.0'
    }
    $taskCompilerBin = Join-Path $env:RAMSEY43_MSVC_ROOT 'bin\Hostx64\x64'
    $taskSdk = $env:RAMSEY43_WINDOWS_SDK_ROOT
    $taskSdkVersion = $env:RAMSEY43_WINDOWS_SDK_VERSION
    $taskBase = Join-Path $taskProject 'results\k43_star_18.sat'
    $taskOnnx = Join-Path $taskProject 'neuro_ramsey_policy.onnx'
    $taskRequired = @(
        (Join-Path $env:TENSORRT_ROOT 'include\NvInfer.h'),
        (Join-Path $env:RAMSEY43_CUDA_ROOT 'bin\nvcc.exe'),
        (Join-Path $taskCompilerBin 'cl.exe'),
        (Join-Path $taskSdk "Include\$taskSdkVersion\ucrt\stdio.h"),
        $taskBase, $taskOnnx
    )
    foreach ($taskPath in $taskRequired) {
        if (-not (Test-Path -LiteralPath $taskPath -PathType Leaf)) {
            throw "Falta el archivo requerido: $taskPath"
        }
    }
    foreach ($taskDll in @('nvinfer_11.dll', 'nvonnxparser_11.dll')) {
        if (-not (Test-Path -LiteralPath (Join-Path $env:TENSORRT_ROOT "bin\$taskDll")) -and
            -not (Test-Path -LiteralPath (Join-Path $env:TENSORRT_ROOT "lib\$taskDll"))) {
            throw "Falta $taskDll en bin o lib de TENSORRT_ROOT: $env:TENSORRT_ROOT"
        }
    }

    $taskCargo = $null
    $taskCargoCandidates = @()
    if ($env:CARGO_HOME) { $taskCargoCandidates += Join-Path $env:CARGO_HOME 'bin\cargo.exe' }
    $taskCargoCandidates += Join-Path $env:USERPROFILE '.cargo\bin\cargo.exe'
    $taskCargoCommand = Get-Command cargo.exe -ErrorAction SilentlyContinue
    if ($taskCargoCommand) { $taskCargoCandidates += $taskCargoCommand.Source }
    foreach ($taskPath in $taskCargoCandidates) {
        if (Test-Path -LiteralPath $taskPath -PathType Leaf) { $taskCargo = $taskPath; break }
    }
    if (-not $taskCargo) { throw 'No se encontro cargo.exe. Revisa la instalacion de Rust.' }
    $env:PATH = "$taskCompilerBin;$(Split-Path -Parent $taskCargo);$env:RAMSEY43_CUDA_ROOT\bin;$env:TENSORRT_ROOT\bin;$env:TENSORRT_ROOT\lib;$env:PATH"
    $env:INCLUDE = "$env:RAMSEY43_MSVC_ROOT\include;$taskSdk\Include\$taskSdkVersion\ucrt;$taskSdk\Include\$taskSdkVersion\shared;$taskSdk\Include\$taskSdkVersion\um;$taskSdk\Include\$taskSdkVersion\winrt"
    $env:LIB = "$env:RAMSEY43_MSVC_ROOT\lib\x64;$taskSdk\Lib\$taskSdkVersion\ucrt\x64;$taskSdk\Lib\$taskSdkVersion\um\x64"
    $env:CC = Join-Path $taskCompilerBin 'cl.exe'
    $env:CXX = $env:CC
    $env:AR = Join-Path $taskCompilerBin 'lib.exe'

    $taskResults = Join-Path $taskProject 'results'
    [System.IO.Directory]::CreateDirectory($taskResults) | Out-Null
    $taskTarget = Join-Path $taskProject 'target\neuro-trt'
    [System.IO.Directory]::CreateDirectory($taskTarget) | Out-Null
    $taskExe = Join-Path $taskTarget 'release\gpu_ramsey43.exe'
    Write-Host '=============================================================='
    Write-Host 'NEURO-RAMSEY K43 - ESTRELLA 18 - TENSORRT REAL'
    Write-Host '=============================================================='
    Write-Host 'Compilando con --features neuro-trt en target\neuro-trt...'
    $taskBuildCode = Invoke-LoggedNative -Executable $taskCargo -Arguments @(
        'build', '--release', '--locked', '--features', 'neuro-trt', '--bin', 'gpu_ramsey43',
        '--target-dir', $taskTarget
    ) -LogPath (Join-Path $taskTarget 'neuro-launcher-build.log')
    if ($taskBuildCode -ne 0) { throw "La compilacion fallo con codigo $taskBuildCode. El motor no se ha iniciado." }
    Assert-NoRunningSolver

    $taskLog = Join-Path $taskResults 'asedio_neuro_k43.log'
    $taskModel = Join-Path $taskResults 'modelo_k43_neuro.sat'
    $taskEpochs = '0'
    $taskSteps = '1000'
    if ($SmokeTest) {
        $taskLog = Join-Path $taskTarget 'neuro-launcher-smoke.log'
        $taskModel = Join-Path $taskTarget 'neuro-launcher-smoke-model.sat'
        $taskEpochs = '50'
        $taskSteps = '100'
        Write-Host 'PRUEBA CORTA: 50 epocas, 100 pasos. UNKNOWN es un resultado permitido.'
    } else {
        if (Test-Path -LiteralPath $taskModel -PathType Leaf) {
            throw "Ya existe un modelo SAT: $taskModel. Auditalo antes de iniciar otra busqueda."
        }
        Write-Host 'Universos: 256. Warps por lote: 128. Semilla: 43.'
        Write-Host 'Epocas: SIN LIMITE; pasos por ronda: 1000.'
    }
    Write-Host "Modelo neuronal: $taskOnnx"
    Write-Host "Modelo SAT: $taskModel"
    Write-Host "Log: $taskLog"
    Write-Host 'Los errores y latidos aparecen aqui y se guardan en el log.'
    Write-Host 'Manten esta ventana abierta. Ctrl+C detiene el motor.'
    $taskExitCode = Invoke-LoggedNative -Executable $taskExe -Arguments @(
        '--base', $taskBase, '--neuro-model', $taskOnnx,
        '--prefix-bits', '8', '--batch-warps', '128', '--seed', '43',
        '--steps', $taskSteps, '--epochs', $taskEpochs, '--model', $taskModel
    ) -LogPath $taskLog
    Write-Host "Motor finalizado. Codigo de salida: $taskExitCode"
} catch {
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    $taskExitCode = 1
} finally {
    if ($taskMutexHeld) { $taskMutex.ReleaseMutex() }
    $taskMutex.Dispose()
}
exit $taskExitCode

