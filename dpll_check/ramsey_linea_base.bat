@echo off
rem Linea base de Ramsey R(5,5) sobre K_N: genera la formula y lanza el enjambre con
rem --ramsey-symm durante un tiempo acotado. Doble clic: N = 43 y 300 segundos.
rem
rem   ramsey_linea_base.bat [N] [segundos]
rem
rem Deja en esta carpeta:
rem   r_5_5_N.cnf      la formula (si ya existe se reutiliza)
rem   r_5_5_N.log      la salida de hunt, tal cual
rem   r_5_5_N.mem.log  memoria del proceso cada 15 segundos
rem
rem Cada 30 segundos hunt imprime una linea `c status:` con los conflictos acumulados, el
rem ritmo desde la linea anterior y los cubos cerrados; la ultima sale al agotar el limite.
setlocal
cd /d "%~dp0"

set N=%~1
if "%N%"=="" set N=43
set LIMITE=%~2
if "%LIMITE%"=="" set LIMITE=300
set WORKERS=%NUMBER_OF_PROCESSORS%
set CNF=r_5_5_%N%.cnf
set LOG=r_5_5_%N%.log
set MEM=r_5_5_%N%.mem.log

rem Un doble clic no siempre hereda el PATH de Rust: se busca cargo en su sitio habitual
rem y, si tampoco aparece, valen los binarios ya compilados.
where cargo >nul 2>nul
if errorlevel 1 if exist "%USERPROFILE%\.cargo\bin\cargo.exe" set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
where cargo >nul 2>nul
if errorlevel 1 goto :sin_cargo
echo == Compilando gen_ramsey y hunt
cargo build --release --bin gen_ramsey --bin hunt || goto :error
goto :compilado

:sin_cargo
if not exist target\release\hunt.exe goto :faltan
if not exist target\release\gen_ramsey.exe goto :faltan
echo == cargo no esta en el PATH: se usan los binarios ya compilados en target\release

:compilado

if exist "%CNF%" (
    echo == %CNF% ya existe: se reutiliza
) else (
    echo == Generando K_%N% en %CNF%
    target\release\gen_ramsey.exe %N% "%CNF%" || goto :error
)
set CABECERA=
for /f "delims=" %%H in ('findstr /b /c:"p cnf " "%CNF%"') do set CABECERA=%%H
for %%F in ("%CNF%") do echo    %CNF%: %%~zF bytes, cabecera "%CABECERA%"
if "%N%"=="43" if not "%CABECERA%"=="p cnf 903 1925196" goto :mal_cnf

if exist "%LOG%" del "%LOG%"
if exist "%MEM%" del "%MEM%"

rem Vigilante de memoria: espera a que hunt arranque y lo muestrea hasta que termine.
start "" /b powershell -NoProfile -Command "$n = 0; while (-not (Get-Process hunt -ErrorAction SilentlyContinue) -and $n -lt 60) { Start-Sleep -Milliseconds 500; $n++ }; while ($p = Get-Process hunt -ErrorAction SilentlyContinue | Select-Object -First 1) { $linea = '   {0:HH:mm:ss}  memoria {1,5:N2} GiB  pico {2,5:N2} GiB  RAM libre {3,5:N1} GiB' -f (Get-Date), ($p.WorkingSet64 / 1GB), ($p.PeakWorkingSet64 / 1GB), ((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB); Write-Host $linea; Add-Content -Path '%MEM%' -Value $linea; Start-Sleep -Seconds 15 }"

echo == Enjambre: %WORKERS% workers, limite %LIMITE% s. Salida en %LOG%
powershell -NoProfile -Command "cmd /c 'target\release\hunt.exe %CNF% --ramsey-symm %N% --workers %WORKERS% --timeout %LIMITE% --heartbeat 30 2>&1' | ForEach-Object { Write-Host $_; Add-Content -Path '%LOG%' -Value $_ }; exit $LASTEXITCODE"
set CODIGO=%ERRORLEVEL%

echo.
echo == Resultado
if "%CODIGO%"=="124" echo    Limite de %LIMITE% s alcanzado: s UNKNOWN, codigo 124. Es el final esperado.
if "%CODIGO%"=="10" echo    SATISFACIBLE, codigo 10.
if "%CODIGO%"=="20" echo    INSATISFACIBLE, codigo 20.
if not "%CODIGO%"=="124" if not "%CODIGO%"=="10" if not "%CODIGO%"=="20" echo    ERROR: hunt termino con codigo %CODIGO%. Mira %LOG%.
findstr /c:"c symm:" "%LOG%"
set ESTADO=
for /f "delims=" %%L in ('findstr /c:"c status:" "%LOG%"') do set ESTADO=%%L
if defined ESTADO echo %ESTADO%
findstr /i /c:"memory allocation" "%LOG%" >nul && echo    FALTA DE MEMORIA: mira %LOG%.
set ULTIMA=
if exist "%MEM%" for /f "delims=" %%L in (%MEM%) do set ULTIMA=%%L
if defined ULTIMA echo    Ultima muestra de memoria: %ULTIMA%
echo    Salida de hunt: %LOG%
if exist "%MEM%" echo    Muestras de memoria: %MEM%
pause
exit /b %CODIGO%

:mal_cnf
echo ERROR: %CNF% no es R(5,5) con N = 43. Borralo y vuelve a lanzar.
pause
exit /b 1

:faltan
echo ERROR: no se encuentra cargo ni hay binarios compilados en target\release.
pause
exit /b 1

:error
echo ERROR: no se pudo compilar o generar la formula.
pause
exit /b 1
