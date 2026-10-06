@echo off
setlocal
cd /d "%~dp0"
if errorlevel 1 (
    echo No se pudo abrir la carpeta del proyecto.
    pause
    exit /b 1
)

set "CARGO_EXE="
if defined CARGO_HOME if exist "%CARGO_HOME%\bin\cargo.exe" set "CARGO_EXE=%CARGO_HOME%\bin\cargo.exe"
if not defined CARGO_EXE if exist "%USERPROFILE%\.cargo\bin\cargo.exe" set "CARGO_EXE=%USERPROFILE%\.cargo\bin\cargo.exe"
if not defined CARGO_EXE for /f "delims=" %%C in ('where cargo.exe 2^>nul') do if not defined CARGO_EXE set "CARGO_EXE=%%C"
if not defined CARGO_EXE (
    echo No se encontro cargo.exe. Revisa la instalacion de Rust.
    pause
    exit /b 1
)
for %%C in ("%CARGO_EXE%") do set "PATH=%%~dpC;%PATH%"

echo ========================================================
echo [FASE FINAL] Asalto al Numero de Ramsey R(5,5) N=43
echo Motor Hibrido: RTX 4060 (Prefiltro) + CPU 16 Hilos (Swarm)
echo ========================================================
echo.
echo El motor se ejecuta en esta ventana. Mantenla abierta.
echo Puedes monitorear el latido abriendo el archivo: r_5_5_43_final.log
echo.
"%CARGO_EXE%" run --release --bin hunt --features gpu -- r_5_5_43.cnf --ramsey-symm 43 --use-gpu --workers 16 --heartbeat 60 --cubes 1024 > r_5_5_43_final.log 2>&1
set "FINAL_EXIT_CODE=%ERRORLEVEL%"
echo.
echo Asalto finalizado con codigo %FINAL_EXIT_CODE%. Revisa el log para ver el veredicto.
pause
exit /b %FINAL_EXIT_CODE%
