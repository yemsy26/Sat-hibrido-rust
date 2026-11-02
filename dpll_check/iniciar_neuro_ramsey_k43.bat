@echo off
setlocal
title Neuro-Ramsey K43 - TensorRT - Inicio unico
color 0A
cd /d "%~dp0"
if errorlevel 1 (
    echo ERROR: no se pudo abrir la carpeta del proyecto.
    pause
    exit /b 1
)

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\iniciar_neuro_ramsey_k43.ps1" %*
set "NEURO_EXIT_CODE=%ERRORLEVEL%"
echo.
echo Codigo de salida: %NEURO_EXIT_CODE%
pause
exit /b %NEURO_EXIT_CODE%
