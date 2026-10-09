@echo off
setlocal
set "PYTHONDONTWRITEBYTECODE=1"
cd /d "%~dp0"
if errorlevel 1 goto :error_carpeta
if not exist "centinela_k43.py" goto :error_script
set "CENTINELA_PYTHON="
set "CENTINELA_PYTHON_ARGS="
if defined PYTHON_EXE call :probar_python "%PYTHON_EXE:"=%" ""
if defined CENTINELA_PYTHON goto :dependencias
for /f "delims=" %%P in ('where py.exe 2^>nul') do (
    if not defined CENTINELA_PYTHON call :probar_python "%%P" "-3"
)
if defined CENTINELA_PYTHON goto :dependencias
for /f "delims=" %%P in ('where python.exe 2^>nul') do (
    if not defined CENTINELA_PYTHON call :probar_python "%%P" ""
)
if defined CENTINELA_PYTHON goto :dependencias
call :probar_python "%USERPROFILE%\Miniconda3\python.exe" ""
if not defined CENTINELA_PYTHON goto :error_python

:dependencias
echo Python: "%CENTINELA_PYTHON%" %CENTINELA_PYTHON_ARGS%
"%CENTINELA_PYTHON%" %CENTINELA_PYTHON_ARGS% -c "import networkx, matplotlib" >nul 2>&1
if not errorlevel 1 goto :ejecutar
echo Preparando networkx y matplotlib. Espera un momento...
"%CENTINELA_PYTHON%" %CENTINELA_PYTHON_ARGS% -m pip install --quiet --disable-pip-version-check networkx matplotlib >nul 2>&1
if errorlevel 1 goto :error_dependencias
"%CENTINELA_PYTHON%" %CENTINELA_PYTHON_ARGS% -c "import networkx, matplotlib" >nul 2>&1
if errorlevel 1 goto :error_dependencias

:ejecutar
set "CENTINELA_MODELO=%CD%\results\modelo_k43_neuro.sat"
echo Motor neuronal actual: --model results\modelo_k43_neuro.sat
echo El centinela espera ese archivo y audita los dos colores cuando aparezca.
echo No necesita reiniciar el motor ni un archivo de log de la consola.
echo Iniciando el centinela K43...
"%CENTINELA_PYTHON%" %CENTINELA_PYTHON_ARGS% "centinela_k43.py" --log "%CENTINELA_MODELO%" --model-path "%CENTINELA_MODELO%" %*
set "CENTINELA_CODIGO=%ERRORLEVEL%"
if not "%CENTINELA_CODIGO%"=="0" echo ERROR: El centinela termino con codigo %CENTINELA_CODIGO%.
goto :fin

:error_carpeta
echo ERROR: No se puede abrir la carpeta del centinela.
goto :fallo
:error_script
echo ERROR: Falta centinela_k43.py junto a este archivo BAT.
goto :fallo
:error_python
echo ERROR: No se encontro un Python 3 usable. Instala Python o define PYTHON_EXE.
goto :fallo
:error_dependencias
echo ERROR: No se pudieron preparar networkx y matplotlib con el Python indicado.
echo Revisa pip y la conexion a Internet, o define PYTHON_EXE con otro Python 3.
:fallo
set "CENTINELA_CODIGO=1"
:fin
echo.
pause
exit /b %CENTINELA_CODIGO%

:probar_python
"%~1" %~2 -c "import sys; assert sys.version_info.major == 3" >nul 2>&1
if errorlevel 1 exit /b 0
set "CENTINELA_PYTHON=%~1"
set "CENTINELA_PYTHON_ARGS=%~2"
exit /b 0
