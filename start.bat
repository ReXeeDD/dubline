@echo off
setlocal
pushd "%~dp0" || exit /b 1
title Dubline

set "PY=.venv\Scripts\python.exe"
if exist "%PY%" goto launch

set "PY=py -3"
%PY% --version >nul 2>&1
if not errorlevel 1 goto launch
set "PY=python"
%PY% --version >nul 2>&1
if not errorlevel 1 goto launch

echo   Install Python 3.10 or newer from https://www.python.org/downloads/
echo   and tick "Add python.exe to PATH" in the installer, then try again.
set "RUN_STATUS=1"
goto done

:launch
if "%PY%"==".venv\Scripts\python.exe" (
    "%PY%" launcher.py %*
) else (
    %PY% launcher.py %*
)
set "RUN_STATUS=%ERRORLEVEL%"

:done
if not "%RUN_STATUS%"=="0" pause
popd
exit /b %RUN_STATUS%
