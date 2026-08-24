@echo off
setlocal
cd /d "%~dp0"
title Dubline

REM ---------------------------------------------------------------------------
REM  Double-click this file. On the first run it builds a private Python
REM  environment next to the app and downloads ffmpeg into bin\; on every run
REM  after that it just starts the server and opens the browser.
REM
REM  Nothing is installed system-wide and nothing is written outside this
REM  folder - including pip's own scratch space, which is pointed at data\tmp
REM  below so a full system drive cannot break the install.
REM ---------------------------------------------------------------------------

if not exist "data\tmp" mkdir "data\tmp"
set "TMP=%CD%\data\tmp"
set "TEMP=%CD%\data\tmp"
set "PIP_CACHE_DIR=%CD%\data\cache\pip"

set "PY=.venv\Scripts\python.exe"
if exist "%PY%" goto ready

echo.
echo   Dubline - first run
echo   Setting up. This takes a few minutes and only happens once.
echo.

REM Prefer the py launcher, fall back to whatever python is on PATH.
set "BOOT="
py -3 --version >nul 2>&1
if not errorlevel 1 set "BOOT=py -3"
if not defined BOOT (
    python --version >nul 2>&1
    if not errorlevel 1 set "BOOT=python"
)
if not defined BOOT goto no_python

echo   [1/3] Creating the Python environment...
%BOOT% -m venv ".venv"
if not exist "%PY%" goto no_venv

echo   [2/3] Installing packages...
"%PY%" -m pip install --upgrade pip --quiet --disable-pip-version-check
"%PY%" -m pip install -r requirements.txt --quiet --disable-pip-version-check
if errorlevel 1 goto no_deps

:ready
if exist "bin\ffmpeg.exe" goto serve
echo   [3/3] Downloading ffmpeg...
"%PY%" setup.py
if not exist "bin\ffmpeg.exe" goto no_ffmpeg

:serve
echo.
"%PY%" run.py --open %*
if errorlevel 1 pause
exit /b 0

REM ------------------------------------------------------------- failures ---
:no_python
echo.
echo   Python was not found.
echo.
echo   Install Python 3.10 or newer from https://www.python.org/downloads/
echo   and tick "Add python.exe to PATH" in the installer, then run this
echo   file again.
echo.
pause
exit /b 1

:no_venv
echo.
echo   Could not create the Python environment in .venv
echo.
echo   This usually means the Python install is missing the venv module.
echo   Reinstall Python from python.org (the Microsoft Store build is the
echo   usual culprit), then delete the .venv folder and run this again.
echo.
pause
exit /b 1

:no_deps
echo.
echo   Installing the Python packages failed - see the messages above.
echo   The most common cause is no internet connection or a proxy.
echo.
echo   You can retry by deleting the .venv folder and running this again.
echo.
pause
exit /b 1

:no_ffmpeg
echo.
echo   ffmpeg could not be downloaded automatically.
echo.
echo   Get a build from https://github.com/BtbN/FFmpeg-Builds/releases
echo   (ffmpeg-master-latest-win64-gpl.zip) and copy ffmpeg.exe and
echo   ffprobe.exe out of its bin\ folder into:
echo       %CD%\bin
echo   then run this file again.
echo.
pause
exit /b 1
