@echo off

cd /d %~dp0..

REM =========================
REM Подключить env (если есть)
REM =========================
if exist "%~dp0env.bat" call "%~dp0env.bat"

REM =========================
REM Python
REM =========================
if not defined PYTHON set "PYTHON=python"

echo Using Python: %PYTHON%

%PYTHON% -V >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found
    exit /b 1
)

REM =========================
REM Аргументы
REM =========================

set "INPUT=%1"
if "%INPUT%"=="" set "INPUT=data\raw\dataset_2025-03-01_2026-03-29_external.csv"

set "OUTPUT=%2"
if "%OUTPUT%"=="" set "OUTPUT=report\scores.csv"

if "%INPUT%"=="" (
    echo ERROR: INPUT is required
    echo Usage: run_score.bat input.csv output.csv
    exit /b 1
)

if "%OUTPUT%"=="" (
    echo ERROR: OUTPUT is required
    echo Usage: run_score.bat input.csv output.csv
    exit /b 1
)

echo Input: %INPUT%
echo Output: %OUTPUT%

REM =========================
REM Запуск
REM =========================

%PYTHON% src\score.py --input "%INPUT%" --output "%OUTPUT%"

echo Done.
pause
