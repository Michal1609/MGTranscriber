@echo off
setlocal
cd /d "%~dp0"

if "%~1"=="" goto help
if "%~1"=="/?" goto help
if /i "%~1"=="-h" goto help
if /i "%~1"=="--help" goto help

set PYTHONIOENCODING=utf-8
set "UVGROUPS="
if exist .torch-cpu set "UVGROUPS=--no-group cu128 --group cpu"
uv run --no-dev %UVGROUPS% transcribe.py %*
exit /b %errorlevel%

:help
echo Prepis videa s mluvcimi (transcribe.py)
echo.
echo Pouziti:  start.bat "URL nebo soubor" [dalsi volby transcribe.py]
echo.
echo Priklady:
echo   start.bat "https://www.youtube.com/watch?v=..."
echo   start.bat "https://www.youtube.com/watch?v=..." --speakers 2 --language en
echo   start.bat rozhovor.mp4 -o D:\Prepisy
echo.
echo URL davejte do uvozovek (kvuli znakum ^& a ?).
echo Volby: --language auto^|en^|cs^|..., --speakers N, --model, --output DIR, --no-diarize
echo Vysledek: output\^<nazev videa^>\prepis.txt
exit /b 1
