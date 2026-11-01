@echo off
rem Installs, repairs (-Repair) or uninstalls (-Uninstall) the app. Options: see install.ps1.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
set "code=%errorlevel%"
if not "%code%"=="0" pause
exit /b %code%
