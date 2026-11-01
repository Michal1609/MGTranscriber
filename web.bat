@echo off
setlocal
cd /d "%~dp0"
rem Spusti webove rozhrani a otevre prohlizec. Dalsi volby se predaji serveru (napr. --host 0.0.0.0 --port 8080).
rem install.ps1 leaves .torch-cpu on machines without an NVIDIA GPU: keep uv on the CPU build of PyTorch
set "UVGROUPS="
if exist .torch-cpu set "UVGROUPS=--no-group cu128 --group cpu"
uv run --no-dev %UVGROUPS% server.py --open %*
