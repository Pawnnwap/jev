@echo off
setlocal
set "DIR=%~dp0"
rem Machine-specific paths live in jev_config.bat (local only, gitignored; demo template:
rem jev_config.bat.example). Without it, PATH defaults are used.
set "PY=python"
set "SERVER=llama-server.exe"
if exist "%DIR%jev_config.bat" call "%DIR%jev_config.bat"
set "ENGINE_PORT=8091"
set "JEV_PORT=8090"

rem Jev-like decision service: llama.cpp engine (minimized window) + /v1/systemone adapter (this console)
rem Stop: run stop_jev.bat, or Ctrl+C here then stop_jev.bat for the engine.

start "jpt-4b llama-server" /min "%SERVER%" -m "%DIR%jpt-4b.Q4_K_M.gguf" --host 127.0.0.1 --port %ENGINE_PORT% -c 32768 -np 4 --jinja --alias jpt-4b

"%PY%" "%DIR%jev_adapter.py" --url http://127.0.0.1:%ENGINE_PORT% --model jpt-4b --temperature 1.036 --port %JEV_PORT%
