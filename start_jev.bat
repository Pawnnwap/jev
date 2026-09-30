@echo off
setlocal
set "DIR=%~dp0"
rem Machine-specific paths live in jev_config.bat (local only, gitignored; demo template:
rem jev_config.bat.example). Without it, PATH defaults + the jpt-4b weights are used.
set "PY=python"
set "SERVER=llama-server.exe"
set "MODEL=%DIR%jpt-4b.Q4_K_M.gguf"
set "MODEL_NAME=jpt-4b"
set "MODEL_TEMP=1.036"
set "NGL=auto"
if exist "%DIR%jev_config.bat" call "%DIR%jev_config.bat"
set "ENGINE_PORT=8091"
set "JEV_PORT=8090"

rem Jev-like decision service: llama.cpp engine (minimized window) + /v1/systemone adapter (this console)
rem Stop: run stop_jev.bat, or Ctrl+C here then stop_jev.bat for the engine.

start "llama-server (%MODEL_NAME%)" /min "%SERVER%" -m "%MODEL%" --host 127.0.0.1 --port %ENGINE_PORT% -c 32768 -np 4 --jinja --alias %MODEL_NAME% -ngl %NGL%

"%PY%" "%DIR%jev_adapter.py" --url http://127.0.0.1:%ENGINE_PORT% --model %MODEL_NAME% --temperature %MODEL_TEMP% --port %JEV_PORT%
