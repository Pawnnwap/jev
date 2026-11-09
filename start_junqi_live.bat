@echo off
rem One-click launcher for the junqi live console (dark-chess game table):
rem   1) junqi live console on 127.0.0.1:8098  -> opens your browser, you press "start"
rem   2) llama.cpp engine on 127.0.0.1:8091    (only if not already listening)
rem   3) JEV adapter      on 127.0.0.1:8090    (/v1/systemone, only if not already listening)
rem
rem Flags:
rem   /NOJEV       skip the two model services - use it while the GPU is busy.
rem                 The console still works: RULE-AI and human need no model at all.
rem   /NOBROWSER   do not open the browser at the end.
rem
rem Machine-specific paths live in jev_config.bat (gitignored).
setlocal EnableDelayedExpansion
set "DIR=%~dp0"

set "WANT_JEV=1"
set "WANT_BROWSER=1"
if not "%~1"=="" for %%A in (%*) do (
  if /I "%%~A"=="/NOJEV" set "WANT_JEV=0"
  if /I "%%~A"=="/NOBROWSER" set "WANT_BROWSER=0"
)

set "PY=python"
set "SERVER=llama-server.exe"
set "MODEL=%DIR%jpt-4b.Q4_K_M.gguf"
set "MODEL_NAME=jpt-4b"
set "MODEL_TEMP=1.036"
set "NGL=auto"
if exist "%DIR%jev_config.bat" call "%DIR%jev_config.bat"

set "ENGINE_PORT=8091"
set "JEV_PORT=8090"
set "LIVE_PORT=8098"

call :listening %ENGINE_PORT%
if not errorlevel 1 set "ENGINE_UP=1" || set "ENGINE_UP=0"
call :listening %JEV_PORT%
if not errorlevel 1 set "JEV_UP=1" || set "JEV_UP=0"
call :listening %LIVE_PORT%
if not errorlevel 1 set "LIVE_UP=1" || set "LIVE_UP=0"

echo ============================================================
echo   junqi live console  -  dark-chess game table
echo ============================================================

rem ---------------- 1) console ----------------
if "%LIVE_UP%"=="1" goto live_kept
echo [1/3] starting junqi live console on port %LIVE_PORT%
start "junqi-live" /min "%PY%" "%DIR%junqi\live_play.py" --port %LIVE_PORT% --url http://127.0.0.1:%JEV_PORT%
goto wait_live
:live_kept
echo [1/3] console already listening on port %LIVE_PORT% - keep it

rem ---------------- wait for the console ----------------
:wait_live
set /a TRIES+=1
call :listening %LIVE_PORT%
if not errorlevel 1 goto live_ready
if !TRIES! GEQ 25 goto live_fail
timeout /t 1 /nobreak >nul
goto wait_live
:live_fail
echo [warn] console did not start. Run it manually:  python junqi\live_play.py --port %LIVE_PORT%
goto :eof
:live_ready

rem ---------------- 2/3) model services ----------------
if "%WANT_JEV%"=="0" goto skip_jev
if "%ENGINE_UP%"=="1" goto engine_kept
echo [2/3] starting llama-server on port %ENGINE_PORT% - model %MODEL_NAME%
start "llama-engine" /min "%SERVER%" -m "%MODEL%" --host 127.0.0.1 --port %ENGINE_PORT% -c 32768 -np 4 --jinja --alias %MODEL_NAME% -ngl %NGL%
goto start_adapter
:engine_kept
echo [2/3] llama-server already listening on port %ENGINE_PORT% - keep it

:start_adapter
if "%JEV_UP%"=="1" goto jev_kept
echo [3/3] starting JEV adapter on port %JEV_PORT%
start "jev-adapter" /min "%PY%" "%DIR%jev_adapter.py" --url http://127.0.0.1:%ENGINE_PORT% --model %MODEL_NAME% --temperature %MODEL_TEMP% --port %JEV_PORT%
goto wait_jev
:jev_kept
echo [3/3] JEV adapter already listening on port %JEV_PORT% - keep it

rem ---------------- wait for JEV (warn only) ----------------
:wait_jev
set /a TRIES2+=1
call :listening %JEV_PORT%
if not errorlevel 1 goto jev_ready
if !TRIES2! GEQ 45 (
  echo [warn] JEV service not reachable yet. Model load takes a while, or the GPU is busy.
  echo        The console still works: set both sides to RULE-AI or human.
  goto open_browser
)
timeout /t 3 /nobreak >nul
goto wait_jev
:jev_ready
echo [ok  ] JEV service is up on port %JEV_PORT%
goto open_browser

:skip_jev
echo [2/3] and [3/3] /NOJEV given - model services skipped, JEV modes disabled in the console

:open_browser
if "%WANT_BROWSER%"=="1" start "" "http://127.0.0.1:%LIVE_PORT%/"
echo.
echo Console: http://127.0.0.1:%LIVE_PORT%/
echo   both sides can be any of: JEV / RULE-AI / human
echo   per-ply thinking time is shown live for both sides
echo   stop console only:  stop_junqi_live.bat      stop console + services:  stop_junqi_live.bat /A
goto :eof

:listening
netstat -ano | findstr /R /C:":%~1 .*LISTENING" >nul
exit /b %errorlevel%