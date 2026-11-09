@echo off
rem Stop the junqi live console (and, with /A, the JEV engine + adapter too).
setlocal
set "PORT=8098"
set "KILL_ALL=0"
if /I "%~1"=="/A" set "KILL_ALL=1"

for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":%PORT% .*LISTENING"') do (
  echo stopping junqi live console, pid %%P
  taskkill /PID %%P /F >nul 2>&1
)

if "%KILL_ALL%"=="1" (
  echo stopping llama-server.exe
  taskkill /IM llama-server.exe /F >nul 2>&1
  for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":8090 .*LISTENING"') do (
    echo stopping JEV adapter, pid %%P
    taskkill /PID %%P /F >nul 2>&1
  )
)
echo done.