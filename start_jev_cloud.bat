@echo off
setlocal
set "DIR=%~dp0"
rem Cloud System One lane: no local weights, no llama-server. The remote
rem spark-x2.6 chat endpoint serves /v1/systemone via logprobs + prefill.
rem Stop the local lane first if it owns port 8090 (stop_jev.bat).
set "JEV_CLOUD_PY=python"
if exist "%DIR%jev_config_cloud.bat" call "%DIR%jev_config_cloud.bat"
if not defined SPARK_API_KEY (
  echo ERROR: SPARK_API_KEY is not set. Copy jev_config_cloud.bat.example to jev_config_cloud.bat.
  exit /b 1
)
echo Starting cloud System One service on %JEV_CLOUD_HOST%:%JEV_CLOUD_PORT% ...
"%JEV_CLOUD_PY%" "%DIR%jev_cloud_adapter.py"
endlocal
