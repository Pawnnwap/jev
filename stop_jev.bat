@echo off
rem Stop the jpt-4b llama.cpp engine started by start_jev.bat (the adapter dies with its console).
taskkill /IM llama-server.exe /F
