@echo off
powershell.exe -NoExit -ExecutionPolicy Bypass -File "%~dp0step_3_set_metadata.ps1" %*
exit /b %errorlevel%

