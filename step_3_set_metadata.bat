@echo off
REM Workaround for PowerShell script execution policy restrictions
REM Usage: step_3_set_metadata.bat captured.mkv
powershell.exe -NoExit -ExecutionPolicy Bypass -File "%~dp0step_3_set_metadata.ps1" %*
exit /b %errorlevel%
