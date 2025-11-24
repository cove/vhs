@echo off
REM Workaround for PowerShell script execution policy restrictions
REM Usage: dp0step_4_restore_and_split.bat archive.mkv
powershell.exe -NoExit -ExecutionPolicy Bypass -File "%~dp0step_4_restore_and_split.ps1" %*
exit /b %errorlevel%
