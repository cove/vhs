@echo off
REM Verifies all files on the drive against the checksum manifest (SHA3-256 or legacy BLAKE3)

set "SCRIPT_DIR=%~dp0"
python "%SCRIPT_DIR%step_8_verify_drive_checksum.py" %*
exit /b %ERRORLEVEL%
