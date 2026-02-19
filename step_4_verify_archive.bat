@echo on
REM Verifies all files in the archive against the checksum manifest (SHA3-256 or legacy BLAKE3)

set "SCRIPT_DIR=%~dp0"
python "%SCRIPT_DIR%step_4_verify_archive.py" %*
exit /b %ERRORLEVEL%
