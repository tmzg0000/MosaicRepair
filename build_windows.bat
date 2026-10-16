@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
call start_windows.bat --build
exit /b %errorlevel%
