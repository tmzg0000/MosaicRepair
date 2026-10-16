@echo off
setlocal EnableExtensions DisableDelayedExpansion
echo ============================================================
echo   MosaicRepair - Windows startup
echo ============================================================
chcp 65001 >nul
title MosaicRepair - Startup
cd /d "%~dp0"
if errorlevel 1 goto folder_failed
echo [1/4] Looking for 64-bit Python 3.11, 3.12 or 3.13...
echo Log: "%CD%\startup.log"
echo.
> "startup.log" echo MosaicRepair startup: %date% %time%
if errorlevel 1 goto folder_failed
if not exist "bootstrap.py" goto files_missing
if not exist "main.py" goto files_missing
if not exist "requirements.txt" goto files_missing
set "MOSAIC_PYTHON="
if exist ".venv\Scripts\python.exe" call :try_path "%CD%\.venv\Scripts\python.exe"
if exist ".venv-win\Scripts\python.exe" call :try_path "%CD%\.venv-win\Scripts\python.exe"
if defined MOSAIC_PYTHON goto launch
where py >> "startup.log" 2>&1
if errorlevel 1 goto search_path
echo Checking registered Python installations...
py -0p >> "startup.log" 2>&1
for %%V in (3.12 3.13 3.11) do call :try_launcher %%V
if defined MOSAIC_PYTHON goto launch
:search_path
for /f "delims=" %%P in ('where python 2^>nul') do call :try_path "%%P"
if defined MOSAIC_PYTHON goto launch
echo ERROR: A supported 64-bit Python installation was not found.
echo Install 64-bit Python 3.12, then run this file again.
echo Download: https://www.python.org/downloads/windows/
echo ERROR: Supported Python not found. >> "startup.log"
goto failed
:launch
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
%MOSAIC_PYTHON% -u "bootstrap.py" %*
set "MOSAIC_EXIT=%errorlevel%"
if not "%MOSAIC_EXIT%"=="0" goto failed
if "%~1"=="--prepare-only" exit /b 0
if "%~1"=="--smoke-test" exit /b 0
echo.
echo Finished. If no app window appeared, send startup.log for diagnosis.
pause
exit /b 0
:try_launcher
if defined MOSAIC_PYTHON exit /b 0
echo Checking Python %~1...
py -%~1 -c "import sys,struct; print(sys.executable,sys.version); sys.exit(0 if (3,11)<=sys.version_info[:2]<=(3,13) and struct.calcsize('P')==8 else 1)" >> "startup.log" 2>&1
if errorlevel 1 exit /b 0
set "MOSAIC_PYTHON=py -%~1"
exit /b 0
:try_path
if defined MOSAIC_PYTHON exit /b 0
echo("%~1"| findstr /i /l /c:"\Microsoft\WindowsApps\python.exe" >nul
if not errorlevel 1 exit /b 0
echo Checking "%~1"...
"%~1" -c "import sys,struct; print(sys.executable,sys.version); sys.exit(0 if (3,11)<=sys.version_info[:2]<=(3,13) and struct.calcsize('P')==8 else 1)" >> "startup.log" 2>&1
if errorlevel 1 exit /b 0
set MOSAIC_PYTHON="%~1"
exit /b 0
:files_missing
echo ERROR: Required project files are missing. Extract the whole ZIP first.
echo ERROR: Project files missing. >> "startup.log"
goto failed
:folder_failed
echo ERROR: Cannot use this folder. Extract the ZIP to a writable local folder.
pause
exit /b 1
:failed
echo.
echo Startup did not finish. Details are in startup.log.
echo Keep this window open and send the log or the last error message.
pause
exit /b 1
