@echo off
REM Schedule Vector Store Regression Tests to run nightly
REM This script creates a Windows scheduled task to run regression tests

echo Creating scheduled task for Vector Store regression tests...

REM Get the full path to the Python script
set SCRIPT_PATH=%~dp0regression_test_vectorstore.py
set PYTHON_PATH=python

REM Set up the task to run at 2 AM daily
schtasks /create /tn "ME2AI MCP Vector Store Regression Tests" ^
         /tr "%PYTHON_PATH% %SCRIPT_PATH% --slack" ^
         /sc daily ^
         /st 02:00 ^
         /ru SYSTEM ^
         /rl HIGHEST ^
         /f

if %ERRORLEVEL% EQU 0 (
    echo Scheduled task created successfully.
    echo Tests will run daily at 2:00 AM.
) else (
    echo Failed to create scheduled task. Error code: %ERRORLEVEL%
    echo Please run this script with administrator privileges.
)

echo.
echo To modify task settings, use Windows Task Scheduler.
echo To run tests manually: python %SCRIPT_PATH% --slack
echo.
