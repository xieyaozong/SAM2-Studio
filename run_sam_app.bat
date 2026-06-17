@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
    python sam2_studio.py %*
) else (
    echo Could not find .venv\Scripts\activate.bat. Falling back to system Python.
    python sam2_studio.py %*
)
pause
