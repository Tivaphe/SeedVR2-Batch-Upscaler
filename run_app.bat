@echo off
REM Launches the SeedVR2 Batch Upscaler graphical interface.
setlocal
cd /d %~dp0
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat

REM Quick check: gradio must be present in THIS venv.
python -c "import gradio" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] gradio is not installed in the .venv environment.
    echo   Run install.bat again - or fix it manually:
    echo     .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

REM GPU status reminder (torch + driver) before opening the interface.
python -m seedvr2_upscaler.gpu_check --summary
if errorlevel 1 (
    echo.
    echo [WARNING] No CUDA GPU detected: SeedVR2 will not start.
    echo   Run: python check_install.py
    echo.
)

python app.py %*
pause
