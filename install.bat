@echo off
REM ============================================================
REM  Setup - SeedVR2 Batch Upscaler (Windows)
REM  Requires Python 3.11+ and git.
REM ============================================================
setlocal
cd /d %~dp0

echo [1/6] Creating the virtual environment...
REM Prefer 3.12 then 3.11 (best CUDA wheel availability for torch),
REM fall back to the default python otherwise.
set "PYLAUNCH="
py -3.12 --version >nul 2>nul && set "PYLAUNCH=py -3.12"
if not defined PYLAUNCH (
    py -3.11 --version >nul 2>nul && set "PYLAUNCH=py -3.11"
)
if not defined PYLAUNCH set "PYLAUNCH=python"
echo   ^> %PYLAUNCH%
if not exist .venv (
    %PYLAUNCH% -m venv .venv
) else (
    echo   .venv already exists, reusing it.
)
call .venv\Scripts\activate.bat
echo   Python in use:
python --version

echo [2/6] Upgrading pip (venv-aware syntax)...
python -m pip install --upgrade pip setuptools wheel

echo [3/6] PyTorch (trying several CUDA indexes: cu130, cu128, then PyPI)...
REM cu130 = torch 2.9-2.13 (CUDA 13, needs NVIDIA driver >= 580 on Windows,
REM wheels for Python 3.10-3.14+) ; cu128 = older torch (<= 2.11, CUDA 12.8,
REM for older drivers) ; plain PyPI wheels as a last resort (CPU-only on
REM Windows — SeedVR2 will refuse to start, see check_install.py).
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
if errorlevel 1 python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 python -m pip install torch torchvision
if errorlevel 1 (
    echo [ERROR] No torch build matches your Python.
    echo   Your Python may be too new for prebuilt CUDA wheels:
    echo   recreate the venv with Python 3.12: py -3.12 -m venv .venv
    echo   then run install.bat again.
)

echo [4/6] Application dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependencies failed - see the pip message above.
    pause
    exit /b 1
)
python -m pip install mediapy

echo [5/6] Cloning the official SeedVR2 repository...
if not exist SeedVR (
    git clone https://github.com/ByteDance-Seed/SeedVR
) else (
    echo   Repository already present, skipping.
)
REM Official repo dependencies: we do NOT follow its requirements.txt
REM literally - it pins torch==2.3.0, which is incompatible with recent
REM Python versions and would downgrade your CUDA torch. We install a
REM curated unpinned selection instead (apex stays separate: prebuilt
REM wheels are linked in the repo README; it cannot compile on Windows).
python -m pip install "diffusers>=0.29" "transformers>=4.38" "rotary-embedding-torch>=0.5" opencv-python mediapy
if errorlevel 1 (
    echo [WARNING] Some official-repo dependencies failed
    echo   (apex is the usual suspect on Windows). Install the prebuilt
    echo   apex wheel linked in the ByteDance-Seed/SeedVR README.
)

echo [6/6] Models folder + sanity check...
if not exist models mkdir models
python check_install.py

echo.
echo Setup complete!
echo  - Drop SeedVR2 weights into models\ (or use the "Download" button
echo    of the interface).
echo  - Start the app with run_app.bat
pause
