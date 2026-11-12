@echo off
setlocal
cd /d "%~dp0"
echo SeedVR2 Batch Upscaler - Mise a jour / Update
echo Fermez l'application avant de continuer. / Close the app first.
set "PYRUN=python"
py -3 --version >nul 2>nul
if not errorlevel 1 set "PYRUN=py -3"
if exist ".venv\Scripts\python.exe" set PYRUN=".venv\Scripts\python.exe"
REM This whole block is parsed before update.py can replace this BAT file.
(
    %PYRUN% "scripts\update.py" --root "%CD%" %*
    if errorlevel 1 (
        echo.
        echo Mise a jour incomplete. Voir les erreurs ci-dessus.
        echo Update incomplete. See the errors above.
        pause
        exit /b 1
    )
    echo.
    pause
    exit /b 0
)
