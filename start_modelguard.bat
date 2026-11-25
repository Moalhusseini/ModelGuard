@echo off
setlocal
cd /d "%~dp0"
title ModelGuard Launcher

echo ==========================================
echo             ModelGuard
echo ==========================================

where py >nul 2>nul
if errorlevel 1 (
    echo Python launcher was not found. Install Python 3.12 and enable Add to PATH.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating project virtual environment...
    py -3.12 -m venv .venv
    if errorlevel 1 (
        echo Could not create the Python 3.12 environment.
        pause
        exit /b 1
    )
)

if not exist ".venv\.modelguard_requirements_installed" (
    echo Installing ModelGuard runtime dependencies...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    if errorlevel 1 goto :install_error
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto :install_error
    echo installed>".venv\.modelguard_requirements_installed"
)

where ollama >nul 2>nul
if errorlevel 1 (
    echo Ollama was not found. Install Ollama from https://ollama.com/download
    pause
    exit /b 1
)

powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing http://localhost:11434/api/tags -TimeoutSec 2 ^| Out-Null; exit 0 } catch { exit 1 }"
if errorlevel 1 (
    echo Starting Ollama...
    start "Ollama Service" /min cmd /c "ollama serve"
    timeout /t 5 /nobreak >nul
)

powershell -NoProfile -Command "try { $r=Invoke-RestMethod http://localhost:11434/api/tags -TimeoutSec 5; if (($r.models.name -like 'qwen2.5:1.5b*').Count -gt 0) { exit 0 } else { exit 1 } } catch { exit 1 }"
if errorlevel 1 (
    echo The required local model qwen2.5:1.5b was not found.
    echo Run this command once in a terminal: ollama pull qwen2.5:1.5b
    pause
    exit /b 1
)

echo Starting Streamlit...
".venv\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 (
    echo Streamlit exited with an error.
    pause
    exit /b 1
)
goto :eof

:install_error
echo Dependency installation failed. Check the error above and your internet connection.
pause
exit /b 1
