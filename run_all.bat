@echo off
REM run_all.bat - full pipeline for Windows.
REM   1. create the virtual environment (.venv) if it does not exist
REM   2. install the pinned requirements
REM   3. download the NASA .mat files if they are missing (needs git)
REM   4. run the four scripts in order
REM   5. start the Streamlit dashboard (stop it with Ctrl+C)
REM Usage: double-click this file, or run "run_all.bat" in Command Prompt.

setlocal
REM Always run from the folder this .bat file is in
cd /d "%~dp0"

REM ---- pick a Python 3.12 or 3.13 interpreter (required by requirements.txt)
set "PY="
py -3.12 --version >nul 2>&1 && set "PY=py -3.12"
if not defined PY ( py -3.13 --version >nul 2>&1 && set "PY=py -3.13" )
if not defined PY (
    python -c "import sys; sys.exit(0 if (3,12) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo ERROR: Python 3.12 or 3.13 is required. Install it from https://www.python.org/downloads/
    goto :fail
)
%PY% --version

REM ---- 1. virtual environment
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment in .venv ...
    %PY% -m venv .venv || goto :fail
)
call ".venv\Scripts\activate.bat" || goto :fail

REM ---- 2. dependencies
python -m pip install --upgrade pip || goto :fail
python -m pip install -r requirements.txt || goto :fail

REM ---- 3. data (only if any of the four .mat files is missing)
if not exist data mkdir data
set "NEED_DATA="
for %%c in (B0005 B0006 B0007 B0018) do if not exist "data\%%c.mat" set "NEED_DATA=1"
if defined NEED_DATA (
    echo Downloading NASA battery data ...
    if exist "%TEMP%\nasa_battery_repo" rmdir /s /q "%TEMP%\nasa_battery_repo"
    git clone --depth 1 https://github.com/anirudhkhatry/SOH-prediction-using-NASA-Dataset "%TEMP%\nasa_battery_repo" || goto :fail
    for %%c in (B0005 B0006 B0007 B0018) do copy /y "%TEMP%\nasa_battery_repo\%%c.mat" data\ >nul || goto :fail
    rmdir /s /q "%TEMP%\nasa_battery_repo"
)

REM ---- 4. pipeline
echo === Step 1: prepare_data.py ===
python prepare_data.py || goto :fail
echo === Step 2: train_models.py ===
python train_models.py || goto :fail
echo === Step 3: train_lstm.py (takes a few minutes) ===
python train_lstm.py || goto :fail
echo === Step 4: make_plots.py ===
python make_plots.py || goto :fail

REM ---- 5. dashboard
REM --server.headless true skips Streamlit's first-run e-mail prompt, which
REM would otherwise wait for input. Open the URL below in your browser.
echo === Step 5: starting Streamlit - open http://localhost:8501 (Ctrl+C to stop) ===
streamlit run app.py --server.headless true
goto :eof

:fail
echo.
echo *** A step failed - see the error message above. ***
pause
exit /b 1
