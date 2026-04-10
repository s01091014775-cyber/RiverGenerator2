@echo off
chcp 65001 >nul
echo ============================================
echo   Test: Foam
echo ============================================

set CKPT=checkpoints\foam\best_G.pth
if not exist "%CKPT%" (
    echo [ERROR] Checkpoint not found: %CKPT%
    echo Run train_foam.bat first.
    pause
    exit /b 1
)

echo Cleaning previous test results...
if exist "test_results\foam" rd /s /q "test_results\foam"
if exist "results\foam" rd /s /q "results\foam"

echo [1/2] Evaluating on validation set...
python inference.py --target foam --eval --checkpoint %CKPT% --limit 30
echo.
echo [2/2] Batch inference on training labels...
python inference.py --target foam --input_dir datasets\train_label --output_dir results\foam --checkpoint %CKPT% --limit 20
echo.
echo Done! Check: test_results\foam\ and results\foam\
pause
