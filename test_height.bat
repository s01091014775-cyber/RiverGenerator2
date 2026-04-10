@echo off
chcp 65001 >nul
echo ============================================
echo   Test: Water Height
echo ============================================

set CKPT=checkpoints\height\best_G.pth
if not exist "%CKPT%" (
    echo [ERROR] Checkpoint not found: %CKPT%
    echo Run train_height.bat first.
    pause
    exit /b 1
)

echo Cleaning previous test results...
if exist "test_results\height" rd /s /q "test_results\height"
if exist "results\height" rd /s /q "results\height"

echo [1/2] Evaluating on validation set...
python inference.py --target height --eval --checkpoint %CKPT% --limit 30
echo.
echo [2/2] Batch inference on training labels...
python inference.py --target height --input_dir datasets\train_label --output_dir results\height --checkpoint %CKPT% --limit 20
echo.
echo Done! Check: test_results\height\ and results\height\
pause
