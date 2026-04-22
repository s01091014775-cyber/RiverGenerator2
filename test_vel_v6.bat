@echo off
chcp 65001 >nul
echo ============================================
echo   Test: Velocity v6 (2.5x amplified data)
echo ============================================

set CKPT=checkpoints\vel_x25\best_G.pth
if not exist "%CKPT%" (
    echo [ERROR] Checkpoint not found: %CKPT%
    echo Run train_vel_v6.bat first.
    pause
    exit /b 1
)

echo Cleaning previous test results...
if exist "test_results\vel_x25" rd /s /q "test_results\vel_x25"
if exist "results\vel_x25" rd /s /q "results\vel_x25"

echo [1/2] Evaluating on validation set...
python inference.py --target vel_x25 --eval --checkpoint %CKPT% --limit 30 --output_act hardtanh --ngf 64

echo.
echo [2/2] Batch inference on training labels...
python inference.py --target vel_x25 --input_dir datasets\train_label --output_dir results\vel_x25 --checkpoint %CKPT% --limit 20 --output_act hardtanh --ngf 64

echo.
echo Done! Check: test_results\vel_x25\ and results\vel_x25\
pause
