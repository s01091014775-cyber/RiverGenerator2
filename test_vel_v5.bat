@echo off
chcp 65001 >nul
echo ============================================
echo   Test: Velocity v5 (angular + magnitude)
echo ============================================

set CKPT=checkpoints\vel\best_G.pth
if not exist "%CKPT%" (
    echo [ERROR] Checkpoint not found: %CKPT%
    echo Run train_vel_v5.bat first.
    pause
    exit /b 1
)

echo Cleaning previous test results...
if exist "test_results\vel" rd /s /q "test_results\vel"
if exist "results\vel" rd /s /q "results\vel"

echo [1/2] Evaluating on validation set...
python inference.py --target vel --eval --checkpoint %CKPT% --limit 30 --output_act hardtanh --ngf 96

echo.
echo [2/2] Batch inference on training labels...
python inference.py --target vel --input_dir datasets\train_label --output_dir results\vel --checkpoint %CKPT% --limit 20 --output_act hardtanh --ngf 96

echo.
echo Done! Check: test_results\vel\ and results\vel\
pause
