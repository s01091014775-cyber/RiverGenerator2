@echo off
chcp 65001 >nul
echo ============================================
echo   Train: Foam (terrain -> foam/spray)
echo ============================================
python train.py --target foam --epochs 200 --batch 4 --img_size 256 --lambda_l1 20 --lambda_vgg 5 --fg_weight 10 --save_every 10
echo.
echo Training complete. Checkpoints in: checkpoints\foam\
pause
