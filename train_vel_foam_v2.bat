@echo off
chcp 65001 >nul

echo ============================================
echo   [1/2] Train: Velocity v2 (optimized)
echo ============================================
python train.py --target vel --epochs 300 --batch 4 --img_size 256 --lr_g 1e-4 --lr_d 4e-4 --lambda_l1 20 --lambda_vgg 0 --lambda_fm 10 --lambda_grad 5 --fg_weight 15 --scheduler cosine --patience 30 --save_every 20
echo.

echo ============================================
echo   [2/2] Train: Foam v2 (optimized)
echo ============================================
python train.py --target foam --epochs 300 --batch 4 --img_size 256 --lr_g 1e-4 --lr_d 4e-4 --lambda_l1 20 --lambda_vgg 5 --lambda_fm 10 --lambda_grad 0 --fg_weight 10 --scheduler cosine --patience 25 --save_every 20
echo.

echo ============================================
echo   Vel + Foam training complete!
echo ============================================
pause
