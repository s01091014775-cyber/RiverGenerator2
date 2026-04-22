@echo off
chcp 65001 >nul

echo ============================================
echo   Train: Velocity v5 (angular + magnitude)
echo   Changes from v4:
echo   - Angular velocity loss (lambda=8)
echo   - Magnitude loss (lambda=5)
echo   - fg_weight 15 -> 25
echo   - lambda_grad 5 -> 8
echo   - ngf 64 -> 96
echo   - Train from scratch (ngf mismatch with v4)
echo ============================================

python train.py --target vel --epochs 400 --batch 4 --img_size 256 --lr_g 8e-5 --lr_d 3e-4 --lambda_l1 25 --lambda_vgg 5 --lambda_fm 10 --lambda_grad 8 --lambda_angular 8 --lambda_mag 5 --fg_weight 25 --scheduler cosine --decay_epoch 150 --patience 50 --save_every 20 --ngf 96 --output_act hardtanh

echo.
echo ============================================
echo   Velocity v5 training complete!
echo ============================================
pause
