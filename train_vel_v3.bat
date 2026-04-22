@echo off
chcp 65001 >nul

echo ============================================
echo   Train: Velocity v3 (renormalized data)
echo   - Background-centered normalization
echo   - VGG perceptual loss enabled
echo   - Spatial gradient loss weight 5
echo ============================================

python train.py --target vel --epochs 300 --batch 4 --img_size 256 --lr_g 1e-4 --lr_d 4e-4 --lambda_l1 20 --lambda_vgg 5 --lambda_fm 10 --lambda_grad 5 --fg_weight 15 --scheduler cosine --decay_epoch 100 --patience 30 --save_every 20 --ngf 64

echo.
echo ============================================
echo   Velocity v3 training complete!
echo ============================================
pause
