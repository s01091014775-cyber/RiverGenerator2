@echo off
chcp 65001 >nul

echo ============================================
echo   Train: Velocity v4 (fixed augmentation)
echo   Fixes:
echo   - Velocity direction correction on flip/rot
echo   - VEL_SCALE 100 -> 186 (full pixel range)
echo   - Hardtanh output (uniform gradient in [0,1])
echo   - Spatial gradient loss weight 5
echo   - Cosine scheduler + early stopping
echo ============================================

python train.py --target vel --epochs 300 --batch 4 --img_size 256 --lr_g 1e-4 --lr_d 4e-4 --lambda_l1 20 --lambda_vgg 5 --lambda_fm 10 --lambda_grad 5 --fg_weight 15 --scheduler cosine --decay_epoch 100 --patience 40 --save_every 20 --ngf 64 --output_act hardtanh

echo.
echo ============================================
echo   Velocity v4 training complete!
echo ============================================
pause
