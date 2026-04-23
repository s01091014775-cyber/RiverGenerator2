@echo off
chcp 65001 >nul

echo ============================================
echo   Train: Velocity v7 (detail-focused, 512)
echo   Key changes vs v6:
echo     - FFT frequency loss (lambda_fft=8)
echo     - Full 512 resolution (vs 256)
echo     - Lower L1 weight (20 vs 25) to reduce blur
echo     - Stronger angular/mag (5 vs 2)
echo     - batch=2 (512 resolution needs more VRAM)
echo ============================================
echo.

set PREV_CKPT=checkpoints\vel_x25\best_G.pth
if not exist "%PREV_CKPT%" (
    echo [ERROR] Previous checkpoint not found: %PREV_CKPT%
    echo Run train_vel_v6.bat first.
    pause
    exit /b 1
)

echo ========== PHASE 1: Regression with FFT ==========
echo.

python train.py --target vel_x25 --pretrain --load_g %PREV_CKPT% --epochs 100 --batch 2 --img_size 512 --lr_g 5e-5 --lambda_l1 20 --lambda_vgg 5 --lambda_grad 5 --lambda_fft 8 --lambda_angular 5 --lambda_mag 5 --fg_weight 20 --scheduler cosine --decay_epoch 50 --patience 40 --save_every 20 --ngf 64 --output_act hardtanh

echo.
echo ========== PHASE 2: GAN Fine-tune with FFT ==========
echo.

set PRETRAIN_CKPT=checkpoints\vel_x25\pretrain_G.pth
if not exist "%PRETRAIN_CKPT%" (
    echo [ERROR] Pretrain checkpoint not found: %PRETRAIN_CKPT%
    pause
    exit /b 1
)

python train.py --target vel_x25 --load_g %PRETRAIN_CKPT% --epochs 200 --batch 2 --img_size 512 --lr_g 8e-5 --lr_d 2e-4 --lambda_l1 20 --lambda_vgg 5 --lambda_fm 10 --lambda_grad 5 --lambda_fft 8 --lambda_angular 5 --lambda_mag 5 --fg_weight 20 --r1_gamma 10 --r1_every 16 --scheduler cosine --decay_epoch 80 --patience 60 --save_every 20 --ngf 64 --output_act hardtanh

echo.
echo ============================================
echo   Velocity v7 training complete!
echo ============================================
pause
