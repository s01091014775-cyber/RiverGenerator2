@echo off
chcp 65001 >nul

echo ============================================
echo   Train: Velocity v6 (two-phase + 2.5x amp)
echo   Data: vel_x25 (2.5x amplified velocity)
echo   Phase 1: Regression pretrain from v4 weights
echo     - No GAN/D, pure L1+VGG+grad+angular+mag
echo     - 80 epochs, cosine scheduler
echo   Phase 2: GAN fine-tune from pretrained G
echo     - Lower D lr (2e-4), R1 penalty (gamma=10)
echo     - 200 epochs with patience 60
echo ============================================
echo.

set V4_CKPT=checkpoints\vel\best_G_v4.pth
if not exist "%V4_CKPT%" (
    echo [ERROR] v4 checkpoint not found: %V4_CKPT%
    echo Please ensure best_G_v4.pth exists.
    pause
    exit /b 1
)

echo ========== PHASE 1: Regression Pretrain ==========
echo Loading v4 weights, training G only (no GAN)...
echo.

python train.py --target vel_x25 --pretrain --load_g %V4_CKPT% --epochs 80 --batch 4 --img_size 256 --lr_g 8e-5 --lambda_l1 25 --lambda_vgg 5 --lambda_grad 5 --lambda_angular 2 --lambda_mag 2 --fg_weight 20 --scheduler cosine --decay_epoch 40 --patience 30 --save_every 20 --ngf 64 --output_act hardtanh

echo.
echo ========== PHASE 2: GAN Fine-tune ==========
echo Loading pretrained G, training with GAN...
echo.

set PRETRAIN_CKPT=checkpoints\vel_x25\pretrain_G.pth
if not exist "%PRETRAIN_CKPT%" (
    echo [ERROR] Pretrain checkpoint not found: %PRETRAIN_CKPT%
    echo Phase 1 may have failed.
    pause
    exit /b 1
)

python train.py --target vel_x25 --load_g %PRETRAIN_CKPT% --epochs 200 --batch 4 --img_size 256 --lr_g 1e-4 --lr_d 2e-4 --lambda_l1 25 --lambda_vgg 5 --lambda_fm 10 --lambda_grad 5 --lambda_angular 2 --lambda_mag 2 --fg_weight 20 --r1_gamma 10 --r1_every 16 --scheduler cosine --decay_epoch 80 --patience 60 --save_every 20 --ngf 64 --output_act hardtanh

echo.
echo ============================================
echo   Velocity v6 training complete!
echo   Phase 1 best: checkpoints\vel_x25\pretrain_G.pth
echo   Phase 2 best: checkpoints\vel_x25\best_G.pth
echo ============================================
pause
