@echo off
chcp 65001 >nul

echo ============================================
echo   [1/3] Resume: Height v2 from epoch 10
echo ============================================
python train.py --target height --epochs 300 --batch 4 --img_size 256 --lr_g 1e-4 --lr_d 4e-4 --lambda_l1 20 --lambda_vgg 5 --lambda_fm 10 --lambda_grad 0 --fg_weight 10 --scheduler cosine --patience 25 --save_every 20 --resume checkpoints\height\ckpt_epoch10.pth
echo.

echo ============================================
echo   [2/3] Resume: Velocity v2 from epoch 30
echo ============================================
python train.py --target vel --epochs 300 --batch 4 --img_size 256 --lr_g 1e-4 --lr_d 4e-4 --lambda_l1 20 --lambda_vgg 0 --lambda_fm 10 --lambda_grad 5 --fg_weight 15 --scheduler cosine --patience 30 --save_every 20 --resume checkpoints\vel\ckpt_epoch30.pth
echo.

echo ============================================
echo   [3/3] Resume: Foam v2 from epoch 200
echo ============================================
python train.py --target foam --epochs 300 --batch 4 --img_size 256 --lr_g 1e-4 --lr_d 4e-4 --lambda_l1 20 --lambda_vgg 5 --lambda_fm 10 --lambda_grad 0 --fg_weight 10 --scheduler cosine --patience 25 --save_every 20 --resume checkpoints\foam\ckpt_epoch200.pth
echo.

echo ============================================
echo   All resumed training complete!
echo ============================================
pause
