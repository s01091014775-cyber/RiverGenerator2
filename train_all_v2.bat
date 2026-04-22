@echo off
chcp 65001 >nul

echo ============================================
echo   [1/3] Train: Height v2 (optimized)
echo   - Self-attention bottleneck
echo   - Cosine annealing
echo   - Early stopping (patience=25)
echo   - 300 epochs
echo ============================================
python train.py --target height --epochs 300 --batch 4 --img_size 256 --lr_g 1e-4 --lr_d 4e-4 --lambda_l1 20 --lambda_vgg 5 --lambda_fm 10 --lambda_grad 0 --fg_weight 10 --scheduler cosine --patience 25 --save_every 20
echo.

echo ============================================
echo   [2/3] Train: Velocity v2 (optimized)
echo   - Self-attention bottleneck
echo   - Spatial gradient loss (lambda_grad=5)
echo   - Cosine annealing
echo   - Early stopping (patience=30)
echo   - Higher fg_weight=15
echo   - 300 epochs
echo ============================================
python train.py --target vel --epochs 300 --batch 4 --img_size 256 --lr_g 1e-4 --lr_d 4e-4 --lambda_l1 20 --lambda_vgg 0 --lambda_fm 10 --lambda_grad 5 --fg_weight 15 --scheduler cosine --patience 30 --save_every 20
echo.

echo ============================================
echo   [3/3] Train: Foam v2 (optimized)
echo   - Self-attention bottleneck
echo   - Cosine annealing
echo   - Early stopping (patience=25)
echo   - 300 epochs
echo ============================================
python train.py --target foam --epochs 300 --batch 4 --img_size 256 --lr_g 1e-4 --lr_d 4e-4 --lambda_l1 20 --lambda_vgg 5 --lambda_fm 10 --lambda_grad 0 --fg_weight 10 --scheduler cosine --patience 25 --save_every 20
echo.

echo ============================================
echo   All training complete!
echo ============================================
pause
