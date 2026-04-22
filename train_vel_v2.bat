@echo off
chcp 65001 >nul
echo ============================================
echo   Train: Velocity v2 (optimized)
echo   - Spatial gradient loss (lambda_grad=5)
echo   - Cosine annealing scheduler
echo   - Early stopping (patience=30)
echo   - Higher fg_weight=15
echo ============================================
python train.py --target vel --epochs 300 --batch 4 --img_size 256 --lambda_l1 20 --lambda_vgg 0 --lambda_fm 10 --lambda_grad 5 --fg_weight 15 --scheduler cosine --patience 30 --save_every 10
echo.
echo Training complete. Checkpoints in: checkpoints\vel\
pause
