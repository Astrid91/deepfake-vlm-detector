#!/usr/bin/env bash
set -e   # 任一步錯誤即停

DATA_ROOT=$1

# 1. 產生 split
echo "[make_splits] Generating CSV splits…"
python utils/make_splits.py --data_root "$DATA_ROOT"

# 2. 訓練
python train.py \
  --data_root "$DATA_ROOT" \
  --train_csv splits/train.csv \
  --val_csv   splits/val.csv

# 3. 推論（僅 NeuralTextures）
python infer.py \
  --data_root "$DATA_ROOT" \
  --test_csv  splits/test.csv \
  --ckpt      weights/best.pt \
  --save_json results.json

# 4. 指標與 ROC
python evaluate.py --result_json results.json
