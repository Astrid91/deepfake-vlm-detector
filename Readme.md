# Cross-Manipulation Deepfake Detection with CLIP + LoRA

This repository contains the code, pre-trained weights, and reproducible scripts for **Assignment-1-DFD** (Vision-Language Model based deepfake detection).

---

## Environment

```bash
# Python 3.10+
conda create -n dfd python=3.10
conda activate dfd
pip install -r requirements.txt        # torch, transformers, peft, scikit-learn, etc.
```

## Dataset

Download the official FaceForensics++ C40 frame package: https://www.dropbox.com/t/2Amyu4D5TulaIofv

Unzip to any folder, e.g.
```bash
DATA_ROOT/
 ├── Real_youtube/
 ├── FaceSwap/
 └── NeuralTextures/
```

## One-command reproduction
```bash
chmod +x run.sh
./run.sh ./DATA_ROOT         
```

The script will

1. build video-level splits (80 / 10 / 10) under splits/
2. train CLIP + LoRA → weights/best.pt
3. run inference on the test split
4. compute AUC / EER / ACC / F1 and save ROC curve to assets/roc.png
5. save per-video scores to results.json



## Directory layout
```bash
deepfake-vlm-detector/
 ├── src/
 │   ├── dataset.py       # FrameDataset + PIL collate
 │   ├── model.py         # CLIP backbone + LoRA injection
 │   └── utils.py         # seed, checkpoint, group_by_video
 ├── utils/make_splits.py # build official train / val / test csv
 ├── train.py             # optimisation loop
 ├── infer.py             # frame-level → video-level inference
 ├── evaluate.py          # metrics + ROC drawing
 ├── run.sh               # one-click pipeline
 ├── requirements.txt
 └── assets/              # ROC curve, Grad-CAMs, report figs

```


## Pre-trained weights
[Download weights.zip from Release](https://github.com/Astrid91/deepfake-vlm-detector/releases/tag/v1.0)

After downloading, please extract the file into the weights/ folder.

```bash
weights/
 ├── best.pt           # full model state_dict (for quick re-train)
 └── lora_adapter/     # PEFT-only, < 100 MB, required by rubric

```

## Expected runtime
```bash
| Stage           | RTX-3060 12 GB | Colab T4 |
| --------------- | -------------- | -------- |
| Training (3 ep) | ~8 min         | ~14 min  |
| Inference       | ~1.5 min       | ~3 min   |
| Eval + Plot     | <10 s          | <10 s    |

```
