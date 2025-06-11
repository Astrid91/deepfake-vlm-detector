#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
官方 split：
  • train = 80% Real_youtube + 90% FaceSwap
  • val   = 10% Real_youtube + 10% FaceSwap
  • test  = 10% Real_youtube + 100% NeuralTextures
"""
import argparse, random, os, pathlib, pandas as pd
from sklearn.model_selection import train_test_split

def collect_imgs(folder):
    exts = ("*.png", "*.jpg", "*.jpeg")
    return sum([list(pathlib.Path(folder).rglob(pat)) for pat in exts], [])

def by_video(folder, label, root):
    rec = []
    for p in collect_imgs(folder):
        vid = p.stem.split("_")[0]         # VID
        rec.append({"video": vid,
                    "path":  str(p.relative_to(root)),
                    "label": label})
    return pd.DataFrame(rec)

def save(df, fn):
    df.to_csv(fn, index=False, header=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir",  default="splits")
    ap.add_argument("--seed",     type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    root = pathlib.Path(args.data_root)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---------- Real (0) ----------
    real_df   = by_video(root / "Real_youtube", 0, root)
    real_vids = real_df.video.unique()
    # 80 / 10 / 10
    real_train_v, real_temp_v = train_test_split(real_vids, test_size=0.2, random_state=args.seed)
    real_val_v,   real_test_v = train_test_split(real_temp_v, test_size=0.5, random_state=args.seed)
    real_train = real_df[real_df.video.isin(real_train_v)]
    real_val   = real_df[real_df.video.isin(real_val_v)]
    real_test  = real_df[real_df.video.isin(real_test_v)]

    # ---------- FaceSwap (1) ----------
    fs_df   = by_video(root / "FaceSwap", 1, root)
    fs_vids = fs_df.video.unique()
    fs_train_v, fs_val_v = train_test_split(fs_vids, test_size=0.1, random_state=args.seed)
    fs_train = fs_df[fs_df.video.isin(fs_train_v)]
    fs_val   = fs_df[fs_df.video.isin(fs_val_v)]

    # ---------- NeuralTextures (1) ----------
    nt_df = by_video(root / "NeuralTextures", 1, root)

    # ---------- Write CSV ----------
    save(pd.concat([real_train, fs_train])[["path", "label"]], f"{args.out_dir}/train.csv")
    save(pd.concat([real_val,   fs_val  ])[["path", "label"]], f"{args.out_dir}/val.csv")
    save(pd.concat([real_test, nt_df   ])[["path", "label"]], f"{args.out_dir}/test.csv")

    for fn in ("train.csv", "val.csv", "test.csv"):
        df = pd.read_csv(f"{args.out_dir}/{fn}", header=None)
        print(f"{fn:<10} → total={len(df):5d} | fake={(df[1]==1).sum():5d} | real={(df[1]==0).sum():5d}")

if __name__ == "__main__":
    main()
