import random, os, torch
from collections import defaultdict
import torch.nn as nn


# ---------------- 通用 ----------------
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, path):
    torch.save({"model": model.state_dict()}, path)


def load_checkpoint(model, path):
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"], strict=False)


# ---------------- 評估輔助 ----------------
def group_by_video(frame_paths, scores):
    """
    依『類別 + 影片 ID』彙整多張 frame 分數。
    回傳：dict{ "Real_youtube/VID": mean_score , ... }
    """
    vids = defaultdict(list)
    for p, s in zip(frame_paths, scores):
        cls, fname = p.split("/")[:2]          # 例：Real_youtube, VID_000123.jpg
        vid = fname.split("_")[0]              # → VID
        key = f"{cls}/{vid}"                   # 保留類別前綴
        vids[key].append(s)
    return {k: float(sum(v) / len(v)) for k, v in vids.items()}
