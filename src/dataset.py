from pathlib import Path
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torch


def pil_collate(batch):
    """
    batch = [(img1, label1, path1), (img2, label2, path2), ...]
    回傳：list[PIL], Tensor[int], list[str]
    """
    images, labels, paths = zip(*batch)          # tuple → tuple
    labels = torch.tensor(labels, dtype=torch.float32)
    return list(images), labels, list(paths)


class FrameDataset(Dataset):
    """
    只讀檔案、回傳 (PIL.Image, int, relative_path)。
    影像尺寸保留原狀，交給 CLIPProcessor 統一 resize。
    """
    def __init__(self, root_dir, split_csv):
        self.root = Path(root_dir)
        # CSV (no header): path,label
        self.samples = pd.read_csv(split_csv, header=None, names=["path", "label"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path = self.samples.iloc[idx]["path"]
        label    = int(self.samples.iloc[idx]["label"])

        img = Image.open(self.root / rel_path).convert("RGB")
        return img, label, rel_path
