import argparse, json, torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from src.dataset import FrameDataset, pil_collate
from src.model import VLMDetector
from src.utils import load_checkpoint, seed_everything, group_by_video


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--test_csv",  required=True)
    ap.add_argument("--ckpt",      required=True)
    ap.add_argument("--save_json", default="results.json")
    return ap.parse_args()


def main():
    args = parse_args()
    seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = VLMDetector().to(device)
    load_checkpoint(model, args.ckpt)

    ds = FrameDataset(args.data_root, args.test_csv)
    dl = DataLoader(ds, batch_size=64, shuffle=False,
                    num_workers=4, collate_fn=pil_collate)

    frame_scores, frame_paths = [], []
    with torch.no_grad():
        for imgs, _, paths in tqdm(dl, desc="Infer"):
            logits = model(model.preprocess(imgs, device))
            probs  = torch.sigmoid(logits).cpu().tolist()

            frame_scores.extend(probs)
            frame_paths.extend(paths)

    video_scores = group_by_video(frame_paths, frame_scores)
    Path(args.save_json).write_text(json.dumps(video_scores, indent=2))
    print(f"✓ Saved {len(video_scores)} video scores → {args.save_json}")


if __name__ == "__main__":
    main()
