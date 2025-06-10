import argparse, os, torch
from torch.utils.data import DataLoader
from src.dataset import FrameDataset, pil_collate
from src.model import VLMDetector
from src.utils import seed_everything, save_checkpoint
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv",   required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs",     type=int, default=32)
    ap.add_argument("--lr",     type=float, default=1e-4)
    ap.add_argument("--out_dir", type=str, default="weights")
    return ap.parse_args()


def main():
    args = parse_args()
    seed_everything(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = VLMDetector().to(device)

    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    train_ds = FrameDataset(args.data_root, args.train_csv)
    val_ds   = FrameDataset(args.data_root, args.val_csv)
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                              num_workers=4, collate_fn=pil_collate)
    val_loader   = DataLoader(val_ds,   batch_size=args.bs, shuffle=False,
                              num_workers=4, collate_fn=pil_collate)

    criterion = torch.nn.BCEWithLogitsLoss()
    best_auc  = 0.0

    for ep in range(1, args.epochs + 1):
        # -------- training --------
        model.train()
        for imgs, labels, _ in tqdm(train_loader, desc=f"Epoch {ep}/{args.epochs}"):
            pixel_vals = model.preprocess(imgs, device)  # imgs: list[PIL]
            labels     = labels.to(device)               # tensor [B]

            logits = model(pixel_vals)
            loss   = criterion(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

        # -------- validation --------
        model.eval()
        y_pred, y_true = [], []
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                logits = model(model.preprocess(imgs, device))
                y_pred.extend(torch.sigmoid(logits).cpu().tolist())
                y_true.extend(labels.cpu().tolist())

        auc = roc_auc_score(y_true, y_pred)
        print(f"[Val] Epoch {ep}: AUC = {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            os.makedirs(args.out_dir, exist_ok=True)
            save_checkpoint(model, f"{args.out_dir}/best.pt")

    # 儲存 LoRA adapter (<100 MB)
    model.clip.save_pretrained(f"{args.out_dir}/lora_adapter")


if __name__ == "__main__":
    main()
