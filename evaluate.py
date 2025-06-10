import json, argparse, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score
import os

os.makedirs("assets", exist_ok=True)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_json", required=True)
    return ap.parse_args()

def main():
    args = parse_args()
    data = json.load(open(args.result_json))

    # y_true: 0=real, 1=fake
    y_true = np.array([0 if "Real_youtube" in k else 1 for k in data.keys()])
    y_pred = np.array(list(data.values()))

    # sanity check
    if len(np.unique(y_true)) < 2:
        raise ValueError("y_true 只有單一類別，請確認 test.csv 是否包含 Real_youtube 與 NeuralTextures。")

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc     = auc(fpr, tpr)

    # Equal-Error-Rate
    eer_idx = np.nanargmin(np.absolute((1 - tpr) - fpr))
    eer     = fpr[eer_idx]

    acc = accuracy_score(y_true, y_pred > 0.5)
    f1  = f1_score(y_true, y_pred > 0.5)

    print(f"AUC={roc_auc:.4f} | EER={eer:.4f} | ACC={acc:.4f} | F1={f1:.4f}")

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend()
    plt.title("NeuralTextures vs Real_youtube")
    plt.savefig("assets/roc.png", dpi=300)


if __name__ == "__main__":
    main()
