"""
evaluate_model.py - Évaluation multi-métriques du meilleur modèle Derma-Scan (ResNet34 focal v3).

Répond à la question : « Quelles sont les métriques de Derma-Scan au-delà du rappel mélanome ? »

Le split validation est reproduit à l'identique des notebooks 02/03
(train_test_split, test_size=0.2, random_state=42, stratify=dx).

Modes de split (--split) :
  image         split d'origine des notebooks (par image) - peut fuiter via lesion_id.
  lesion-clean  split d'origine, mais on retire les images dont le lesion_id apparait aussi
                dans le train : evaluation sans fuite pour des poids entraines sur le split
                par image. (defaut des sorties : evaluation_lesion_clean/)

Option --export-grouped-split : ecrit un NOUVEAU split 80/20 groupe par lesion_id
(StratifiedGroupKFold) dans grouped_split.csv, a utiliser pour REENTRAINER le modele.
Evaluer les poids actuels sur ce split serait contamine (ils ont vu ces lesions).

Usage (depuis la racine du projet) :
    python evaluate_model.py
    python evaluate_model.py --split lesion-clean --out evaluation_lesion_clean
    python evaluate_model.py --export-grouped-split

Sorties dans ./evaluation/ : metrics.json, report.md, confusion_matrix.png,
roc_curves.png, pr_curves.png, melanoma_threshold.png, calibration.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    classification_report, cohen_kappa_score, confusion_matrix, f1_score,
    log_loss, matthews_corrcoef, precision_recall_curve, roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# Ordre d'entraînement (notebooks 02/03, api/model_handler.py)
CLASSES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}
MEL = LABEL_MAP["mel"]
# Classes malignes / pré-malignes (cf. LABEL_MAP de l'API : mel, bcc, akiec)
MALIGNANT = [LABEL_MAP[c] for c in ("mel", "bcc", "akiec")]

NORM_MEAN, NORM_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])


class SkinDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        return VAL_TRANSFORMS(img), LABEL_MAP[row["dx"]]


def load_model(weights, device):
    model = models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(weights, map_location=device))
    return model.to(device).eval()


@torch.no_grad()
def predict(model, loader, device):
    probs, labels = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(device)), dim=1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(probs)


# ----------------------------------------------------------------------
# Métriques
# ----------------------------------------------------------------------
def binary_stats(y_true, y_pred):
    """Sensibilité, spécificité, VPP, VPN, F1 pour un problème binaire (1 = positif)."""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    div = lambda a, b: a / b if b else float("nan")
    ppv = div(tp, tp + fp)
    sens = div(tp, tp + fn)
    return {
        "sensitivity_recall": sens,
        "specificity": div(tn, tn + fp),
        "ppv_precision": ppv,
        "npv": div(tn, tn + fn),
        "f1": div(2 * ppv * sens, ppv + sens) if tp else 0.0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def expected_calibration_error(y_true, probs, n_bins=10):
    conf = probs.max(1)
    correct = (probs.argmax(1) == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            acc, c = correct[m].mean(), conf[m].mean()
            ece += m.mean() * abs(acc - c)
            rows.append((c, acc, int(m.sum())))
    return float(ece), rows


def global_metrics(y, probs):
    pred = probs.argmax(1)
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y, pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y, pred),
        "cohen_kappa": cohen_kappa_score(y, pred),
        "log_loss": log_loss(y, probs, labels=range(len(CLASSES))),
    }


def bootstrap_ci(y, probs, n_boot, seed=0):
    """IC 95 % par bootstrap pour les métriques clés."""
    rng = np.random.default_rng(seed)
    n = len(y)
    keys = {"balanced_accuracy": [], "f1_macro": [], "mcc": [],
            "melanoma_recall": [], "melanoma_auc": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], probs[idx]
        if (yb == MEL).sum() == 0 or (yb != MEL).sum() == 0:
            continue
        pred = pb.argmax(1)
        keys["balanced_accuracy"].append(balanced_accuracy_score(yb, pred))
        keys["f1_macro"].append(f1_score(yb, pred, average="macro", zero_division=0))
        keys["mcc"].append(matthews_corrcoef(yb, pred))
        keys["melanoma_recall"].append(((pred == MEL) & (yb == MEL)).sum() / (yb == MEL).sum())
        keys["melanoma_auc"].append(roc_auc_score(yb == MEL, pb[:, MEL]))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in keys.items()}


def evaluate(y, probs, n_boot):
    pred = probs.argmax(1)
    y_bin = label_binarize(y, classes=range(len(CLASSES)))
    out = {"n_samples": int(len(y)), "global": global_metrics(y, probs)}

    # Par classe (one-vs-rest)
    per_class = {}
    for i, c in enumerate(CLASSES):
        s = binary_stats(y_bin[:, i], (pred == i).astype(int))
        s["support"] = int(y_bin[:, i].sum())
        s["roc_auc"] = roc_auc_score(y_bin[:, i], probs[:, i])
        s["pr_auc"] = average_precision_score(y_bin[:, i], probs[:, i])
        per_class[c] = s
    out["per_class"] = per_class
    out["macro_roc_auc_ovr"] = float(np.mean([v["roc_auc"] for v in per_class.values()]))
    out["macro_pr_auc"] = float(np.mean([v["pr_auc"] for v in per_class.values()]))

    # Mélanome : métriques détaillées + balayage de seuil
    out["melanoma"] = {**per_class["mel"]}
    thresholds = {}
    for t in (0.1, 0.2, 0.3, 0.4, 0.5):
        thresholds[str(t)] = binary_stats(y_bin[:, MEL], (probs[:, MEL] >= t).astype(int))
    out["melanoma_threshold_sweep"] = thresholds

    # Tâche clinique : malin (mel/bcc/akiec) vs bénin
    y_mal = np.isin(y, MALIGNANT).astype(int)
    p_mal = probs[:, MALIGNANT].sum(1)
    out["malignant_vs_benign"] = {
        **binary_stats(y_mal, np.isin(pred, MALIGNANT).astype(int)),
        "roc_auc": roc_auc_score(y_mal, p_mal),
        "pr_auc": average_precision_score(y_mal, p_mal),
    }

    # Calibration
    ece, bins = expected_calibration_error(y, probs)
    out["calibration"] = {"ece": ece, "bins": bins}

    # Où part-on se tromper sur les mélanomes ?
    cm = confusion_matrix(y, pred, labels=range(len(CLASSES)))
    out["melanoma_misclassified_as"] = {
        CLASSES[j]: int(cm[MEL, j]) for j in range(len(CLASSES)) if j != MEL and cm[MEL, j]
    }
    out["confusion_matrix"] = cm.tolist()
    out["bootstrap_ci95"] = bootstrap_ci(y, probs, n_boot) if n_boot else {}
    return out


# ----------------------------------------------------------------------
# Graphiques & rapport
# ----------------------------------------------------------------------
def make_plots(y, probs, res, outdir):
    cm = np.array(res["confusion_matrix"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, data, title, fmt in (
        (axes[0], cm, "Matrice de confusion (effectifs)", "d"),
        (axes[1], cm / cm.sum(1, keepdims=True), "Normalisée par ligne (= rappel)", ".2f"),
    ):
        ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(7), CLASSES, rotation=45)
        ax.set_yticks(range(7), CLASSES)
        ax.set_xlabel("Prédit"); ax.set_ylabel("Réel"); ax.set_title(title)
        for i in range(7):
            for j in range(7):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                        color="white" if data[i, j] > data.max() / 2 else "black", fontsize=8)
    fig.tight_layout(); fig.savefig(outdir / "confusion_matrix.png", dpi=130); plt.close(fig)

    y_bin = label_binarize(y, classes=range(7))
    for kind, fname in (("roc", "roc_curves.png"), ("pr", "pr_curves.png")):
        fig, ax = plt.subplots(figsize=(7, 6))
        for i, c in enumerate(CLASSES):
            if kind == "roc":
                a, b, _ = roc_curve(y_bin[:, i], probs[:, i])
                ax.plot(a, b, label=f"{c} (AUC={res['per_class'][c]['roc_auc']:.3f})")
            else:
                p, r, _ = precision_recall_curve(y_bin[:, i], probs[:, i])
                ax.plot(r, p, label=f"{c} (AP={res['per_class'][c]['pr_auc']:.3f})")
        if kind == "roc":
            ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("1 - Spécificité" if kind == "roc" else "Rappel")
        ax.set_ylabel("Sensibilité" if kind == "roc" else "Précision")
        ax.set_title("Courbes ROC (one-vs-rest)" if kind == "roc" else "Courbes Précision-Rappel")
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(outdir / fname, dpi=130); plt.close(fig)

    ts = res["melanoma_threshold_sweep"]
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = [float(t) for t in ts]
    for key, lab in (("sensitivity_recall", "Sensibilité"), ("specificity", "Spécificité"),
                     ("ppv_precision", "Précision (VPP)")):
        ax.plot(xs, [ts[t][key] for t in ts], marker="o", label=lab)
    ax.set_xlabel("Seuil de probabilité 'mel'"); ax.set_title("Mélanome : compromis selon le seuil")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(outdir / "melanoma_threshold.png", dpi=130); plt.close(fig)

    bins = res["calibration"]["bins"]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.plot([b[0] for b in bins], [b[1] for b in bins], marker="o")
    ax.set_xlabel("Confiance moyenne"); ax.set_ylabel("Exactitude observée")
    ax.set_title(f"Calibration (ECE={res['calibration']['ece']:.3f})")
    fig.tight_layout(); fig.savefig(outdir / "calibration.png", dpi=130); plt.close(fig)


def export_grouped_split(df, path, seed=42):
    """Split ~80/20 stratifie par dx, sans aucun lesion_id commun entre train et val."""
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    tr_idx, va_idx = next(sgkf.split(df, df["dx"], groups=df["lesion_id"]))
    out = df[["image_id", "lesion_id", "dx"]].copy()
    out["split"] = "train"
    out.iloc[va_idx, out.columns.get_loc("split")] = "val"
    out.to_csv(path, index=False)
    tr, va = df.iloc[tr_idx], df.iloc[va_idx]
    assert not set(tr["lesion_id"]) & set(va["lesion_id"])
    print(f"Split groupe ecrit dans {path} : train={len(tr)} val={len(va)} (0 lesion commune)")
    print(va["dx"].value_counts().to_string())


def write_report(res, outdir, weights, split_info=""):
    g, ci = res["global"], res["bootstrap_ci95"]
    f = lambda v: "n/a" if v != v else f"{v:.3f}"
    def with_ci(name, v):
        return f"{v:.3f} [{ci[name][0]:.3f} – {ci[name][1]:.3f}]" if name in ci else f"{v:.3f}"

    L = [f"# Évaluation Derma-Scan — `{weights}`", "",
         f"Jeu de validation : {res['n_samples']} images. {split_info}", "",
         "## Métriques globales", "", "| Métrique | Valeur |", "|---|---|",
         f"| Accuracy | {f(g['accuracy'])} |",
         f"| Balanced accuracy | {with_ci('balanced_accuracy', g['balanced_accuracy'])} |",
         f"| F1 macro | {with_ci('f1_macro', g['f1_macro'])} |",
         f"| F1 pondéré | {f(g['f1_weighted'])} |",
         f"| MCC | {with_ci('mcc', g['mcc'])} |",
         f"| Cohen's kappa | {f(g['cohen_kappa'])} |",
         f"| Log-loss | {f(g['log_loss'])} |",
         f"| ROC-AUC macro (OvR) | {f(res['macro_roc_auc_ovr'])} |",
         f"| PR-AUC macro | {f(res['macro_pr_auc'])} |",
         f"| ECE (calibration) | {f(res['calibration']['ece'])} |", "",
         "## Par classe", "",
         "| Classe | Support | Rappel | Spécificité | Précision | NPV | F1 | ROC-AUC | PR-AUC |",
         "|---|---|---|---|---|---|---|---|---|"]
    for c, s in res["per_class"].items():
        L.append(f"| {c} | {s['support']} | {f(s['sensitivity_recall'])} | {f(s['specificity'])} | "
                 f"{f(s['ppv_precision'])} | {f(s['npv'])} | {f(s['f1'])} | {f(s['roc_auc'])} | {f(s['pr_auc'])} |")

    m = res["melanoma"]
    L += ["", "## Mélanome", "",
          f"- Rappel (sensibilité) : {with_ci('melanoma_recall', m['sensitivity_recall'])}",
          f"- Spécificité : {f(m['specificity'])} — Précision (VPP) : {f(m['ppv_precision'])} — NPV : {f(m['npv'])}",
          f"- ROC-AUC : {with_ci('melanoma_auc', m['roc_auc'])} — PR-AUC : {f(m['pr_auc'])}",
          f"- TP={m['tp']} FN={m['fn']} FP={m['fp']} TN={m['tn']}",
          f"- Mélanomes ratés → classés comme : {res['melanoma_misclassified_as']}", "",
          "### Balayage de seuil (proba 'mel' ≥ t)", "",
          "| Seuil | Sensibilité | Spécificité | Précision | FN | FP |", "|---|---|---|---|---|---|"]
    for t, s in res["melanoma_threshold_sweep"].items():
        L.append(f"| {t} | {f(s['sensitivity_recall'])} | {f(s['specificity'])} | "
                 f"{f(s['ppv_precision'])} | {s['fn']} | {s['fp']} |")

    b = res["malignant_vs_benign"]
    L += ["", "## Tâche clinique : malin (mel + bcc + akiec) vs bénin", "",
          f"- Sensibilité : {f(b['sensitivity_recall'])} — Spécificité : {f(b['specificity'])}",
          f"- Précision : {f(b['ppv_precision'])} — NPV : {f(b['npv'])} — F1 : {f(b['f1'])}",
          f"- ROC-AUC : {f(b['roc_auc'])} — PR-AUC : {f(b['pr_auc'])}", "",
          "## Limites", "",
          "- Le jeu de validation a servi à la sélection du modèle (early stopping / meilleur checkpoint) : "
          "les chiffres sont légèrement optimistes, pas un vrai jeu de test.",
          "- Si le split est fait **par image** ; HAM10000 contient plusieurs images par lésion (`lesion_id`), "
          "ce qui provoque une fuite entre train et validation (voir le mode `lesion-clean`).",
          "- Dataset mono-source (ISIC) : aucune validation externe ni sur images de smartphone.",
          "- Les IC bootstrap ne reflètent que la variance d'échantillonnage (peu d'exemples pour df/vasc/akiec).", ""]
    (outdir / "report.md").write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="models/resnet34_derma_focal_v3.pth")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default=None, help="défaut : evaluation (image) / evaluation_lesion_clean")
    ap.add_argument("--split", choices=["image", "lesion-clean"], default="image")
    ap.add_argument("--export-grouped-split", action="store_true",
                    help="écrit grouped_split.csv (split groupé par lesion_id) pour réentraîner, puis quitte")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-boot", type=int, default=500, help="itérations bootstrap (0 = désactivé)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = Path(args.data_dir)
    df = pd.read_csv(data / "HAM10000_metadata.csv")
    df["path"] = df["image_id"].apply(lambda x: data / "all_images" / f"{x}.jpg")
    if args.export_grouped_split:
        export_grouped_split(df, "grouped_split.csv")
        return
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["dx"])
    n_val = len(val_df)
    leak = val_df["lesion_id"].isin(set(train_df["lesion_id"]))
    split_info = (f"Split d'origine stratifié 80/20 (seed 42) ; {int(leak.sum())}/{n_val} "
                  f"({leak.mean():.1%}) des images de validation partagent un `lesion_id` avec le train.")
    if args.split == "lesion-clean":
        val_df = val_df[~leak]
        split_info += (f" **Mode lesion-clean : ces images sont retirées**, il reste {len(val_df)} images "
                       f"({val_df['dx'].value_counts().to_dict()}). Effectifs faibles pour mel/df/vasc : "
                       "lire les IC avec prudence.")
    print(split_info)
    print(f"Validation : {len(val_df)} images | device : {device}")
    out_name = args.out or ("evaluation" if args.split == "image" else "evaluation_lesion_clean")

    model = load_model(args.weights, device)
    loader = DataLoader(SkinDataset(val_df), batch_size=args.batch_size, shuffle=False, num_workers=0)
    y, probs = predict(model, loader, device)

    res = evaluate(y, probs, args.n_boot)
    outdir = Path(out_name); outdir.mkdir(exist_ok=True)
    make_plots(y, probs, res, outdir)
    write_report(res, outdir, args.weights, split_info)
    (outdir / "metrics.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")

    print(classification_report(y, probs.argmax(1), target_names=CLASSES, digits=3, zero_division=0))
    print(f"Rapport et graphiques écrits dans {outdir.resolve()}")


if __name__ == "__main__":
    main()
