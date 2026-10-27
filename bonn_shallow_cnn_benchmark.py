"""
Bonn EEG: systematic shallow-CNN benchmark for binary ictal/non-ictal classification.

Expected processed-data layout (created by the user's preprocessing pipeline):
pipeline_notch50_bp0.5_80_impulse2.5/
    entrainement/{A,B,C,D,E}/segment_XXXX.npy
    test/{A,B,C,D,E}/segment_XXXX.npy

Labels:
    A, B, C, D -> 0 (non-ictal)
    E           -> 1 (ictal)

The test split is never used for model selection. Validation is carved out of
the existing training split, separately within each original Bonn subset.
"""

from __future__ import annotations

import csv
import itertools
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

# ============================================================
# Configuration générale
# ============================================================
DOSSIER_DONNEES = Path("pipeline_notch50_bp0.5_80_impulse2.5")
DOSSIER_SORTIE = Path("resultats_shallow_cnn")
CLASSES_ORIGINALES = ["A", "B", "C", "D", "E"]
CLASSES_NON_ICTALES = ["A", "B", "C", "D"]
CLASSE_ICTALE = "E"

FS = 173.61
VALIDATION_FRACTION = 0.20
SEEDS = [0, 1, 2]                 # Répétitions pour quantifier la variabilité
EPOCHS_MAX = 100
PATIENCE = 12                     # Early stopping sur la loss de validation
BATCH_SIZE = 32
NUM_WORKERS = 0                   # portable Windows/Linux
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# demandé : active/désactive la focal loss
FOCAL_LOSS = False
COMPARER_FOCAL_LOSS = True  # True : exécute False et True ; False : utilise FOCAL_LOSS

# Stratégie de déséquilibre quand FOCAL_LOSS=False :
# "all" = utiliser tous les exemples du train ; "balanced" = sous-échantillonner
# le pool A+B+C+D pour qu'il ait autant d'exemples que E.
BALANCING_MODE = "all"

# Normalisation à comparer :
# "per_segment" = z-score de chaque segment indépendamment
# "train_global" = moyenne/écart-type globaux appris uniquement sur le train
NORMALIZATION = "per_segment"

# Grille compacte mais explicite. Chaque ligne représente une architecture.
# Les variantes ne sont pas toutes issues telles quelles d'un article : elles
# sont des ablations motivées par la littérature et doivent être évaluées.
ARCHITECTURES = [
    {
        "name": "shallow_square_log",
        "kind": "shallow",
        "filters": 16,
        "kernel_size": 25,   # ~144 ms à 173.61 Hz
        "pool_size": 75,     # ~432 ms
        "dropout": 0.5,
    },
    {
        "name": "cnn1d_small",
        "kind": "cnn",
        "filters": 16,
        "kernel_size": 7,
        "pool_size": 2,
        "dropout": 0.3,
    },
    {
        "name": "cnn1d_medium",
        "kind": "cnn",
        "filters": 32,
        "kernel_size": 15,
        "pool_size": 4,
        "dropout": 0.5,
    },
    {
        "name": "cnn1d_multiscale",
        "kind": "multiscale",
        "filters": 16,
        "kernel_size": 7,
        "pool_size": 2,
        "dropout": 0.5,
    },
]

LEARNING_RATES = [1e-3, 3e-4]
WEIGHT_DECAYS = [0.0, 1e-4]
OPTIMIZERS = ["adam"]  # AdamW peut être ajouté en deuxième étape
# Pour une première exploration complète mais raisonnable, ces dimensions
# donnent 4 x 2 x 2 x 2 normalisations x 2 pertes x 2 modes d'équilibrage x 3 seeds.
# Pour éviter de sous-échantillonner avec focal loss, on force le mode "all".
# Le mode "balanced" est exécuté uniquement avec FOCAL_LOSS=False.


# ============================================================
# Reproductibilité
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Déterminisme autant que possible ; certaines opérations CUDA peuvent varier.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Chargement des segments prétraités
# ============================================================
def load_class_folder(split: str, classe: str) -> tuple[np.ndarray, list[str]]:
    folder = DOSSIER_DONNEES / split / classe
    if not folder.exists():
        raise FileNotFoundError(
            f"Dossier absent : {folder.resolve()}\n"
            "Vérifie DOSSIER_DONNEES et la structure produite par le pipeline."
        )
    files = sorted(folder.glob("segment_*.npy"))
    if not files:
        raise FileNotFoundError(f"Aucun segment_*.npy dans {folder.resolve()}")
    signals = [np.asarray(np.load(f, allow_pickle=False), dtype=np.float32).squeeze()
               for f in files]
    lengths = {x.shape for x in signals}
    if len(lengths) != 1 or any(x.ndim != 1 for x in signals):
        raise ValueError(f"Signaux non homogènes ou non 1D dans {folder}")
    return np.stack(signals), [f.name for f in files]


def load_split(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, groups = [], [], []
    for c in CLASSES_ORIGINALES:
        x, names = load_class_folder(split, c)
        y = 1 if c == CLASSE_ICTALE else 0
        xs.append(x)
        ys.extend([y] * len(x))
        groups.extend([c] * len(x))
    return np.concatenate(xs), np.asarray(ys, dtype=np.int64), np.asarray(groups)


def split_train_validation(x, y, groups, seed):
    """Validation tirée du train existant, stratifiée par classe originale A-E."""
    train_idx, val_idx = [], []
    rng = np.random.RandomState(seed)
    for c in CLASSES_ORIGINALES:
        idx = np.flatnonzero(groups == c)
        if len(idx) < 2:
            raise ValueError(f"Pas assez d'exemples de {c} pour créer la validation.")
        tr, va = train_test_split(
            idx, test_size=VALIDATION_FRACTION, random_state=seed,
            shuffle=True
        )
        train_idx.extend(tr.tolist())
        val_idx.extend(va.tolist())
    return np.asarray(train_idx), np.asarray(val_idx)


# ============================================================
# Normalisation : les statistiques globales sont apprises sur le train seulement
# ============================================================
def normalize_data(x_train, x_val, x_test, mode):
    eps = 1e-8
    if mode == "per_segment":
        def zscore_segments(x):
            mean = x.mean(axis=1, keepdims=True)
            std = x.std(axis=1, keepdims=True)
            return (x - mean) / np.maximum(std, eps)
        return zscore_segments(x_train), zscore_segments(x_val), zscore_segments(x_test), {
            "mode": mode
        }

    if mode == "train_global":
        mean = float(x_train.mean())
        std = float(x_train.std())
        std = max(std, eps)
        return (x_train - mean) / std, (x_val - mean) / std, (x_test - mean) / std, {
            "mode": mode, "mean_train": mean, "std_train": std
        }
    raise ValueError(f"Normalisation inconnue : {mode}")


# ============================================================
# Sous-échantillonnage binaire équilibré, sur train uniquement
# ============================================================
def balance_train_indices(y, groups, seed):
    """Conserve E entier et échantillonne A-D pour obtenir N(non-ictal)=N(ictal).

    Les exemples non ictaux sont tirés proportionnellement aux effectifs disponibles
    de A, B, C et D, afin de ne pas donner artificiellement le même poids à chaque
    sous-ensemble original. Le test et la validation restent inchangés.
    """
    rng = np.random.RandomState(seed)
    ictal = np.flatnonzero(y == 1)
    nonictal = np.flatnonzero(y == 0)
    target = len(ictal)
    if len(nonictal) <= target:
        return np.arange(len(y))
    # Allocation proportionnelle par groupe A-D, avec arrondis puis ajustement.
    group_counts = {c: int(np.sum(groups[nonictal] == c)) for c in CLASSES_NON_ICTALES}
    total_nonictal = sum(group_counts.values())
    quotas = {c: target * group_counts[c] / total_nonictal for c in CLASSES_NON_ICTALES}
    allocations = {c: int(np.floor(quotas[c])) for c in CLASSES_NON_ICTALES}
    remainder = target - sum(allocations.values())
    for c in sorted(CLASSES_NON_ICTALES, key=lambda k: quotas[k] - allocations[k], reverse=True)[:remainder]:
        allocations[c] += 1

    selected = []
    for c in CLASSES_NON_ICTALES:
        idx = np.flatnonzero((y == 0) & (groups == c))
        n_take = min(allocations[c], len(idx))
        selected.extend(rng.choice(idx, size=n_take, replace=False).tolist())
    # Correct any shortfall if a group had too few examples.
    if len(selected) < target:
        remaining = np.setdiff1d(nonictal, np.asarray(selected, dtype=int))
        selected.extend(rng.choice(remaining, size=target-len(selected), replace=False).tolist())
    return np.asarray(sorted(selected + ictal.tolist()), dtype=int)


# ============================================================
# Architectures
# ============================================================
class ShallowSquareLog(nn.Module):
    """Adaptation mono-canal d'une shallow ConvNet : conv, carré, moyenne, log."""
    def __init__(self, filters=16, kernel_size=25, pool_size=75, dropout=0.5):
        super().__init__()
        self.temporal = nn.Conv1d(1, filters, kernel_size, padding="same", bias=False)
        self.bn = nn.BatchNorm1d(filters)
        self.pool = nn.AvgPool1d(kernel_size=pool_size, stride=max(1, pool_size // 2))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(filters, 2)

    def forward(self, x):
        x = self.temporal(x)
        x = self.bn(x)
        x = x.square()
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = x.mean(dim=-1)
        return self.classifier(self.dropout(x))


class CNN1D(nn.Module):
    def __init__(self, filters=16, kernel_size=7, pool_size=2, dropout=0.3):
        super().__init__()
        f = filters
        self.features = nn.Sequential(
            nn.Conv1d(1, f, kernel_size, padding="same", bias=False),
            nn.BatchNorm1d(f), nn.ReLU(), nn.MaxPool1d(pool_size),
            nn.Conv1d(f, 2*f, kernel_size, padding="same", bias=False),
            nn.BatchNorm1d(2*f), nn.ReLU(), nn.MaxPool1d(pool_size),
            nn.Conv1d(2*f, 2*f, kernel_size, padding="same", bias=False),
            nn.BatchNorm1d(2*f), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(2*f, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class MultiScaleCNN(nn.Module):
    """Trois tailles de noyaux en parallèle pour tester plusieurs échelles temporelles."""
    def __init__(self, filters=16, kernel_size=7, pool_size=2, dropout=0.5):
        super().__init__()
        f = filters
        kernels = [3, 7, 15]
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, f, k, padding="same", bias=False),
                nn.BatchNorm1d(f), nn.ReLU(),
                nn.MaxPool1d(pool_size),
                nn.AdaptiveAvgPool1d(1),
            ) for k in kernels
        ])
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(3*f, 2)
        )

    def forward(self, x):
        feats = [branch(x) for branch in self.branches]
        return self.classifier(torch.cat(feats, dim=1))


def build_model(cfg):
    if cfg["kind"] == "shallow":
        return ShallowSquareLog(cfg["filters"], cfg["kernel_size"],
                                cfg["pool_size"], cfg["dropout"])
    if cfg["kind"] == "cnn":
        return CNN1D(cfg["filters"], cfg["kernel_size"],
                     cfg["pool_size"], cfg["dropout"])
    if cfg["kind"] == "multiscale":
        return MultiScaleCNN(cfg["filters"], cfg["kernel_size"],
                             cfg["pool_size"], cfg["dropout"])
    raise ValueError(cfg["kind"])


# ============================================================
# Pertes
# ============================================================
class FocalLoss(nn.Module):
    """Focal loss multiclasses, alpha pondéré selon les fréquences du train."""
    def __init__(self, alpha: torch.Tensor, gamma=2.0):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma

    def forward(self, logits, targets):
        log_probs = torch.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets[:, None]).squeeze(1)
        pt = log_pt.exp()
        alpha_t = self.alpha[targets]
        return (-alpha_t * (1.0 - pt).pow(self.gamma) * log_pt).mean()


def make_loss(y_train, focal):
    if not focal:
        return nn.CrossEntropyLoss()
    counts = np.bincount(y_train, minlength=2).astype(np.float32)
    weights = counts.sum() / (2.0 * np.maximum(counts, 1.0))
    weights = weights / weights.mean()
    return FocalLoss(torch.tensor(weights, dtype=torch.float32, device=DEVICE), gamma=2.0)


# ============================================================
# Entraînement et métriques
# ============================================================
def make_loader(x, y, shuffle):
    tx = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
    ty = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(tx, ty), batch_size=BATCH_SIZE,
                      shuffle=shuffle, num_workers=NUM_WORKERS,
                      pin_memory=(DEVICE == "cuda"))


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total_n = 0.0, 0
    probs_all, y_all = [], []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += float(loss.item()) * len(yb)
        total_n += len(yb)
        probs_all.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        y_all.append(yb.cpu().numpy())
    y_true = np.concatenate(y_all)
    p = np.concatenate(probs_all)
    pred = (p >= 0.5).astype(int)
    metrics = {
        "loss": total_loss / max(total_n, 1),
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision_ictal": precision_score(y_true, pred, zero_division=0),
        "recall_sensitivity_ictal": recall_score(y_true, pred, zero_division=0),
        "specificity_non_ictal": recall_score(1-y_true, 1-pred, zero_division=0),
        "f1_ictal": f1_score(y_true, pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }
    metrics["roc_auc"] = roc_auc_score(y_true, p) if len(np.unique(y_true)) == 2 else float("nan")
    return metrics


def fit_one(config, x_train, y_train, x_val, y_val, seed, focal_loss):
    set_seed(seed)
    model = build_model(config).to(DEVICE)
    criterion = make_loss(y_train, focal_loss)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    train_loader = make_loader(x_train, y_train, True)
    val_loader = make_loader(x_val, y_val, False)
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    history = []

    for epoch in range(1, EPOCHS_MAX + 1):
        model.train()
        running_loss, seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(yb)
            seen += len(yb)

        val_metrics = evaluate(model, val_loader, criterion)
        history.append({
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "val_loss": val_metrics["loss"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        })

        if val_metrics["loss"] < best_val_loss - 1e-5:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    if best_state is None:
        raise RuntimeError("Aucun état de modèle sauvegardé.")
    model.load_state_dict(best_state)
    # Retourne les métriques de validation du meilleur état ; le test reste intact.
    best_val_metrics = evaluate(model, val_loader, criterion)
    return model, best_epoch, history, best_val_metrics


# ============================================================
# Expériences
# ============================================================
def run_benchmark():
    DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)
    x_train_full, y_train_full, g_train_full = load_split("entrainement")
    x_test, y_test, g_test = load_split("test")

    if x_train_full.shape[1] != x_test.shape[1]:
        raise ValueError("Longueurs différentes entre train et test.")
    if set(np.unique(y_train_full)) != {0, 1} or set(np.unique(y_test)) != {0, 1}:
        raise ValueError("Le train et le test doivent contenir les deux classes binaires.")

    print(f"Device : {DEVICE}")
    print(f"Train existant : {len(y_train_full)} ; test indépendant : {len(y_test)}")
    print("Train classes originales :", {c: int(np.sum(g_train_full == c)) for c in CLASSES_ORIGINALES})
    print("Train binaire :", {str(k): int(np.sum(y_train_full == k)) for k in [0, 1]})
    print("Test binaire :", {str(k): int(np.sum(y_test == k)) for k in [0, 1]})

    # Toutes les combinaisons architecture x lr x wd x normalisation x stratégie.
    # Focal loss et sous-échantillonnage ne sont pas combinés ici afin d'isoler
    # les deux interventions : focal loss sur toutes les données ; sous-échantillonnage
    # avec cross-entropy standard.
    results = []
    best_candidate = None
    run_id = 0
    for arch, lr, wd, norm_mode, focal, balance_mode, seed in itertools.product(
        ARCHITECTURES, LEARNING_RATES, WEIGHT_DECAYS,
        ["per_segment", "train_global"],
        ([False, True] if COMPARER_FOCAL_LOSS else [FOCAL_LOSS]),
        ["all", "balanced"],
        SEEDS,
    ):
        if focal and balance_mode == "balanced":
            continue
        if (not focal) and balance_mode == "balanced":
            actual_balance = "balanced"
        else:
            actual_balance = "all"

        run_id += 1
        print(f"\n[{run_id}] {arch['name']} | lr={lr:g} wd={wd:g} "
              f"norm={norm_mode} focal={focal} balance={actual_balance} seed={seed}")

        tr_idx, va_idx = split_train_validation(
            x_train_full, y_train_full, g_train_full, seed
        )
        x_tr, y_tr, g_tr = x_train_full[tr_idx], y_train_full[tr_idx], g_train_full[tr_idx]
        x_va, y_va = x_train_full[va_idx], y_train_full[va_idx]

        if actual_balance == "balanced":
            keep = balance_train_indices(y_tr, g_tr, seed)
            x_tr, y_tr = x_tr[keep], y_tr[keep]

        # Les statistiques sont calculées uniquement sur le train retenu.
        x_tr, x_va, _unused_test, norm_meta = normalize_data(x_tr, x_va, x_test, norm_mode)
        cfg = dict(arch)
        cfg.update({"learning_rate": lr, "weight_decay": wd})
        start = time.time()
        model, best_epoch, history, val_metrics = fit_one(
            cfg, x_tr, y_tr, x_va, y_va, seed, focal
        )
        elapsed = time.time() - start

        record = {
            "run_id": run_id,
            "architecture": arch["name"],
            "learning_rate": lr,
            "weight_decay": wd,
            "normalization": norm_mode,
            "focal_loss": focal,
            "focal_gamma": 2.0 if focal else None,
            "balancing_mode": actual_balance,
            "seed": seed,
            "n_train_after_sampling": len(y_tr),
            "n_val": len(y_va),
            "best_epoch": best_epoch,
            "runtime_seconds": round(elapsed, 2),
            "normalization_metadata": json.dumps(norm_meta),
            **{f"val_{k}": v for k, v in val_metrics.items() if k != "confusion_matrix"},
            "val_confusion_matrix": json.dumps(val_metrics["confusion_matrix"]),
        }
        results.append(record)
        # Le gagnant est choisi sur la validation, jamais sur le test.
        candidate_score = (val_metrics["balanced_accuracy"], val_metrics["recall_sensitivity_ictal"])
        if best_candidate is None or candidate_score > best_candidate["score"]:
            best_candidate = {
                "score": candidate_score,
                "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "config": cfg,
                "seed": seed,
                "normalization": norm_meta,
                "focal_loss": focal,
                "balancing_mode": actual_balance,
                "architecture": arch["name"],
                "best_epoch": best_epoch,
                "train_class_counts": np.bincount(y_tr, minlength=2).tolist(),
            }

        # Sauvegarder l'historique et le meilleur modèle pour chaque configuration.
        prefix = (
            f"{run_id:04d}_{arch['name']}_norm-{norm_mode}_focal-{int(focal)}"
            f"_bal-{actual_balance}_lr-{lr:g}_wd-{wd:g}_seed-{seed}"
        )
        with open(DOSSIER_SORTIE / f"{prefix}_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        torch.save(
            {"model_state_dict": model.state_dict(), "config": cfg,
             "best_epoch": best_epoch, "normalization": norm_meta},
            DOSSIER_SORTIE / f"{prefix}_model.pt"
        )
        with open(DOSSIER_SORTIE / "results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    # Résumé par famille de configuration, moyenne et écart-type sur les seeds.
    import pandas as pd
    df = pd.DataFrame(results)
    metrics = ["balanced_accuracy", "accuracy", "precision_ictal",
               "recall_sensitivity_ictal", "specificity_non_ictal", "f1_ictal", "roc_auc"]
    group_cols = ["architecture", "learning_rate", "weight_decay", "normalization",
                  "focal_loss", "balancing_mode"]
    summary = df.groupby(group_cols)[metrics].agg(["mean", "std"]).reset_index()
    summary.to_csv(DOSSIER_SORTIE / "summary_by_configuration.csv", index=False)
    df.sort_values(["balanced_accuracy", "recall_sensitivity_ictal"], ascending=False).to_csv(
        DOSSIER_SORTIE / "results_sorted_validation_selected.csv", index=False
    )
    print(f"\nTerminé. Résultats : {DOSSIER_SORTIE.resolve()}")
    print("Attention : ne choisis pas un modèle sur le score test. Utilise la validation")
    print("pour la sélection, puis rapporte le test final du modèle retenu.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    run_benchmark()
