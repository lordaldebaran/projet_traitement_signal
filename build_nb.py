import json
cells = []
def md(s):  cells.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
def code(s):cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.strip("\n")})

md(r'''# Benchmark systématique de shallow-CNN — Bonn EEG

Classification binaire **ictal / non-ictal**.

## Structure
## Labels
- A, B, C, D -> `0` (non-ictal)
- E -> `1` (ictal)

## Protocole
- Test jamais utilise pour la selection.
- Validation extraite du train, stratifiee par A–E.
- Stats de normalisation apprises sur le train seulement.
- Meilleur modele choisi sur balanced accuracy + sensibilite ictale en validation.''')

code(r'''from __future__ import annotations
import csv, itertools, json, random, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
import matplotlib.pyplot as plt''')

code(r'''DOSSIER_DONNEES = Path("pipeline_notch50_bp0.5_80_impulse2.5")
DOSSIER_SORTIE = Path("resultats_shallow_cnn")
CLASSES_ORIGINALES = ["A","B","C","D","E"]
CLASSES_NON_ICTALES = ["A","B","C","D"]
CLASSE_ICTALE = "E"
FS = 173.61
VALIDATION_FRACTION = 0.20
SEEDS = [0,1,2]
EPOCHS_MAX = 100
PATIENCE = 12
BATCH_SIZE = 32
NUM_WORKERS = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FOCAL_LOSS = False
COMPARER_FOCAL_LOSS = True
BALANCING_MODE = "all"
NORMALIZATION = "per_segment"
ARCHITECTURES = [
    {"name":"shallow_square_log","kind":"shallow","filters":16,"kernel_size":25,"pool_size":75,"dropout":0.5},
    {"name":"cnn1d_small","kind":"cnn","filters":16,"kernel_size":7,"pool_size":2,"dropout":0.3},
    {"name":"cnn1d_medium","kind":"cnn","filters":32,"kernel_size":15,"pool_size":4,"dropout":0.5},
    {"name":"cnn1d_multiscale","kind":"multiscale","filters":16,"kernel_size":7,"pool_size":2,"dropout":0.5},
]
LEARNING_RATES = [1e-3, 3e-4]
WEIGHT_DECAYS = [0.0, 1e-4]
OPTIMIZERS = ["adam"]
print("Device :", DEVICE)''')

code(r'''def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False''')
code(r'''def load_class_folder(split, classe):
    folder = DOSSIER_DONNEES / split / classe
    if not folder.exists():
        raise FileNotFoundError(f"Dossier absent : {folder.resolve()}")
    files = sorted(folder.glob("segment_*.npy"))
    if not files: raise FileNotFoundError(f"Aucun segment dans {folder}")
    signals = [np.asarray(np.load(f, allow_pickle=False), dtype=np.float32).squeeze() for f in files]
    if len({x.shape for x in signals}) != 1 or any(x.ndim != 1 for x in signals):
        raise ValueError(f"Signaux non homogenes dans {folder}")
    return np.stack(signals), [f.name for f in files]

def load_split(split):
    xs, ys, groups = [], [], []
    for c in CLASSES_ORIGINALES:
        x, _ = load_class_folder(split, c)
        y = 1 if c == CLASSE_ICTALE else 0
        xs.append(x); ys.extend([y]*len(x)); groups.extend([c]*len(x))
    return np.concatenate(xs), np.asarray(ys, dtype=np.int64), np.asarray(groups)

def split_train_validation(x, y, groups, seed):
    tr_idx, va_idx = [], []
    for c in CLASSES_ORIGINALES:
        idx = np.flatnonzero(groups == c)
        if len(idx) < 2: raise ValueError(f"Pas assez d'exemples de {c}.")
        tr, va = train_test_split(idx, test_size=VALIDATION_FRACTION, random_state=seed)
        tr_idx.extend(tr.tolist()); va_idx.extend(va.tolist())
    return np.asarray(tr_idx), np.asarray(va_idx)''')

code(r'''def normalize_data(x_train, x_val, x_test, mode):
    eps = 1e-8
    if mode == "per_segment":
        def z(x):
            m = x.mean(axis=1, keepdims=True); s = x.std(axis=1, keepdims=True)
            return (x - m) / np.maximum(s, eps)
        return z(x_train), z(x_val), z(x_test), {"mode": mode}
    if mode == "train_global":
        m = float(x_train.mean()); s = max(float(x_train.std()), eps)
        return ((x_train-m)/s, (x_val-m)/s, (x_test-m)/s,
                {"mode": mode, "mean_train": m, "std_train": s})
    raise ValueError(f"Normalisation inconnue : {mode}")''')

code(r'''def balance_train_indices(y, groups, seed):
    rng = np.random.RandomState(seed)
    ictal = np.flatnonzero(y == 1); nonictal = np.flatnonzero(y == 0)
    target = len(ictal)
    if len(nonictal) <= target: return np.arange(len(y))
    gc = {c: int(np.sum(groups[nonictal] == c)) for c in CLASSES_NON_ICTALES}
    tot = sum(gc.values())
    q = {c: target * gc[c] / tot for c in CLASSES_NON_ICTALES}
    alloc = {c: int(np.floor(q[c])) for c in CLASSES_NON_ICTALES}
    rem = target - sum(alloc.values())
    for c in sorted(CLASSES_NON_ICTALES, key=lambda k: q[k]-alloc[k], reverse=True)[:rem]:
        alloc[c] += 1
    sel = []
    for c in CLASSES_NON_ICTALES:
        idx = np.flatnonzero((y == 0) & (groups == c))
        sel.extend(rng.choice(idx, size=min(alloc[c], len(idx)), replace=False).tolist())
    if len(sel) < target:
        rest = np.setdiff1d(nonictal, np.asarray(sel, dtype=int))
        sel.extend(rng.choice(rest, size=target-len(sel), replace=False).tolist())
    return np.asarray(sorted(sel + ictal.tolist()), dtype=int)''')
code(r'''class ShallowSquareLog(nn.Module):
    def __init__(self, filters=16, kernel_size=25, pool_size=75, dropout=0.5):
        super().__init__()
        self.temporal = nn.Conv1d(1, filters, kernel_size, padding="same", bias=False)
        self.bn = nn.BatchNorm1d(filters)
        self.pool = nn.AvgPool1d(pool_size, stride=max(1, pool_size // 2))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(filters, 2)
    def forward(self, x):
        x = self.bn(self.temporal(x)).square()
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        return self.classifier(self.dropout(x.mean(dim=-1)))

class CNN1D(nn.Module):
    def __init__(self, filters=16, kernel_size=7, pool_size=2, dropout=0.3):
        super().__init__(); f = filters
        self.features = nn.Sequential(
            nn.Conv1d(1, f, kernel_size, padding="same", bias=False),
            nn.BatchNorm1d(f), nn.ReLU(), nn.MaxPool1d(pool_size),
            nn.Conv1d(f, 2*f, kernel_size, padding="same", bias=False),
            nn.BatchNorm1d(2*f), nn.ReLU(), nn.MaxPool1d(pool_size),
            nn.Conv1d(2*f, 2*f, kernel_size, padding="same", bias=False),
            nn.BatchNorm1d(2*f), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(2*f, 2))
    def forward(self, x): return self.classifier(self.features(x))

class MultiScaleCNN(nn.Module):
    def __init__(self, filters=16, kernel_size=7, pool_size=2, dropout=0.5):
        super().__init__(); f = filters
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(1, f, k, padding="same", bias=False),
                          nn.BatchNorm1d(f), nn.ReLU(), nn.MaxPool1d(pool_size),
                          nn.AdaptiveAvgPool1d(1)) for k in [3, 7, 15]])
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(3*f, 2))
    def forward(self, x):
        return self.classifier(torch.cat([b(x) for b in self.branches], dim=1))

def build_model(cfg):
    if cfg["kind"] == "shallow":
        return ShallowSquareLog(cfg["filters"], cfg["kernel_size"], cfg["pool_size"], cfg["dropout"])
    if cfg["kind"] == "cnn":
        return CNN1D(cfg["filters"], cfg["kernel_size"], cfg["pool_size"], cfg["dropout"])
    if cfg["kind"] == "multiscale":
        return MultiScaleCNN(cfg["filters"], cfg["kernel_size"], cfg["pool_size"], cfg["dropout"])
    raise ValueError(cfg["kind"])''')

code(r'''class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super().__init__(); self.register_buffer("alpha", alpha); self.gamma = gamma
    def forward(self, logits, targets):
        lp = torch.log_softmax(logits, dim=1)
        lpt = lp.gather(1, targets[:, None]).squeeze(1)
        pt = lpt.exp()
        return (-self.alpha[targets] * (1.0 - pt).pow(self.gamma) * lpt).mean()

def make_loss(y_train, focal):
    if not focal: return nn.CrossEntropyLoss()
    c = np.bincount(y_train, minlength=2).astype(np.float32)
    w = c.sum() / (2.0 * np.maximum(c, 1.0)); w = w / w.mean()
    return FocalLoss(torch.tensor(w, dtype=torch.float32, device=DEVICE), gamma=2.0)''')

code(r'''def make_loader(x, y, shuffle):
    return DataLoader(TensorDataset(
        torch.tensor(x, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y, dtype=torch.long)),
        batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"))

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval(); tl, tn = 0.0, 0; ps, ys = [], []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        lg = model(xb); loss = criterion(lg, yb)
        tl += float(loss.item()) * len(yb); tn += len(yb)
        ps.append(torch.softmax(lg, dim=1)[:, 1].cpu().numpy()); ys.append(yb.cpu().numpy())
    y = np.concatenate(ys); p = np.concatenate(ps); pr = (p >= 0.5).astype(int)
    m = {"loss": tl / max(tn, 1),
         "accuracy": accuracy_score(y, pr),
         "balanced_accuracy": balanced_accuracy_score(y, pr),
         "precision_ictal": precision_score(y, pr, zero_division=0),
         "recall_sensitivity_ictal": recall_score(y, pr, zero_division=0),
         "specificity_non_ictal": recall_score(1-y, 1-pr, zero_division=0),
         "f1_ictal": f1_score(y, pr, zero_division=0),
         "confusion_matrix": confusion_matrix(y, pr, labels=[0, 1]).tolist()}
    m["roc_auc"] = roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")
    return m

def fit_one(config, x_train, y_train, x_val, y_val, seed, focal_loss):
    set_seed(seed)
    model = build_model(config).to(DEVICE)
    criterion = make_loss(y_train, focal_loss)
    opt = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                           weight_decay=config["weight_decay"])
    tl = make_loader(x_train, y_train, True); vl = make_loader(x_val, y_val, False)
    bvl, bst, be, st = float("inf"), None, 0, 0; hist = []
    for ep in range(1, EPOCHS_MAX+1):
        model.train(); rl, seen = 0.0, 0
        for xb, yb in tl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb); loss.backward(); opt.step()
            rl += float(loss.item()) * len(yb); seen += len(yb)
        vm = evaluate(model, vl, criterion)
        hist.append({"epoch": ep, "train_loss": rl/max(seen,1),
                     "val_loss": vm["loss"], "val_balanced_accuracy": vm["balanced_accuracy"]})
        if vm["loss"] < bvl - 1e-5:
            bvl = vm["loss"]; st = 0
            bst = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            be = ep
        else:
            st += 1
            if st >= PATIENCE: break
    if bst is None: raise RuntimeError("Pas d'etat sauvegarde.")
    model.load_state_dict(bst)
    return model, be, hist, evaluate(model, vl, criterion)''')
code(r'''def run_benchmark():
    DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)
    x_train_full, y_train_full, g_train_full = load_split("entrainement")
    x_test, y_test, g_test = load_split("test")
    if x_train_full.shape[1] != x_test.shape[1]:
        raise ValueError("Longueurs differentes train/test.")
    print(f"Device : {DEVICE}")
    print(f"Train : {len(y_train_full)} ; test : {len(y_test)}")
    results, best, rid = [], None, 0
    for arch, lr, wd, nm, focal, bm, seed in itertools.product(
        ARCHITECTURES, LEARNING_RATES, WEIGHT_DECAYS,
        ["per_segment", "train_global"],
        ([False, True] if COMPARER_FOCAL_LOSS else [FOCAL_LOSS]),
        ["all", "balanced"], SEEDS):
        if focal and bm == "balanced": continue
        ab = "balanced" if (not focal and bm == "balanced") else "all"
        rid += 1
        print(f"[{rid}] {arch['name']} lr={lr:g} wd={wd:g} norm={nm} focal={focal} bal={ab} seed={seed}")
        tr, va = split_train_validation(x_train_full, y_train_full, g_train_full, seed)
        xt, yt, gt = x_train_full[tr], y_train_full[tr], g_train_full[tr]
        xv, yv = x_train_full[va], y_train_full[va]
        if ab == "balanced":
            k = balance_train_indices(yt, gt, seed); xt, yt = xt[k], yt[k]
        xt, xv, _, nm_meta = normalize_data(xt, xv, x_test, nm)
        cfg = dict(arch); cfg.update({"learning_rate": lr, "weight_decay": wd})
        t0 = time.time()
        model, be, hist, vm = fit_one(cfg, xt, yt, xv, yv, seed, focal)
        el = time.time() - t0
        rec = {"run_id": rid, "architecture": arch["name"], "learning_rate": lr,
               "weight_decay": wd, "normalization": nm, "focal_loss": focal,
               "focal_gamma": 2.0 if focal else None, "balancing_mode": ab,
               "seed": seed, "n_train_after_sampling": len(yt), "n_val": len(yv),
               "best_epoch": be, "runtime_seconds": round(el, 2),
               "normalization_metadata": json.dumps(nm_meta),
               **{f"val_{k}": v for k, v in vm.items() if k != "confusion_matrix"},
               "val_confusion_matrix": json.dumps(vm["confusion_matrix"])}
        results.append(rec)
        sc = (vm["balanced_accuracy"], vm["recall_sensitivity_ictal"])
        if best is None or sc > best["score"]:
            best = {"score": sc,
                    "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "config": cfg, "seed": seed, "normalization": nm_meta,
                    "focal_loss": focal, "balancing_mode": ab,
                    "architecture": arch["name"], "best_epoch": be,
                    "train_class_counts": np.bincount(yt, minlength=2).tolist()}
        pre = (f"{rid:04d}_{arch['name']}_norm-{nm}_focal-{int(focal)}"
               f"_bal-{ab}_lr-{lr:g}_wd-{wd:g}_seed-{seed}")
        with open(DOSSIER_SORTIE / f"{pre}_history.json", "w", encoding="utf-8") as f:
            json.dump(hist, f, indent=2)
        torch.save({"model_state_dict": model.state_dict(), "config": cfg,
                    "best_epoch": be, "normalization": nm_meta},
                   DOSSIER_SORTIE / f"{pre}_model.pt")
        with open(DOSSIER_SORTIE / "results.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader(); w.writerows(results)
    df = pd.DataFrame(results)
    metrics = ["balanced_accuracy","accuracy","precision_ictal",
               "recall_sensitivity_ictal","specificity_non_ictal","f1_ictal","roc_auc"]
    gcols = ["architecture","learning_rate","weight_decay",
             "normalization","focal_loss","balancing_mode"]
    summ = df.groupby(gcols)[metrics].agg(["mean","std"]).reset_index()
    summ.to_csv(DOSSIER_SORTIE / "summary_by_configuration.csv", index=False)
    df.sort_values(["balanced_accuracy","recall_sensitivity_ictal"],
                   ascending=False).to_csv(
        DOSSIER_SORTIE / "results_sorted_validation_selected.csv", index=False)
    print(f"\nTermine : {DOSSIER_SORTIE.resolve()}")
    print(summ.to_string(index=False))
    return df, summ, best''')

code(r'''df, summary, best_candidate = run_benchmark()''')

md(r'''## Analyse des resultats''')

code(r'''top = df.sort_values(["val_balanced_accuracy", "val_recall_sensitivity_ictal"],
                     ascending=False).head(15)
cols = ["run_id","architecture","learning_rate","weight_decay","normalization",
        "focal_loss","balancing_mode","seed","val_balanced_accuracy",
        "val_recall_sensitivity_ictal","val_specificity_non_ictal","val_roc_auc"]
print(top[cols].to_string(index=False))''')

code(r'''fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, metric in zip(axes, ["val_balanced_accuracy", "val_roc_auc"]):
    m = df.groupby("focal_loss")[metric].mean()
    s = df.groupby("focal_loss")[metric].std()
    ax.bar(m.index.astype(str), m.values, yerr=s.values, capsize=6)
    ax.set_title(f"{metric} - focal vs CE"); ax.set_xlabel("focal_loss")
    ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.show()''')

md(r'''## Evaluation finale sur le test (une seule fois)''')

code(r'''best_model = build_model(best_candidate["config"]).to(DEVICE)
best_model.load_state_dict(best_candidate["model_state_dict"])
nm = best_candidate["normalization"]
if nm["mode"] == "per_segment":
    m = x_test.mean(axis=1, keepdims=True)
    s = np.maximum(x_test.std(axis=1, keepdims=True), 1e-8)
    x_test_norm = (x_test - m) / s
else:
    x_test_norm = (x_test - nm["mean_train"]) / nm["std_train"]
crit = make_loss(np.array([0, 1]), best_candidate["focal_loss"]).to(DEVICE)
tml = make_loader(x_test_norm, y_test, shuffle=False)
tmet = evaluate(best_model, tml, crit)
print("=== Test final ===")
print(json.dumps({k: v for k, v in tmet.items() if k != "confusion_matrix"}, indent=2))
print("Confusion [0=non-ictal, 1=ictal]:"); print(np.array(tmet["confusion_matrix"]))''')

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3"}},
      "nbformat": 4, "nbformat_minor": 5}
out = "benchmark_shallow_cnn.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Notebook cree : {out} ({len(cells)} cellules)")
