"""
Hyperparameter search for the Wefabricate CNN (Assignment 2).

Runs TWO hyperparameter optimization methods on the same search space, with the
same 5-fold stratified cross-validation and the SAME number of epochs per fold,
so the comparison between them is fair:

  1. Random Search   (Optuna RandomSampler)
  2. TPE             (Optuna TPESampler -- a Bayesian method)

For each method it: runs the search, saves the trials/folds/best-config, plots the
validation-accuracy-over-trials curve, then retrains a final model on the full
training set using that method's own best hyperparameters and saves both the
weights and the learning curves.

Run it from the repository root (the folder that contains baseline.ipynb and the
"WF-data and support code" folder):

    python hyperparameter_search.py
"""

import os
import sys
import json
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import StratifiedKFold

import optuna
from optuna.samplers import RandomSampler, TPESampler


# ----------------------------------------------------------------------------
# Settings (defaults are the real run; can be overridden with env vars for a
# quick smoke test, e.g. CV_EPOCHS=2 N_TRIALS=2 FINAL_EPOCHS=2)
# ----------------------------------------------------------------------------
SEED = 42
N_TRIALS = int(os.environ.get("N_TRIALS", 10))      # candidate configs per method
CV_EPOCHS = int(os.environ.get("CV_EPOCHS", 30))    # epochs per fold during search (SAME for both methods)
N_SPLITS = int(os.environ.get("N_SPLITS", 5))       # cross-validation folds
FINAL_EPOCHS = int(os.environ.get("FINAL_EPOCHS", 30))  # epochs to train each final model

device = ("cuda" if torch.cuda.is_available()
          else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
          else "cpu")


def set_seed(seed):
    """Make the run as reproducible as possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# Paths -- figure out where we are and where to write outputs
# ----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent          # repository root
SUPPORT_DIR = SCRIPT_DIR / "WF-data and support code"  # holds support.py + WF-data

FIGURE_DIR = SCRIPT_DIR / "figures"
RS_DIR = SCRIPT_DIR / "random_search_results"
TPE_DIR = SCRIPT_DIR / "tpe_search_results"
for d in (FIGURE_DIR, RS_DIR, TPE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# support.py loads the images with the relative path "WF-data/train", so we must
# import it AND run from inside its folder.
sys.path.insert(0, str(SUPPORT_DIR))
os.chdir(SUPPORT_DIR)
from support import load_dataset  # noqa: E402


# ----------------------------------------------------------------------------
# Model + train/eval helpers (identical to baseline.ipynb, inlined so this
# script is self-contained and does not depend on the notebook)
# ----------------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self, conv_channels=(32, 64), hidden=128, dropout=0.5, num_classes=2):
        super().__init__()
        c1, c2 = conv_channels
        self.conv1a = nn.Conv2d(3, c1, kernel_size=3, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        # 60x30 input -> 15x7 after two 2x2 pools
        self.fc1 = nn.Linear(c2 * 15 * 7, hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(dim=1) == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total


# ----------------------------------------------------------------------------
# Load data + build the fixed CV folds (shared by both search methods)
# ----------------------------------------------------------------------------
set_seed(SEED)
train_dataset, test_dataset = load_dataset()
labels = np.array(train_dataset.targets)
class_names = train_dataset.classes
print(f"device={device} | train={len(train_dataset)} test={len(test_dataset)} | "
      f"CV_EPOCHS={CV_EPOCHS} N_TRIALS={N_TRIALS} N_SPLITS={N_SPLITS} FINAL_EPOCHS={FINAL_EPOCHS}")

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
splits = list(cv.split(np.zeros(len(labels)), labels))


# ----------------------------------------------------------------------------
# Search space (the five hyperparameters)
# ----------------------------------------------------------------------------
SEARCH_SPACE_DESCRIPTION = {
    "lr": "log-uniform [1e-4, 3e-3]",
    "batch_size": "[8, 16, 32]",
    "base_filters": "[16, 32, 48, 64]",
    "dropout": "uniform [0.1, 0.6]",
    "hidden": "[64, 128, 256]",
}


def make_baseline_config(params):
    """Turn the five sampled values into the arguments the CNN/optimizer need."""
    base_filters = int(params["base_filters"])
    return {
        "lr": float(params["lr"]),
        "batch_size": int(params["batch_size"]),
        "base_filters": base_filters,
        "dropout": float(params["dropout"]),
        "hidden": int(params["hidden"]),
        "conv_channels": (base_filters, 2 * base_filters),
    }


def sample_config(trial):
    params = {
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "base_filters": trial.suggest_categorical("base_filters", [16, 32, 48, 64]),
        "dropout": trial.suggest_float("dropout", 0.10, 0.60),
        "hidden": trial.suggest_categorical("hidden", [64, 128, 256]),
    }
    return make_baseline_config(params)


def calculate_cv(config, trial_number=None):
    """5-fold CV mean validation accuracy for one configuration."""
    fold_scores, fold_results = [], []
    for fold_number, (train_idx, val_idx) in enumerate(splits, start=1):
        set_seed(SEED + fold_number)
        trainset = Subset(train_dataset, train_idx)
        valset = Subset(train_dataset, val_idx)
        train_loader = DataLoader(trainset, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(valset, batch_size=config["batch_size"], shuffle=False)

        model = CNN(conv_channels=config["conv_channels"],
                    hidden=config["hidden"],
                    dropout=config["dropout"]).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])

        best_val_acc, best_epoch = 0.0, 0
        best_train_loss = best_val_loss = 0.0
        for epoch in range(1, CV_EPOCHS + 1):
            train_loss, _ = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            if val_acc > best_val_acc:
                best_val_acc, best_epoch = val_acc, epoch
                best_train_loss, best_val_loss = train_loss, val_loss

        fold_scores.append(best_val_acc)
        fold_results.append({
            "trial": trial_number, "fold": fold_number,
            "best_val_acc": best_val_acc, "best_epoch": best_epoch,
            "best_train_loss": best_train_loss, "best_val_loss": best_val_loss,
            "lr": config["lr"], "batch_size": config["batch_size"],
            "base_filters": config["base_filters"], "dropout": config["dropout"],
            "hidden": config["hidden"],
        })
    return float(np.mean(fold_scores)), float(np.std(fold_scores)), fold_scores, fold_results


def run_search(method_name, sampler, out_dir):
    """Run one Optuna study and save trials, folds, best config, and the curve."""
    print(f"\n{'='*60}\nRunning {method_name}\n{'='*60}")
    fold_rows = []

    def objective(trial):
        config = sample_config(trial)
        mean_acc, std_acc, fold_scores, fold_results = calculate_cv(config, trial.number)
        fold_rows.extend(fold_results)
        trial.set_user_attr("mean_val_acc", mean_acc)
        trial.set_user_attr("std_val_acc", std_acc)
        trial.set_user_attr("fold_scores", fold_scores)
        print(f"  trial {trial.number:2d}: mean CV acc = {mean_acc*100:5.1f}%  config={config}")
        return mean_acc

    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name=f"{method_name}_5fold")
    tic = time.perf_counter()
    study.optimize(objective, n_trials=N_TRIALS)
    elapsed = time.perf_counter() - tic

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    trials_df.to_csv(out_dir / f"{method_name}_trials.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out_dir / f"{method_name}_folds.csv", index=False)

    best_cfg = make_baseline_config(study.best_params)
    best_config = {
        "best_trial": study.best_trial.number,
        "mean_validation_accuracy": study.best_value,
        "five_hyperparameters": study.best_params,
        "baseline_model_arguments": {
            "conv_channels": best_cfg["conv_channels"],
            "hidden": best_cfg["hidden"],
            "dropout": best_cfg["dropout"],
        },
        "optimizer_arguments": {"lr": best_cfg["lr"]},
        "data_loader_arguments": {"batch_size": best_cfg["batch_size"]},
        "n_trials": N_TRIALS, "cv_epochs": CV_EPOCHS, "n_splits": N_SPLITS,
        "elapsed_seconds": elapsed,
        "search_space": SEARCH_SPACE_DESCRIPTION,
    }
    with open(out_dir / f"best_{method_name}_config.json", "w") as f:
        json.dump(best_config, f, indent=2)

    # validation-accuracy-over-trials curve
    p = trials_df.sort_values("number").copy()
    p["best_so_far"] = p["value"].cummax()
    plt.figure(figsize=(7, 4))
    plt.plot(p["number"], p["value"] * 100, marker="o", label="trial mean CV accuracy")
    plt.plot(p["number"], p["best_so_far"] * 100, marker="s", label="best so far")
    plt.xlabel("trial"); plt.ylabel("5-fold mean validation accuracy (%)")
    plt.title(f"{method_name} validation over trials ({N_TRIALS} trials, {CV_EPOCHS} epochs/fold)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / f"{method_name}_validation_curve.png", bbox_inches="tight", dpi=150)
    plt.close()

    print(f"  -> best trial {study.best_trial.number}: "
          f"{study.best_value*100:.1f}% CV  ({elapsed:.0f}s)")
    return best_cfg, study.best_value


def train_final_model(config, weights_path, title, fig_path):
    """Train one final model on the full training set and save weights + curves."""
    set_seed(SEED)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    model = CNN(conv_channels=config["conv_channels"], hidden=config["hidden"],
                dropout=config["dropout"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()

    tr_loss, tr_acc, te_loss, te_acc = [], [], [], []
    for epoch in range(1, FINAL_EPOCHS + 1):
        l, a = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl, va = evaluate(model, test_loader, criterion, device)
        tr_loss.append(l); tr_acc.append(a); te_loss.append(vl); te_acc.append(va)
    torch.save(model.state_dict(), weights_path)

    best_ep = int(np.argmax(te_acc)) + 1
    best_acc = max(te_acc) * 100
    ep = range(1, FINAL_EPOCHS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(ep, tr_loss, label="train", color="steelblue")
    ax1.plot(ep, te_loss, label="test", color="tomato", linestyle="--")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("cross-entropy loss")
    ax1.set_title(f"{title} - Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(ep, [v*100 for v in tr_acc], label="train", color="steelblue")
    ax2.plot(ep, [v*100 for v in te_acc], label="test", color="tomato", linestyle="--")
    ax2.axvline(best_ep, color="green", linestyle=":", linewidth=1.5,
                label=f"best test ({best_acc:.1f}% @ ep{best_ep})")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy (%)"); ax2.set_ylim(0, 105)
    ax2.set_title(f"{title} - Accuracy"); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  {title}: best test {best_acc:.1f}% @ ep{best_ep} -> {weights_path.name}")
    return best_acc, best_ep


# ----------------------------------------------------------------------------
# Run both searches (same folds, same epochs) and train each final model with
# its OWN best config.
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    rs_cfg, rs_cv = run_search("random_search", RandomSampler(seed=SEED), RS_DIR)
    tpe_cfg, tpe_cv = run_search("tpe_search", TPESampler(seed=SEED, n_startup_trials=3), TPE_DIR)

    print(f"\n{'='*60}\nTraining final models\n{'='*60}")
    rs_test, rs_ep = train_final_model(
        rs_cfg, SCRIPT_DIR / "model_with_random_search.pth",
        "Optimized CNN (Random Search)", SCRIPT_DIR / "optimized_random_search_learning_curves.png")
    tpe_test, tpe_ep = train_final_model(
        tpe_cfg, SCRIPT_DIR / "model_with_tpe_search.pth",
        "Optimized CNN (TPE Search)", SCRIPT_DIR / "optimized_tpe_search_learning_curves.png")

    print(f"\n{'='*60}\nSUMMARY (all at {CV_EPOCHS} epochs/fold, {N_TRIALS} trials)\n{'='*60}")
    print(f"{'Method':<16}{'best CV acc':>14}{'final test acc':>18}")
    print(f"{'Random Search':<16}{rs_cv*100:>13.1f}%{rs_test:>16.1f}%")
    print(f"{'TPE':<16}{tpe_cv*100:>13.1f}%{tpe_test:>16.1f}%")
    print(f"\nRandom Search best config: {rs_cfg}")
    print(f"TPE best config:           {tpe_cfg}")
