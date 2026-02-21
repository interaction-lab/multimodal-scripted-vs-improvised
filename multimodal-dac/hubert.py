#imports
import re
import os
import csv
import pandas as pd
import numpy as np
import json

from pathlib import Path
from typing import List, Dict, Optional, Any

import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter

import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report, f1_score, precision_score, recall_score

import matplotlib.pyplot as plt
import seaborn as sns

#grid for HuBERT
AUDIO_GRID = {
    "dropout": [0.1, 0.2],
    "batch_size": [16, 32],
    "learning_rate": [1e-3, 1e-4, 1e-5]
}

BASE_RESULTS_DIR = Path(
    "/home1/anniegao/multimodal-scripted-vs-improvised/multimodal-dac/RESULTS_AUDIO"
)
BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALL_RESULTS_PATH = BASE_RESULTS_DIR / "HUBERT_RESULTS_R22_R42.csv"

device = torch.device("cpu")

final_df = pd.read_csv("HuBERT/final_df.csv")

# ======================================================
# SPLITS – AUDIO RUNS R22-42
# ======================================================

# Scenario-based; cross_test = same scenario number but opposite style.
SPLITS_AUDIO: Dict[str, Dict[str, Any]] = {
    # SCRIPTED ONLY (block S)
    "R22": dict(level="scenario", train_type="S", val_type="S", test_type="S",
                train=[1, 2, 3, 4, 5], val=[6], test=[7], cross_test=[7]),
    "R23": dict(level="scenario", train_type="S", val_type="S", test_type="S",
                train=[2, 3, 4, 5, 6], val=[7], test=[1], cross_test=[1]),
    "R24": dict(level="scenario", train_type="S", val_type="S", test_type="S",
                train=[3, 4, 5, 6, 7], val=[1], test=[2], cross_test=[2]),
    "R25": dict(level="scenario", train_type="S", val_type="S", test_type="S",
                train=[1, 4, 5, 6, 7], val=[2], test=[3], cross_test=[3]),
    "R26": dict(level="scenario", train_type="S", val_type="S", test_type="S",
                train=[1, 2, 5, 6, 7], val=[3], test=[4], cross_test=[4]),
    "R27": dict(level="scenario", train_type="S", val_type="S", test_type="S",
                train=[1, 2, 3, 6, 7], val=[4], test=[5], cross_test=[5]),
    "R28": dict(level="scenario", train_type="S", val_type="S", test_type="S",
                train=[1, 2, 3, 4, 7], val=[5], test=[6], cross_test=[6]),

    # IMPROVISED ONLY (block I)
    "R29": dict(level="scenario", train_type="I", val_type="I", test_type="I",
                train=[1, 2, 3, 4, 5], val=[6], test=[7], cross_test=[7]),
    "R30": dict(level="scenario", train_type="I", val_type="I", test_type="I",
                train=[2, 3, 4, 5, 6], val=[7], test=[1], cross_test=[1]),
    "R31": dict(level="scenario", train_type="I", val_type="I", test_type="I",
                train=[3, 4, 5, 6, 7], val=[1], test=[2], cross_test=[2]),
    "R32": dict(level="scenario", train_type="I", val_type="I", test_type="I",
                train=[1, 4, 5, 6, 7], val=[2], test=[3], cross_test=[3]),
    "R33": dict(level="scenario", train_type="I", val_type="I", test_type="I",
                train=[1, 2, 5, 6, 7], val=[3], test=[4], cross_test=[4]),
    "R34": dict(level="scenario", train_type="I", val_type="I", test_type="I",
                train=[1, 2, 3, 6, 7], val=[4], test=[5], cross_test=[5]),
    "R35": dict(level="scenario", train_type="I", val_type="I", test_type="I",
                train=[1, 2, 3, 4, 7], val=[5], test=[6], cross_test=[6]),

    # MIXED (S + I) – train/val on both; test_type "I/S" means:
    #   test       → scripted version of scenario N
    #   cross_test → improvised version of scenario N
    "R36": dict(level="scenario", train_type="S+I", val_type="S+I", test_type="S",
                train=[1, 2, 3, 4, 5], val=[6], test=[7], cross_test=[7]),
    "R37": dict(level="scenario", train_type="S+I", val_type="S+I", test_type="S",
                train=[2, 3, 4, 5, 6], val=[7], test=[1], cross_test=[1]),
    "R38": dict(level="scenario", train_type="S+I", val_type="S+I", test_type="S",
                train=[3, 4, 5, 6, 7], val=[1], test=[2], cross_test=[2]),
    "R39": dict(level="scenario", train_type="S+I", val_type="S+I", test_type="S",
                train=[1, 4, 5, 6, 7], val=[2], test=[3], cross_test=[3]),
    "R40": dict(level="scenario", train_type="S+I", val_type="S+I", test_type="S",
                train=[1, 2, 5, 6, 7], val=[3], test=[4], cross_test=[4]),
    "R41": dict(level="scenario", train_type="S+I", val_type="S+I", test_type="S",
                train=[1, 2, 3, 6, 7], val=[4], test=[5], cross_test=[5]),
    "R42": dict(level="scenario", train_type="S+I", val_type="S+I", test_type="S",
                train=[1, 2, 3, 4, 7], val=[5], test=[6], cross_test=[6]),
}

SCRIPTED_BLOCK = ["R22", "R23", "R24", "R25", "R26", "R27", "R28"]
IMPROV_BLOCK   = ["R29", "R30", "R31", "R32", "R33", "R34", "R35"]
MIXED_BLOCK    = ["R36", "R37", "R38", "R39", "R40", "R41", "R42"]

# ======================================================
# CORE: ONE AUDIO EXPERIMENT
# ======================================================
def run_audio_experiment(
    run_id: str,
    cfg: dict,
    df_video: pd.DataFrame,
    hp: dict,
    hp_search: bool = False,
) -> Dict[str, float]:
    """
    One experiment for a given run_id
    - Make train/val/test/cross splits
    - Build HuBERT datasets
    - Train
    - Evaluate on val, test, cross
    - If hp_search=False: append TEST + CROSS rows to CSV

    hp must contain dropout, batch size, learning rate
    """

    #define train, val, test scenarios
    #helper function
    def expand_scenarios(scenarios, split_type):
        """
        scenarios: list[int]
        split_type: one of {"S", "I", "S+I", "I/S"}
        """
        out = []
        for n in scenarios:
            if split_type == "S":
                out.append(f"{n}s")
            elif split_type == "I":
                out.append(f"{n}i")
            elif split_type in {"S+I", "I/S"}:
                out.extend([f"{n}s", f"{n}i"])
            else:
                raise ValueError(f"Unknown split_type: {split_type}")
        return out

    split = SPLITS_AUDIO["R22"]

    train_scenarios = expand_scenarios(split["train"], split["train_type"])
    val_scenarios   = expand_scenarios(split["val"], split["val_type"])
    test_scenarios  = expand_scenarios(split["test"], split["test_type"])
    cross_scenarios = []
    if split["test_type"] == "I":
      cross_scenarios = expand_scenarios(split["cross_test"], "S")
    elif split["test_type"] == "S":
      cross_scenarios = expand_scenarios(split["cross_test"], "I")
    else: #"S+I", test on "S", cross on "I"
      cross_scenarios = expand_scenarios(split["cross_test"], "I")
    
    label_list = sorted(final_df["EDA"].unique())
    label2id = {label: i for i, label in enumerate(label_list)}
    print("Label mapping:", label2id)

    #load scenarios
    #helper function
    def load_scenarios(scenario_list):
        X, y = [], []
        for s in scenario_list:
            data = torch.load(f"HuBERT/{s}.pt")
            for item in data:
                X.append(item["x"])
                y.append(label2id[item["y"]])  # convert string label → int
        X = torch.stack(X)  # [num_samples, 768]
        y = torch.tensor(y, dtype=torch.long)  # numeric labels
        return X, y

    X_train, y_train = load_scenarios(train_scenarios)
    X_val, y_val = load_scenarios(val_scenarios)
    X_test, y_test = load_scenarios(test_scenarios)
    X_cross, y_cross = load_scenarios(cross_scenarios)
    
    print("Train:", X_train.shape, y_train.shape)
    print("Val:", X_val.shape, y_val.shape)
    print("Test:", X_test.shape, y_test.shape)
    print("Cross:", X_cross.shape, y_cross.shape)

    #load embeddings
    class EmbeddingDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    batch_size = hp["batch_size"] #hyperparameter
    
    train_loader = DataLoader(EmbeddingDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(EmbeddingDataset(X_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(EmbeddingDataset(X_test, y_test), batch_size=batch_size)
    cross_loader = DataLoader(EmbeddingDataset(X_cross, y_cross), batch_size=batch_size)

    #FFNN for training HuBERT embeddings
    class SimpleNN(nn.Module):
        def __init__(self, input_dim=768, hidden_dim=128, num_classes=12):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, num_classes)
            self.dropout = nn.Dropout(hp["dropout"]) #hyperparameter
    
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x

    #training
    model = SimpleNN(input_dim=768, hidden_dim=128, num_classes=12).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"]) #hyperparameter
    criterion = nn.CrossEntropyLoss()
    num_epochs = 7

    for epoch in range(num_epochs):
        # ---- Train ----
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)
    
        # ---- Validation ----
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                loss = criterion(outputs, yb)
                val_loss += loss.item() * xb.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / total
        val_loss /= len(val_loader.dataset)
    
        print(f"Epoch {epoch+1}/{num_epochs} — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    #evaluation
    def evaluate(model, loader, device):
        model.eval()
        all_preds = []
        all_labels = []
    
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
    
                outputs = model(xb)
                preds = outputs.argmax(dim=1)
    
                all_preds.append(preds.cpu())
                all_labels.append(yb.cpu())
    
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
    
        accuracy = (all_preds == all_labels).mean()
        f1_macro = f1_score(all_labels, all_preds, average="macro")
        precision_macro = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        recall_macro = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    
        return accuracy, f1_macro, precision_macro, recall_macro

    val_acc, val_f1, val_precision, val_recall = evaluate(model, val_loader, device)
    test_acc, test_f1, test_precision, test_recall = evaluate(model, test_loader, device)
    cross_acc, cross_f1, cross_precision, cross_recall = evaluate(model, cross_loader, device)

    metrics = {
        "val_accuracy":   val_acc,
        "val_f1_macro":   val_f1,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "test_accuracy":  test_acc,
        "test_f1_macro":  test_f1,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "cross_accuracy": cross_acc,
        "cross_f1_macro": cross_f1,
        "cross_precision": cross_precision,
        "cross_recall": cross_recall,
    }

    print(f"[{run_id}] Metrics:", metrics)

    if not hp_search:
        # Two rows: TEST + CROSS
        with open(ALL_RESULTS_PATH, "a", newline="") as f:
            w = csv.writer(f)

            # SAME-STYLE TEST
            w.writerow([
                run_id,
                "hubert",
                "TEST",
                hp["learning_rate"],
                hp["batch_size"],
                hp["dropout"],
                metrics["val_accuracy"],
                metrics["val_f1_macro"],
                metrics["val_precision"],
                metrics["val_recall"],
                metrics["test_accuracy"],
                metrics["test_f1_macro"],
                metrics["test_precision"],
                metrics["test_recall"],
            ])

            # CROSS-STYLE TEST
            w.writerow([
                run_id,
                "hubert",
                "CROSS",
                hp["learning_rate"],
                hp["batch_size"],
                hp["dropout"],
                metrics["val_accuracy"],
                metrics["val_f1_macro"],
                metrics["val_precision"],
                metrics["val_recall"],
                metrics["cross_accuracy"],
                metrics["cross_f1_macro"],
                metrics["cross_precision"],
                metrics["cross_recall"],
            ])

        print(f"[{run_id}] Saved TEST + CROSS rows to CSV.")

    return metrics

# ======================================================
# GRID SEARCH UTILITIES
# ======================================================

def grid_search_for_block(
    rep_run_id: str,
    cfg: dict,
    df_video: pd.DataFrame,
) -> dict:
    """
    Grid search for a representative run in a block:
      - rep_run_id: "R43" (scripted), "R50" (improv), or "R57" (mixed).
      - returns best hp dict.
    """
    print(f"\n######## GRID SEARCH for {rep_run_id} ########")

    best_hp: Optional[dict] = None
    best_metrics: Optional[Dict[str, float]] = None

    for dp in AUDIO_GRID["dropout"]:
        for bs in AUDIO_GRID["batch_size"]:
            for lr in AUDIO_GRID["learning_rate"]:
                hp = dict(
                    dropout=dp,
                    batch_size=bs,
                    learning_rate=lr
                )
                print(f"Try hp: lr={lr}, dropout={dp}, bs={bs}")

                metrics = run_audio_experiment(
                    rep_run_id,
                    cfg,
                    df_video,
                    hp,
                    hp_search=True,
                )
                if np.isnan(metrics["val_f1_macro"]):
                    continue
                if (best_metrics is None) or (
                    metrics["val_f1_macro"] > best_metrics["val_f1_macro"]
                ):
                    best_metrics = metrics
                    best_hp = hp
                    print(
                        f"NEW BEST (val_f1={metrics['val_f1_macro']:.4f}) "
                        f"with hp={hp}"
                    )
    print(f"Best HP for {rep_run_id}: {best_hp}, metrics={best_metrics}")
    return best_hp if best_hp is not None else {}
    

# ======================================================
# MAIN
# ======================================================

def main():
    print("==========================================")
    print(f"SLURM_JOB_ID         = {os.environ.get('SLURM_JOB_ID', 'N/A')}")
    print(f"SLURM_JOB_NODELIST   = {os.environ.get('SLURM_JOB_NODELIST', 'N/A')}")
    print(f"TMPDIR               = {os.environ.get('TMPDIR', 'N/A')}")
    print("==========================================")

    #load CSV for dataframe
    final_df = pd.read_csv("HuBERT/final_df.csv")
    
    #initialize results CSV (overwrite each new run)
    with open(ALL_RESULTS_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id",
            "model",
            "eval_type",        # TEST or CROSS
            "learning_rate",
            "batch_size",
            "dropout",
            "val_accuracy",
            "val_f1_macro",
            "val_precision",
            "val_recall",
            "accuracy",
            "f1_macro",
            "precision",
            "recall",
        ])

    print("✅ Audio experiment pipeline ready.")
    print("✅ Result CSV:", ALL_RESULTS_PATH)
    print("✅ Grid size per block:", len(AUDIO_GRID["dropout"])
        * len(AUDIO_GRID["batch_size"])
        * len(AUDIO_GRID["learning_rate"]))

    # -------------------------------
    # Block 1: SCRIPTED (R22-28), grid on R22
    # -------------------------------
    print("\n====== SCRIPTED BLOCK (R22–R28) ======")
    best_S = grid_search_for_block("R22", SPLITS_AUDIO["R22"], final_df)

    for run_id in SCRIPTED_BLOCK:
        cfg = SPLITS_AUDIO[run_id]
        print(f"\n=== FINAL AUDIO RUN (SCRIPTED) {run_id} ===")
        run_audio_experiment(run_id, cfg, final_df, best_S, hp_search=False)

    # -------------------------------
    # Block 2: IMPROV (R50–R56), grid on R50
    # -------------------------------
    print("\n====== IMPROV BLOCK (R29–R35) ======")
    best_I = grid_search_for_block("R29", SPLITS_AUDIO["R29"], final_df)

    for run_id in IMPROV_BLOCK:
        cfg = SPLITS_AUDIO[run_id]
        print(f"\n=== FINAL AUDIO RUN (IMPROV) {run_id} ===")
        run_audio_experiment(run_id, cfg, final_df, best_I, hp_search=False)

    # -------------------------------
    # Block 3: MIXED (R57–R63), grid on R57
    # -------------------------------
    print("\n====== MIXED BLOCK (R36-42) ======")
    best_SI = grid_search_for_block("R36", SPLITS_AUDIO["R36"], final_df)

    for run_id in MIXED_BLOCK:
        cfg = SPLITS_AUDIO[run_id]
        print(f"\n=== FINAL AUDIO RUN (MIXED) {run_id} ===")
        run_audio_experiment(run_id, cfg, final_df, best_SI, hp_search=False)

    print("\nDONE — All AUDIO runs completed.")

if __name__ == "__main__":
    main()