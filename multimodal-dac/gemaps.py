#imports
import re
import os
import csv
import pandas as pd
import numpy as np
import json

from pathlib import Path
from typing import List, Dict, Optional, Any

from torch.utils.data import Dataset, DataLoader, TensorDataset
from collections import Counter

from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

import IPython.display as ipd
import torchaudio
import torchaudio.transforms as T

import evaluate
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report, f1_score

import matplotlib.pyplot as plt
import seaborn as sns

#grid for GeMAPs
AUDIO_GRID = {
    "dropout": [0.1, 0.2, 0.3],
    "batch_size": [16, 32],
    "learning_rate": [1e-3, 1e-4, 1e-5]
}

BASE_RESULTS_DIR = Path(
    "/home1/anniegao/multimodal-scripted-vs-improvised/multimodal-dac/RESULTS_AUDIO"
)
BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALL_RESULTS_PATH = BASE_RESULTS_DIR / "GEMAPS_RESULTS_R22_R42.csv"

device = torch.device("cpu")

final_df = pd.read_csv("GeMAPs/final_df.csv")

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
    - Build GeMAPs datasets
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
        dfs = []
        for s in scenario_list:
            # load CSV for this scenario
            df = pd.read_csv(f"GeMAPs/{s}.csv")
            # map string labels to ints
            df["labels"] = df["labels"].map(label2id)
            dfs.append(df)
        # combine all scenarios into one DataFrame
        full_df = pd.concat(dfs, ignore_index=True)
        
        #drop filename
        full_df = full_df.drop('filename', axis=1)
        #map labels
        #full_df["labels"] = full_df["labels"].map(label2id)
        features, labels = full_df.drop(["labels", "Unnamed: 0"], axis=1).values, full_df["labels"].values

        #feature scaling
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        #create TensorDatasets
        dataset = TensorDataset(features_tensor, labels_tensor)

        return dataset

    #TensorDatasets from helper function
    train_dataset = load_scenarios(train_scenarios)
    val_dataset = load_scenarios(val_scenarios)
    test_dataset = load_scenarios(test_scenarios)
    cross_dataset = load_scenarios(cross_scenarios)

    #create Dataloaders - batch size is a hyperparameter
    train_loader = DataLoader(train_dataset, batch_size=hp["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hp["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=hp["batch_size"], shuffle=False)
    cross_loader = DataLoader(cross_dataset, batch_size=hp["batch_size"], shuffle=False)

    #FFNN for training on GeMAPs features
    class SimpleNN(nn.Module):
        def __init__(self, feature_dim=88, num_classes=None):
            super().__init__()
            self.fc1 = nn.Linear(feature_dim, 128)
            self.bn1 = nn.BatchNorm1d(128)
            self.dropout1 = nn.Dropout(hp["dropout"])
    
            self.fc2 = nn.Linear(128, 64)
            self.bn2 = nn.BatchNorm1d(64)
            self.dropout2 = nn.Dropout(hp["dropout"])
    
            self.fc3 = nn.Linear(64, num_classes)
    
        def forward(self, x):
            x = torch.flatten(x, 1)
    
            x = self.fc1(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.dropout1(x)
    
            x = self.fc2(x)
            x = self.bn2(x)
            x = F.relu(x)
            x = self.dropout2(x)
    
            x = self.fc3(x)
            return x

    #training
    model = SimpleNN(feature_dim=88, num_classes=len(label2id))
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4, lr=hp["learning_rate"]) #hyperparameter
    criterion = nn.CrossEntropyLoss()
    num_epochs = 5

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
    
        return accuracy, f1_macro

    val_acc, val_f1 = evaluate(model, val_loader, device)
    test_acc, test_f1 = evaluate(model, test_loader, device)
    cross_acc, cross_f1 = evaluate(model, cross_loader, device)

    metrics = {
        "val_accuracy":   val_acc,
        "val_f1_macro":   val_f1,
        "test_accuracy":  test_acc,
        "test_f1_macro":  test_f1,
        "cross_accuracy": cross_acc,
        "cross_f1_macro": cross_f1,
    }

    print(f"[{run_id}] Metrics:", metrics)

    if not hp_search:
        # Two rows: TEST + CROSS
        with open(ALL_RESULTS_PATH, "a", newline="") as f:
            w = csv.writer(f)

            # SAME-STYLE TEST
            w.writerow([
                run_id,
                "gemaps",
                "TEST",
                hp["learning_rate"],
                hp["batch_size"],
                hp["dropout"],
                metrics["val_accuracy"],
                metrics["val_f1_macro"],
                metrics["test_accuracy"],
                metrics["test_f1_macro"],
            ])

            # CROSS-STYLE TEST
            w.writerow([
                run_id,
                "gemaps",
                "CROSS",
                hp["learning_rate"],
                hp["batch_size"],
                hp["dropout"],
                metrics["val_accuracy"],
                metrics["val_f1_macro"],
                metrics["cross_accuracy"],
                metrics["cross_f1_macro"],
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
    #load CSV for dataframe
    final_df = pd.read_csv("GeMAPs/final_df.csv")
    
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
            "accuracy",
            "f1_macro",
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
    