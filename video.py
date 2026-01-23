#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VideoMAE-based dialog act classification on IEMOCAP utterance videos.

- Scenario-based runs: R43–R63
  * R43–R49  : scripted-only (S)
  * R50–R56  : improv-only (I)
  * R57–R63  : mixed S+I (train/val on both, test impro, cross-test scripted)

- For each run:
  * Train on scenario-based split
  * Evaluate on same-style TEST
  * Evaluate on CROSS-STYLE (opposite conversation style, same scenarios)

- Grid search (block-wise) on:
    * learning_rate
    * num_train_epochs
    * batch_size
    * num_frames
    * sample_rate

  Grid is run on:
    - R43 for scripted block (applied to R43–R49)
    - R50 for improv block   (applied to R50–R56)
    - R57 for mixed block    (applied to R57–R63)

- Results written to a CSV:
    ALL_VIDEO_RESULTS_R43_R63.csv

  Each run contributes TWO rows:
    - TEST   : same-style test
    - CROSS  : opposite-style test (cross-test)
"""

import os
import re
import csv
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Any

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset

# decord optional
try:
    import decord
    from decord import VideoReader
    HAVE_DECORD = True
except Exception:
    HAVE_DECORD = False
    warnings.warn("decord not available; falling back to torchvision (slower).")

from torchvision.io import read_video
from torchvision.transforms import Lambda, Resize

import evaluate
from sklearn.metrics import f1_score, confusion_matrix

from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification,
    TrainingArguments,
    Trainer,
)

# ======================================================
# CONFIG – PATHS, LABELS, GRID
# ======================================================

# Root folder where your per-utterance videos live
# Example: /scratch1/anurades/utt/video/Session1/Ses01F_impro01/Ses01F_impro01_M004.mp4
VIDEO_ROOT = Path("/scratch1/anurades/utt/video")

# CSV with TURN + dialog acts
LABEL_CSV = Path(
    "/home1/anurades/multimodal-scripted-vs-improvised/multimodal-dac/all_sessions.csv"
)

# Column that contains your 12-class dialog acts:
#   g, ap, c, q, ans, ag, dag, o, s, a, b, oth
LABEL_COL = "DA"  # <-- change this if your 12-class column has a different name

# Where to store the VIDEO CSV results (small, OK on /home1)
BASE_RESULTS_DIR = Path(
    "/home1/anurades/multimodal-scripted-vs-improvised/multimodal-dac/RESULTS_VIDEO"
)
BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALL_RESULTS_PATH = BASE_RESULTS_DIR / "ALL_VIDEO_RESULTS_R43_R63.csv"

# Where to keep VideoMAE checkpoints / logs (better on scratch to avoid quota issues)
VIDEO_MODEL_BASE = Path("/scratch1/anurades/videomae-dac-checkpoints")
VIDEO_MODEL_BASE.mkdir(parents=True, exist_ok=True)

# Local VideoMAE checkpoint (adjust if you moved it)
VMAE_BACKBONE = "/scratch1/anurades/hf_models/videomae-large"

# 12-class labels you specified
TRUE_LABELS = ["g", "ap", "c", "q", "ans", "ag", "dag", "o", "s", "a", "b", "oth"]

# Grid for VideoMAE
VIDEO_GRID = {
    "learning_rate":    [5e-5, 1e-5],
    "num_train_epochs": [5],
    "batch_size":       [4],
    "num_frames":       [16],
    "sample_rate":      [2, 4],
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================
# SPLITS – VIDEO RUNS R43–R63 (mirror text R01–R21)
# ======================================================

# Scenario-based; cross_test = same scenario number but opposite style.
SPLITS_VIDEO: Dict[str, Dict[str, Any]] = {
    # SCRIPTED ONLY (block S)
    "R43": dict(level="scenario", train_type="S", val_type="S", test_type="S",
                train=[1, 2, 3, 4, 5], val=[6], test=[7], cross_test=[7]),
    "R44": dict(level="scenario", train_type="S", val_type="S", test_type="S",
                train=[2, 3, 4, 5, 6], val=[7], test=[1], cross_test=[1]),
    "R45": dict(level="scenario", train_type="S", val_type="S", test_type="S",
                train=[3, 4, 5, 6, 7], val=[1], test=[2], cross_test=[2]),
    "R46": dict(level="scenario", train_type="S", val_type="S", test_type="S",
                train=[1, 4, 5, 6, 7], val=[2], test=[3], cross_test=[3]),
    "R47": dict(level="scenario", train_type="S", val_type="S", test_type="S",
                train=[1, 2, 5, 6, 7], val=[3], test=[4], cross_test=[4]),
    "R48": dict(level="scenario", train_type="S", val_type="S", test_type="S",
                train=[1, 2, 3, 6, 7], val=[4], test=[5], cross_test=[5]),
    "R49": dict(level="scenario", train_type="S", val_type="S", test_type="S",
                train=[1, 2, 3, 4, 7], val=[5], test=[6], cross_test=[6]),

    # IMPROVISED ONLY (block I)
    "R50": dict(level="scenario", train_type="I", val_type="I", test_type="I",
                train=[1, 2, 3, 4, 5], val=[6], test=[7], cross_test=[7]),
    "R51": dict(level="scenario", train_type="I", val_type="I", test_type="I",
                train=[2, 3, 4, 5, 6], val=[7], test=[1], cross_test=[1]),
    "R52": dict(level="scenario", train_type="I", val_type="I", test_type="I",
                train=[3, 4, 5, 6, 7], val=[1], test=[2], cross_test=[2]),
    "R53": dict(level="scenario", train_type="I", val_type="I", test_type="I",
                train=[1, 4, 5, 6, 7], val=[2], test=[3], cross_test=[3]),
    "R54": dict(level="scenario", train_type="I", val_type="I", test_type="I",
                train=[1, 2, 5, 6, 7], val=[3], test=[4], cross_test=[4]),
    "R55": dict(level="scenario", train_type="I", val_type="I", test_type="I",
                train=[1, 2, 3, 6, 7], val=[4], test=[5], cross_test=[5]),
    "R56": dict(level="scenario", train_type="I", val_type="I", test_type="I",
                train=[1, 2, 3, 4, 7], val=[5], test=[6], cross_test=[6]),

    # MIXED (S + I) – train/val on both; test_type "I/S" means:
    #   test       → improvised version of scenario N
    #   cross_test → scripted version of scenario N
    "R57": dict(level="scenario", train_type="S+I", val_type="S+I", test_type="I/S",
                train=[1, 2, 3, 4, 5], val=[6], test=[7], cross_test=[7]),
    "R58": dict(level="scenario", train_type="S+I", val_type="S+I", test_type="I/S",
                train=[2, 3, 4, 5, 6], val=[7], test=[1], cross_test=[1]),
    "R59": dict(level="scenario", train_type="S+I", val_type="S+I", test_type="I/S",
                train=[3, 4, 5, 6, 7], val=[1], test=[2], cross_test=[2]),
    "R60": dict(level="scenario", train_type="S+I", val_type="S+I", test_type="I/S",
                train=[1, 4, 5, 6, 7], val=[2], test=[3], cross_test=[3]),
    "R61": dict(level="scenario", train_type="S+I", val_type="S+I", test_type="I/S",
                train=[1, 2, 5, 6, 7], val=[3], test=[4], cross_test=[4]),
    "R62": dict(level="scenario", train_type="S+I", val_type="S+I", test_type="I/S",
                train=[1, 2, 3, 6, 7], val=[4], test=[5], cross_test=[5]),
    "R63": dict(level="scenario", train_type="S+I", val_type="S+I", test_type="I/S",
                train=[1, 2, 3, 4, 7], val=[5], test=[6], cross_test=[6]),
}

SCRIPTED_BLOCK = ["R43", "R44", "R45", "R46", "R47", "R48", "R49"]
IMPROV_BLOCK   = ["R50", "R51", "R52", "R53", "R54", "R55", "R56"]
MIXED_BLOCK    = ["R57", "R58", "R59", "R60", "R61", "R62", "R63"]


# ======================================================
# HELPERS: SCENARIO / STYLE / PATHS
# ======================================================

def extract_scenario_number(turn: str) -> Optional[int]:
    """
    Extract scenario IDs 1..7 from TURN strings like:
       Ses01F_script01_1_M019
       Ses01F_impro04_M029

    Script mapping:
        script01_1 → 1
        script01_2 → 2
        script01_3 → 3
        script02_1 → 4
        script02_2 → 5
        script03_1 → 6
        script03_2 → 7

    Improvised mapping:
        impro01 → 1
        impro02 → 2
        impro03 → 3
        impro04 → 4
        impro05 → 5
        impro06 → 6
        impro07 → 7
    """
    turn = str(turn)

    # improXX
    m = re.search(r"impro(\d{2})", turn)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 7:
            return n
        return None

    # scriptXX_Y
    m = re.search(r"script(\d{2})_(\d)", turn)
    if m:
        block = int(m.group(1))
        idx = int(m.group(2))

        if block == 1:       # script01_1,2,3 → 1,2,3
            return idx
        elif block == 2:     # script02_1,2   → 4,5
            return 3 + idx
        elif block == 3:     # script03_1,2   → 6,7
            return 5 + idx

    return None


def extract_session_number_from_turn(turn: str) -> Optional[int]:
    """
    TURN: e.g. 'Ses01F_impro01_M004' → session 1
    """
    m = re.search(r"Ses0*([0-9]+)", str(turn))
    if not m:
        return None
    return int(m.group(1))


def turn_to_video_path(turn: str) -> Optional[Path]:
    """
    Build the expected video path from TURN.

    Example:
        TURN: 'Ses01F_impro01_M004'
        Session: 1
        Conversation id: 'Ses01F_impro01'
        Path: VIDEO_ROOT / 'Session1' / 'Ses01F_impro01' / 'Ses01F_impro01_M004.mp4'
    """
    turn = str(turn)
    session_num = extract_session_number_from_turn(turn)
    if session_num is None:
        return None

    parts = turn.split("_")
    if len(parts) < 3:
        return None

    conv_id = "_".join(parts[:-1])  # Ses01F_impro01
    fname = f"{turn}.mp4"

    candidate = VIDEO_ROOT / f"Session{session_num}" / conv_id / fname
    return candidate


def filter_split_video(df: pd.DataFrame, which: str, cfg: dict,
                       opposite: bool = False) -> pd.DataFrame:
    """
    Filter df to get train/val/test for a given run_id config.

    Styles:
      S   : scripted
      I   : improvised
      S+I : both scripted and improvised
      I/S : only used for test_type;
            - normal test (opposite=False): improvised
            - cross-test (opposite=True): scripted
    """
    level = cfg["level"]
    ids = cfg[which]
    tcode = cfg[f"{which}_type"]

    if not opposite:
        if tcode == "S":
            df_sub = df[df["TURN"].str.contains("script", case=False)]
        elif tcode == "I":
            df_sub = df[df["TURN"].str.contains("impro", case=False)]
        elif tcode == "S+I":
            df_sub = df.copy()
        elif tcode == "I/S":
            df_sub = df[df["TURN"].str.contains("impro", case=False)]
        else:
            raise ValueError(f"Unknown type {tcode}")
    else:
        if tcode == "S":
            df_sub = df[df["TURN"].str.contains("impro", case=False)]
        elif tcode == "I":
            df_sub = df[df["TURN"].str.contains("script", case=False)]
        elif tcode == "S+I":
            raise ValueError("cross_test for S+I determined via test_type I/S.")
        elif tcode == "I/S":
            df_sub = df[df["TURN"].str.contains("script", case=False)]
        else:
            raise ValueError(f"Unknown type {tcode}")

    if level == "scenario":
        df_sub = df_sub[df_sub["scenario_number"].isin(ids)]
    else:
        df_sub = df_sub[df_sub["session_number"].isin(ids)]

    # keep only rows that actually have a video
    df_sub = df_sub[df_sub["video_path"].notnull()]
    return df_sub.copy()


# ======================================================
# DATASET
# ======================================================

class VideoUtteranceDataset(Dataset):
    def __init__(self, items, image_processor, num_frames, sample_rate, resize_to):
        """
        items: list of dicts {"path": str, "label": int}
        """
        self.items = items
        self.ip = image_processor
        self.num_frames = num_frames
        self.sample_rate = sample_rate
        self.resize = Resize(resize_to)

        self.normalize = Lambda(
            lambda x: (x - torch.tensor(self.ip.image_mean).view(3, 1, 1)) /
                      torch.tensor(self.ip.image_std).view(3, 1, 1)
        )

    def __len__(self):
        return len(self.items)

    def _read_frames(self, path: str) -> torch.Tensor:
        if HAVE_DECORD:
            vr = VideoReader(path)
            total = len(vr)
            raw_idx = np.linspace(0, total - 1,
                                  num=self.num_frames * self.sample_rate).astype(int)
            idx = raw_idx[::self.sample_rate][:self.num_frames]
            idx = np.clip(idx, 0, total - 1)
            frames = vr.get_batch(idx)
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        else:
            video, _, _ = read_video(path, pts_unit="sec")
            T = video.shape[0]
            raw_idx = np.linspace(0, T - 1,
                                  num=self.num_frames * self.sample_rate).astype(int)
            idx = raw_idx[::self.sample_rate][:self.num_frames]
            idx = np.clip(idx, 0, T - 1)
            frames = video[idx].permute(0, 3, 1, 2).float() / 255.0

        # resize and normalize per frame
        frames = torch.stack([self.resize(f) for f in frames])
        frames = torch.stack([self.normalize(f) for f in frames])
        return frames

    def __getitem__(self, i):
        rec = self.items[i]
        path = rec["path"]
        label = rec["label"]
        frames = self._read_frames(path)
        return {"video": frames, "labels": label}


def collate_fn(batch):
    vids = torch.stack([b["video"] for b in batch])
    labels = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
    return {"pixel_values": vids, "labels": labels}


# ======================================================
# METRICS & TRAINER
# ======================================================

accuracy_metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_metric.compute(predictions=preds, references=labels)["accuracy"],
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
    }


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
        else:
            loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ======================================================
# CORE: ONE VIDEO EXPERIMENT
# ======================================================

def run_video_experiment(
    run_id: str,
    cfg: dict,
    df_video: pd.DataFrame,
    hp: dict,
    hp_search: bool = False,
) -> Dict[str, float]:
    """
    One experiment for a given run_id:
      - Make train/val/test/cross splits
      - Build VideoMAE datasets
      - Train
      - Evaluate on val, test, cross
      - If hp_search=False: append TEST + CROSS rows to CSV.

    hp must contain:
      - learning_rate
      - num_train_epochs
      - batch_size
      - num_frames
      - sample_rate
    """
    print(f"\n=== VIDEO RUN {run_id} (hp_search={hp_search}) ===")
    print("HP:", hp)

    # --- splits ---
    train_df = filter_split_video(df_video, "train", cfg, opposite=False)
    val_df   = filter_split_video(df_video, "val",   cfg, opposite=False)
    test_df  = filter_split_video(df_video, "test",  cfg, opposite=False)
    cross_df = filter_split_video(df_video, "test",  cfg, opposite=True)

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0 or len(cross_df) == 0:
        print(f"SKIP EMPTY SPLIT {run_id}")
        return {
            "val_accuracy": np.nan,
            "val_f1_macro": np.nan,
            "test_accuracy": np.nan,
            "test_f1_macro": np.nan,
            "cross_accuracy": np.nan,
            "cross_f1_macro": np.nan,
        }

    # Label mapping
    labels_sorted = sorted(df_video[LABEL_COL].unique())
    label2id = {lbl: i for i, lbl in enumerate(labels_sorted)}
    id2label = {i: lbl for lbl, i in label2id.items()}

    # Items for datasets
    def mk_items(df_):
        return [
            {"path": str(p), "label": label2id[lab]}
            for p, lab in zip(df_["video_path"], df_[LABEL_COL])
        ]

    train_items = mk_items(train_df)
    val_items   = mk_items(val_df)
    test_items  = mk_items(test_df)
    cross_items = mk_items(cross_df)

    print(f"[{run_id}] n_train={len(train_items)}, n_val={len(val_items)}, "
          f"n_test={len(test_items)}, n_cross={len(cross_items)}")

    # Class weights from train
    counts = np.zeros(len(label2id), dtype=np.float32)
    for it in train_items:
        counts[it["label"]] += 1
    # avoid division by zero
    counts = np.where(counts == 0, 1.0, counts)
    class_weights = torch.tensor(1.0 / counts, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * len(label2id)

    # Processor + model
    ip = VideoMAEImageProcessor.from_pretrained(VMAE_BACKBONE)

    model = VideoMAEForVideoClassification.from_pretrained(
        VMAE_BACKBONE,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    ).to(device)

    # Resize dimensions
    if isinstance(ip.size, dict):
        if "shortest_edge" in ip.size:
            H = ip.size["shortest_edge"]
        elif "height" in ip.size:
            H = ip.size["height"]
        else:
            H = 224
    else:
        H = ip.size
    W = H
    resize_to = (H, W)

    # Datasets
    train_ds = VideoUtteranceDataset(train_items, ip, hp["num_frames"],
                                     hp["sample_rate"], resize_to)
    val_ds   = VideoUtteranceDataset(val_items,   ip, hp["num_frames"],
                                     hp["sample_rate"], resize_to)
    test_ds  = VideoUtteranceDataset(test_items,  ip, hp["num_frames"],
                                     hp["sample_rate"], resize_to)
    cross_ds = VideoUtteranceDataset(cross_items, ip, hp["num_frames"],
                                     hp["sample_rate"], resize_to)

    # Training args (checkpoints/logs go to scratch)
    out_dir = VIDEO_MODEL_BASE / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    args_hf = TrainingArguments(
        output_dir=str(out_dir),
        learning_rate=hp["learning_rate"],
        per_device_train_batch_size=hp["batch_size"],
        per_device_eval_batch_size=hp["batch_size"],
        num_train_epochs=hp["num_train_epochs"],
        eval_strategy="epoch",         # compatible alias
        save_strategy="no",            # don't save checkpoints to save disk
        warmup_ratio=0.1,
        load_best_model_at_end=False,  # since we don't save checkpoints
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        report_to=["none"],
        logging_strategy="epoch",
    )

    trainer = WeightedTrainer(
        model=model,
        args=args_hf,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    try:
        trainer.train()
        val_metrics   = trainer.evaluate(val_ds)
        test_metrics  = trainer.evaluate(test_ds)
        cross_metrics = trainer.evaluate(cross_ds)
    except SystemExit as e:
        print(f"🚨 SYSTEM EXIT during run {run_id} with hp={hp}")
        print("Likely silent HF/torchvision failure.")
        raise RuntimeError("Hard failure instead of silent exit") from e
    except Exception as e:
        print(f"🔥 CRASH during run {run_id} with hp={hp}")
        raise


    metrics = {
        "val_accuracy":   float(val_metrics["eval_accuracy"]),
        "val_f1_macro":   float(val_metrics["eval_f1_macro"]),
        "test_accuracy":  float(test_metrics["eval_accuracy"]),
        "test_f1_macro":  float(test_metrics["eval_f1_macro"]),
        "cross_accuracy": float(cross_metrics["eval_accuracy"]),
        "cross_f1_macro": float(cross_metrics["eval_f1_macro"]),
    }

    print(f"[{run_id}] Metrics:", metrics)

    if not hp_search:
        # Two rows: TEST + CROSS
        with open(ALL_RESULTS_PATH, "a", newline="") as f:
            w = csv.writer(f)

            # SAME-STYLE TEST
            w.writerow([
                run_id,
                "videomae-large",
                "TEST",
                hp["learning_rate"],
                hp["num_train_epochs"],
                hp["batch_size"],
                hp["num_frames"],
                hp["sample_rate"],
                metrics["val_accuracy"],
                metrics["val_f1_macro"],
                metrics["test_accuracy"],
                metrics["test_f1_macro"],
            ])

            # CROSS-STYLE TEST
            w.writerow([
                run_id,
                "videomae-large",
                "CROSS",
                hp["learning_rate"],
                hp["num_train_epochs"],
                hp["batch_size"],
                hp["num_frames"],
                hp["sample_rate"],
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

    for lr in VIDEO_GRID["learning_rate"]:
        for epochs in VIDEO_GRID["num_train_epochs"]:
            for bs in VIDEO_GRID["batch_size"]:
                for nf in VIDEO_GRID["num_frames"]:
                    for sr in VIDEO_GRID["sample_rate"]:
                        hp = dict(
                            learning_rate=lr,
                            num_train_epochs=epochs,
                            batch_size=bs,
                            num_frames=nf,
                            sample_rate=sr,
                        )
                        print(
                            f"Try hp: lr={lr}, epochs={epochs}, bs={bs}, "
                            f"num_frames={nf}, sample_rate={sr}"
                        )

                        metrics = run_video_experiment(
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

    # Load CSV and prep dataframe with video paths / scenario / session
    df = pd.read_csv(LABEL_CSV)
    if LABEL_COL not in df.columns:
        raise ValueError(
            f"Expected column '{LABEL_COL}' in {LABEL_CSV}. "
            f"Please update LABEL_COL at top of script."
        )

    needed_cols = ["TURN", LABEL_COL]
    for c in needed_cols:
        if c not in df.columns:
            raise ValueError(f"CSV missing column '{c}'")

    df = df.dropna(subset=["TURN", LABEL_COL]).copy()

    df["scenario_number"] = df["TURN"].apply(extract_scenario_number)
    df["session_number"] = df["TURN"].apply(extract_session_number_from_turn)

    # Attach video paths
    video_paths = []
    exists_flags = []
    for t in df["TURN"]:
        p = turn_to_video_path(t)
        if p is not None and p.exists():
            video_paths.append(str(p))
            exists_flags.append(True)
        else:
            video_paths.append(None)
            exists_flags.append(False)

    df["video_path"] = video_paths
    df_video = df[df["video_path"].notnull()].copy()

    print(f"Total rows in CSV: {len(df)}")
    print(f"Rows with existing video files: {len(df_video)}")

    # Optional sanity: print labels actually present
    actual_labels = sorted(df_video[LABEL_COL].unique())
    print("✅ Labels in video subset:", actual_labels)

    # Initialize results CSV (overwrite each new run)
    with open(ALL_RESULTS_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id",
            "model",
            "eval_type",        # TEST or CROSS
            "learning_rate",
            "num_epochs",
            "batch_size",
            "num_frames",
            "sample_rate",
            "val_accuracy",
            "val_f1_macro",
            "accuracy",
            "f1_macro",
        ])

    print("✅ Video experiment pipeline ready.")
    print("✅ Result CSV:", ALL_RESULTS_PATH)
    print("✅ Grid size per block:", len(VIDEO_GRID["learning_rate"])
          * len(VIDEO_GRID["num_train_epochs"])
          * len(VIDEO_GRID["batch_size"])
          * len(VIDEO_GRID["num_frames"])
          * len(VIDEO_GRID["sample_rate"]))

    # -------------------------------
    # Block 1: SCRIPTED (R43–R49), grid on R43
    # -------------------------------
    print("\n====== SCRIPTED BLOCK (R43–R49) ======")
    best_S = grid_search_for_block("R43", SPLITS_VIDEO["R43"], df_video)

    for run_id in SCRIPTED_BLOCK:
        cfg = SPLITS_VIDEO[run_id]
        print(f"\n=== FINAL VIDEO RUN (SCRIPTED) {run_id} ===")
        run_video_experiment(run_id, cfg, df_video, best_S, hp_search=False)

    # -------------------------------
    # Block 2: IMPROV (R50–R56), grid on R50
    # -------------------------------
    print("\n====== IMPROV BLOCK (R50–R56) ======")
    best_I = grid_search_for_block("R50", SPLITS_VIDEO["R50"], df_video)

    for run_id in IMPROV_BLOCK:
        cfg = SPLITS_VIDEO[run_id]
        print(f"\n=== FINAL VIDEO RUN (IMPROV) {run_id} ===")
        run_video_experiment(run_id, cfg, df_video, best_I, hp_search=False)

    # -------------------------------
    # Block 3: MIXED (R57–R63), grid on R57
    # -------------------------------
    print("\n====== MIXED BLOCK (R57–R63) ======")
    best_SI = grid_search_for_block("R57", SPLITS_VIDEO["R57"], df_video)

    for run_id in MIXED_BLOCK:
        cfg = SPLITS_VIDEO[run_id]
        print(f"\n=== FINAL VIDEO RUN (MIXED) {run_id} ===")
        run_video_experiment(run_id, cfg, df_video, best_SI, hp_search=False)

    print("\nDONE — All VIDEO runs completed.")


if __name__ == "__main__":
    main()
