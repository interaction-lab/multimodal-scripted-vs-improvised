#!/usr/bin/env python3
"""
Run multimodal DAC experiments for R01–R21.

"""

import re
import time
import argparse
from pathlib import Path
from functools import lru_cache
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, f1_score


SPLITS: Dict[str, Dict[str, Any]] = {
    # SCRIPTED ONLY
    "R01": dict(level="scenario", train_type="S", val_type="S", test_type="S",
                train=[1, 2, 3, 4, 5], val=[6], test=[7], cross_test=[7]),
    "R02": dict(level="scenario", train_type="S", val_type="S", test_type="S",
                train=[2, 3, 4, 5, 6], val=[7], test=[1], cross_test=[1]),
    "R03": dict(level="scenario", train_type="S", val_type="S", test_type="S",
                train=[3, 4, 5, 6, 7], val=[1], test=[2], cross_test=[2]),
    "R04": dict(level="scenario", train_type="S", val_type="S", test_type="S",
                train=[1, 4, 5, 6, 7], val=[2], test=[3], cross_test=[3]),
    "R05": dict(level="scenario", train_type="S", val_type="S", test_type="S",
                train=[1, 2, 5, 6, 7], val=[3], test=[4], cross_test=[4]),
    "R06": dict(level="scenario", train_type="S", val_type="S", test_type="S",
                train=[1, 2, 3, 6, 7], val=[4], test=[5], cross_test=[5]),
    "R07": dict(level="scenario", train_type="S", val_type="S", test_type="S",
                train=[1, 2, 3, 4, 7], val=[5], test=[6], cross_test=[6]),

    # IMPROVISED ONLY
    "R08": dict(level="scenario", train_type="I", val_type="I", test_type="I",
                train=[1, 2, 3, 4, 5], val=[6], test=[7], cross_test=[7]),
    "R09": dict(level="scenario", train_type="I", val_type="I", test_type="I",
                train=[2, 3, 4, 5, 6], val=[7], test=[1], cross_test=[1]),
    "R10": dict(level="scenario", train_type="I", val_type="I", test_type="I",
                train=[3, 4, 5, 6, 7], val=[1], test=[2], cross_test=[2]),
    "R11": dict(level="scenario", train_type="I", val_type="I", test_type="I",
                train=[1, 4, 5, 6, 7], val=[2], test=[3], cross_test=[3]),
    "R12": dict(level="scenario", train_type="I", val_type="I", test_type="I",
                train=[1, 2, 5, 6, 7], val=[3], test=[4], cross_test=[4]),
    "R13": dict(level="scenario", train_type="I", val_type="I", test_type="I",
                train=[1, 2, 3, 6, 7], val=[4], test=[5], cross_test=[5]),
    "R14": dict(level="scenario", train_type="I", val_type="I", test_type="I",
                train=[1, 2, 3, 4, 7], val=[5], test=[6], cross_test=[6]),

    # MIXED (S + I)
    # test_type "I/S" means:
    #   test      → improvised version of scenario N
    #   cross_test → scripted version of scenario N
    "R15": dict(level="scenario", train_type="S+I", val_type="S+I", test_type="I/S",
                train=[1, 2, 3, 4, 5], val=[6], test=[7], cross_test=[7]),
    "R16": dict(level="scenario", train_type="S+I", val_type="S+I", test_type="I/S",
                train=[2, 3, 4, 5, 6], val=[7], test=[1], cross_test=[1]),
    "R17": dict(level="scenario", train_type="S+I", val_type="S+I", test_type="I/S",
                train=[3, 4, 5, 6, 7], val=[1], test=[2], cross_test=[2]),
    "R18": dict(level="scenario", train_type="S+I", val_type="S+I", test_type="I/S",
                train=[1, 4, 5, 6, 7], val=[2], test=[3], cross_test=[3]),
    "R19": dict(level="scenario", train_type="S+I", val_type="S+I", test_type="I/S",
                train=[1, 2, 5, 6, 7], val=[3], test=[4], cross_test=[4]),
    "R20": dict(level="scenario", train_type="S+I", val_type="S+I", test_type="I/S",
                train=[1, 2, 3, 6, 7], val=[4], test=[5], cross_test=[5]),
    "R21": dict(level="scenario", train_type="S+I", val_type="S+I", test_type="I/S",
                train=[1, 2, 3, 4, 7], val=[5], test=[6], cross_test=[6]),
}

SCRIPTED_BLOCK = ["R01", "R02", "R03", "R04", "R05", "R06", "R07"]
IMPROV_BLOCK   = ["R08", "R09", "R10", "R11", "R12", "R13", "R14"]
MIXED_BLOCK    = ["R15", "R16", "R17", "R18", "R19", "R20", "R21"]

GRID_RUNS = {"R01", "R08", "R15"}


# ---------------------------
# Turn parsing (style/scenario)
# ---------------------------

# Matches TURNs like:
#   Ses01F_impro01_F000
#   Ses01F_script02_F123
_IMPRO_RE = re.compile(r"_impro(\d+)_", re.IGNORECASE)
_SCRIPT_RE = re.compile(r"_script(\d+)_", re.IGNORECASE)

def parse_style_and_scenario(turn: str):
    m = _IMPRO_RE.search(turn)
    if m:
        return ("I", int(m.group(1)))
    m = _SCRIPT_RE.search(turn)
    if m:
        return ("S", int(m.group(1)))
    return (None, None)


# ---------------------------
# Device selection (avoid P100 sm_60 issue)
# ---------------------------

def pick_device(force_cpu: bool = False):
    if force_cpu:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    cap = torch.cuda.get_device_capability(0)  # (major, minor)
    if cap[0] < 7:
        return torch.device("cpu")
    return torch.device("cuda")


# ---------------------------
# Loading embeddings
# ---------------------------

@lru_cache(maxsize=512)
def _load_pt_cached(pt_path: str):
    return torch.load(pt_path, map_location="cpu")

def load_vector_pt(path: str) -> torch.Tensor:
    obj = _load_pt_cached(path)
    if isinstance(obj, torch.Tensor):
        x = obj
    elif isinstance(obj, dict):
        for k in ["x", "emb", "embedding", "feat", "features"]:
            if k in obj and isinstance(obj[k], torch.Tensor):
                x = obj[k]
                break
        else:
            raise ValueError(f"Don't know how to read tensor from dict in {path}. Keys={list(obj.keys())}")
    else:
        raise ValueError(f"Unexpected .pt type at {path}: {type(obj)}")

    x = x.detach().cpu()
    if x.ndim == 2 and x.shape[0] == 1:
        x = x.squeeze(0)
    return x.float()

def load_hubert_item(hubert_pt_path: str, item_idx: int) -> torch.Tensor:
    obj = _load_pt_cached(hubert_pt_path)
    if isinstance(obj, torch.Tensor):
        X = obj
    elif isinstance(obj, dict):
        if "x" in obj and isinstance(obj["x"], torch.Tensor):
            X = obj["x"]
        else:
            X = None
            for v in obj.values():
                if isinstance(v, torch.Tensor) and v.ndim == 2:
                    X = v
                    break
            if X is None:
                raise ValueError(f"Couldn't find (N,D) tensor in {hubert_pt_path}. Keys={list(obj.keys())}")
    else:
        raise ValueError(f"Unexpected HuBERT .pt type: {type(obj)} at {hubert_pt_path}")

    X = X.detach().cpu()
    return X[int(item_idx)].float()


# ---------------------------
# Dataset
# ---------------------------

class MultiModalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label2id: dict):
        self.df = df.reset_index(drop=True)
        self.label2id = label2id

        # Guard against empty splits (better error message upstream, but safe here too)
        if len(self.df) == 0:
            self.d_rob = self.d_vid = self.d_moc = self.d_hub = None
            return

        r0 = self.df.iloc[0]
        self.d_rob = int(load_vector_pt(r0["roberta_path"]).numel())
        self.d_vid = int(load_vector_pt(r0["videomae_path"]).numel())
        self.d_moc = int(load_vector_pt(r0["mocap_path"]).numel())
        self.d_hub = int(load_hubert_item(r0["hubert_pt_path"], int(r0["hubert_item_idx"])).numel())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        rob = load_vector_pt(r["roberta_path"])
        vid = load_vector_pt(r["videomae_path"])
        moc = load_vector_pt(r["mocap_path"])
        hub = load_hubert_item(r["hubert_pt_path"], int(r["hubert_item_idx"]))
        y = self.label2id[r["DA"]]
        return rob, hub, vid, moc, y

def collate_fn(batch):
    rob, hub, vid, moc, y = zip(*batch)
    return (
        torch.stack(rob, dim=0),
        torch.stack(hub, dim=0),
        torch.stack(vid, dim=0),
        torch.stack(moc, dim=0),
        torch.tensor(y, dtype=torch.long),
    )


# ---------------------------
# Model
# ---------------------------

class FusionMLP(nn.Module):
    def __init__(self, d_rob, d_hub, d_vid, d_moc, proj_dim=256, hidden=512, dropout=0.2, n_classes=12):
        super().__init__()

        def proj(in_dim):
            return nn.Sequential(
                nn.Linear(in_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        self.p_rob = proj(d_rob)
        self.p_hub = proj(d_hub)
        self.p_vid = proj(d_vid)
        self.p_moc = proj(d_moc)

        fusion_dim = 4 * proj_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, rob, hub, vid, moc):
        z = torch.cat([self.p_rob(rob), self.p_hub(hub), self.p_vid(vid), self.p_moc(moc)], dim=1)
        return self.head(z)


# ---------------------------
# Train / Eval
# ---------------------------

@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    ys, ps = [], []
    for rob, hub, vid, moc, y in loader:
        rob = rob.to(device)
        hub = hub.to(device)
        vid = vid.to(device)
        moc = moc.to(device)
        logits = model(rob, hub, vid, moc)
        pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
        ps.append(pred)
        ys.append(y.numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

def train_one_setting(
    train_df, val_df,
    label2id,
    device,
    proj_dim, hidden, dropout,
    lr, weight_decay,
    batch_size,
    epochs,
    seed,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError(f"Empty train/val split (train={len(train_df)}, val={len(val_df)})")

    train_ds = MultiModalDataset(train_df, label2id)
    val_ds   = MultiModalDataset(val_df, label2id)

    model = FusionMLP(
        train_ds.d_rob, train_ds.d_hub, train_ds.d_vid, train_ds.d_moc,
        proj_dim=proj_dim, hidden=hidden, dropout=dropout, n_classes=len(label2id)
    ).to(device)

    use_cuda = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=use_cuda, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=use_cuda, collate_fn=collate_fn)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best = {"epoch": -1, "val_f1_macro": -1.0, "state": None, "val_metrics": None}

    for ep in range(1, epochs + 1):
        model.train()
        for rob, hub, vid, moc, y in train_loader:
            rob = rob.to(device)
            hub = hub.to(device)
            vid = vid.to(device)
            moc = moc.to(device)
            y   = y.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(rob, hub, vid, moc)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        vm = eval_model(model, val_loader, device)
        if vm["f1_macro"] > best["val_f1_macro"]:
            best["epoch"] = ep
            best["val_f1_macro"] = vm["f1_macro"]
            best["val_metrics"] = vm
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best["state"])
    return model, best


# ---------------------------
# Split builder (uses SPLITS exactly)
# ---------------------------

def filter_by_style(df: pd.DataFrame, train_type: str) -> pd.DataFrame:
    if train_type == "S":
        return df[df["style"] == "S"]
    if train_type == "I":
        return df[df["style"] == "I"]
    if train_type == "S+I":
        return df[df["style"].isin(["S", "I"])]
    raise ValueError(f"Unknown type: {train_type}")

def make_split_df(df_all: pd.DataFrame, run_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spec = SPLITS[run_name]

    # training/val pool style restriction
    train_pool = filter_by_style(df_all, spec["train_type"])
    val_pool   = filter_by_style(df_all, spec["val_type"])

    # scenario membership
    train_df = train_pool[train_pool["scenario"].isin(spec["train"])].copy()
    val_df   = val_pool[val_pool["scenario"].isin(spec["val"])].copy()

    # test / cross rules
    if spec["test_type"] == "S":
        test_df  = df_all[(df_all["style"] == "S") & (df_all["scenario"].isin(spec["test"]))].copy()
        cross_df = df_all[(df_all["style"] == "I") & (df_all["scenario"].isin(spec["cross_test"]))].copy()
    elif spec["test_type"] == "I":
        test_df  = df_all[(df_all["style"] == "I") & (df_all["scenario"].isin(spec["test"]))].copy()
        cross_df = df_all[(df_all["style"] == "S") & (df_all["scenario"].isin(spec["cross_test"]))].copy()
    elif spec["test_type"] == "I/S":
        # per your comment: test = impro, cross_test = scripted (same scenario ids)
        test_df  = df_all[(df_all["style"] == "I") & (df_all["scenario"].isin(spec["test"]))].copy()
        cross_df = df_all[(df_all["style"] == "S") & (df_all["scenario"].isin(spec["cross_test"]))].copy()
    else:
        raise ValueError(f"Unknown test_type: {spec['test_type']}")

    return train_df, val_df, test_df, cross_df


# ---------------------------
# Grid iterator
# ---------------------------

def iter_grid(grid: Dict[str, List[Any]]):
    # Avoid numpy meshgrid object coercion issues; do explicit product
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]

    def _rec(i, cur):
        if i == len(keys):
            yield dict(cur)
            return
        k = keys[i]
        for v in vals[i]:
            cur[k] = v
            yield from _rec(i + 1, cur)
        cur.pop(k, None)

    yield from _rec(0, {})


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all_embed_csv", type=str, required=True)
    ap.add_argument("--out_csv", type=str, default="results_R01_R21.csv")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force_cpu", action="store_true")
    args = ap.parse_args()

    device = pick_device(force_cpu=args.force_cpu)
    print(f"Device: {device}")

    df = pd.read_csv(args.all_embed_csv)

    # enrich with style/scenario
    df["TURN"] = df["TURN"].astype(str)
    styles, scens = [], []
    bad = 0
    for t in df["TURN"].tolist():
        s, c = parse_style_and_scenario(t)
        if s is None or c is None:
            bad += 1
        styles.append(s)
        scens.append(c)
    df["style"] = styles
    df["scenario"] = scens

    if bad:
        print(f"WARNING: {bad} TURN rows couldn't parse style/scenario")
        # show a few examples to make debugging easy
        bad_rows = df[df["style"].isna() | df["scenario"].isna()][["TURN"]].head(10)
        print("Example unparsed TURNs:\n", bad_rows.to_string(index=False))

    need_cols = ["DA","roberta_path","videomae_path","mocap_path","hubert_pt_path","hubert_item_idx","style","scenario"]
    before = len(df)
    df = df.dropna(subset=need_cols).copy()
    after = len(df)

    print(f"Rows before dropna: {before} | after: {after}")
    print("Style counts:\n", df["style"].value_counts(dropna=False))
    print("Scenario counts:\n", df["scenario"].value_counts(dropna=False).sort_index())

    labels = sorted(df["DA"].unique().tolist())
    label2id = {lab: i for i, lab in enumerate(labels)}
    print(f"Rows: {len(df)} | Labels: {len(labels)}")

    # Grid ONLY for R01/R08/R15
    grid = {
        "proj_dim": [128, 256],
        "hidden": [256, 512],
        "dropout": [0.2, 0.4],
        "lr": [1e-3, 3e-4],
        "weight_decay": [1e-4, 1e-2],
    }

    # store best HP per block from the grid runs
    best_hp_by_run: Dict[str, Dict[str, Any]] = {}
    best_hp_by_block: Dict[str, Dict[str, Any]] = {}
    rows_out: List[Dict[str, Any]] = []

    def run_eval_once(run_name: str, hp: Dict[str, Any], tag: str):
        t0 = time.time()
        train_df, val_df, test_df, cross_df = make_split_df(df, run_name)

        # ---- guardrails: fail fast with useful debug output ----
        def _check_nonempty(name: str, sdf: pd.DataFrame):
            if len(sdf) == 0:
                print(f"\n[ERROR] {run_name}: {name} split is EMPTY")
                print("Split sizes:",
                      "train", len(train_df),
                      "val", len(val_df),
                      "test", len(test_df),
                      "cross", len(cross_df))
                print("Spec:", SPLITS[run_name])
                print("df style counts:\n", df["style"].value_counts(dropna=False))
                print("df scenario counts:\n", df["scenario"].value_counts(dropna=False).sort_index())
                print("Example TURNs:\n", df["TURN"].head(10).tolist())
                raise ValueError(f"{run_name}: {name} split empty — check parse_style_and_scenario() and make_split_df()")

        _check_nonempty("train", train_df)
        _check_nonempty("val", val_df)
        _check_nonempty("test", test_df)
        _check_nonempty("cross", cross_df)

        # dataloaders for test/cross
        use_cuda = (device.type == "cuda")
        test_loader = DataLoader(MultiModalDataset(test_df, label2id), batch_size=args.batch_size, shuffle=False,
                                 num_workers=0, pin_memory=use_cuda, collate_fn=collate_fn)
        cross_loader = DataLoader(MultiModalDataset(cross_df, label2id), batch_size=args.batch_size, shuffle=False,
                                  num_workers=0, pin_memory=use_cuda, collate_fn=collate_fn)

        model, best = train_one_setting(
            train_df=train_df, val_df=val_df,
            label2id=label2id, device=device,
            proj_dim=int(hp["proj_dim"]),
            hidden=int(hp["hidden"]),
            dropout=float(hp["dropout"]),
            lr=float(hp["lr"]),
            weight_decay=float(hp["weight_decay"]),
            batch_size=args.batch_size,
            epochs=args.epochs,
            seed=args.seed,
        )

        test_metrics = eval_model(model, test_loader, device)
        cross_metrics = eval_model(model, cross_loader, device)

        row = {
            "run": run_name,
            "tag": tag,  # grid / best_from_block / etc.
            "proj_dim": int(hp["proj_dim"]),
            "hidden": int(hp["hidden"]),
            "dropout": float(hp["dropout"]),
            "lr": float(hp["lr"]),
            "weight_decay": float(hp["weight_decay"]),
            "best_epoch": int(best["epoch"]),
            "val_acc": float(best["val_metrics"]["acc"]),
            "val_f1_macro": float(best["val_metrics"]["f1_macro"]),
            "val_f1_weighted": float(best["val_metrics"]["f1_weighted"]),
            "test_acc": float(test_metrics["acc"]),
            "test_f1_macro": float(test_metrics["f1_macro"]),
            "test_f1_weighted": float(test_metrics["f1_weighted"]),
            "cross_acc": float(cross_metrics["acc"]),
            "cross_f1_macro": float(cross_metrics["f1_macro"]),
            "cross_f1_weighted": float(cross_metrics["f1_weighted"]),
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(len(test_df)),
            "n_cross": int(len(cross_df)),
            "seconds": round(time.time() - t0, 2),
        }
        return row

    # 1) grid runs (R01, R08, R15)
    for run in ["R01", "R08", "R15"]:
        print(f"\n=== GRID: {run} ===")
        best_row = None

        for gi, hp in enumerate(iter_grid(grid), start=1):
            row = run_eval_once(run, hp, tag=f"grid_{gi:03d}")
            rows_out.append(row)

            print(f"[{run}] {gi:03d} valF1={row['val_f1_macro']:.4f} testF1={row['test_f1_macro']:.4f} crossF1={row['cross_f1_macro']:.4f}")

            if (best_row is None) or (row["val_f1_macro"] > best_row["val_f1_macro"]):
                best_row = row
                best_hp_by_run[run] = dict(hp)

        print(f"Best HP for {run}: {best_hp_by_run[run]} (valF1={best_row['val_f1_macro']:.4f})")

    # map best HP to blocks
    best_hp_by_block["SCRIPTED"] = best_hp_by_run["R01"]
    best_hp_by_block["IMPROV"]   = best_hp_by_run["R08"]
    best_hp_by_block["MIXED"]    = best_hp_by_run["R15"]

    # 2) run remaining experiments once using block best
    def block_of(run_name: str) -> str:
        if run_name in SCRIPTED_BLOCK:
            return "SCRIPTED"
        if run_name in IMPROV_BLOCK:
            return "IMPROV"
        if run_name in MIXED_BLOCK:
            return "MIXED"
        raise ValueError(run_name)

    for run in SPLITS.keys():
        if run in GRID_RUNS:
            continue
        blk = block_of(run)
        hp = best_hp_by_block[blk]
        print(f"\n=== {run} (single) using {blk} best HP ===")
        row = run_eval_once(run, hp, tag=f"best_{blk.lower()}")
        rows_out.append(row)
        print(f"[{run}] valF1={row['val_f1_macro']:.4f} testF1={row['test_f1_macro']:.4f} crossF1={row['cross_f1_macro']:.4f}")

    out = pd.DataFrame(rows_out)
    out.to_csv(args.out_csv, index=False)
    print(f"\nSaved -> {args.out_csv}")

    # Print compact summary: best-per-run among rows (grid rows included)
    bests = out.sort_values(["run", "val_f1_macro"], ascending=[True, False]).groupby("run", as_index=False).head(1)
    print("\n=== Best row per run (by val_f1_macro) ===")
    print(bests[[
        "run","tag","proj_dim","hidden","dropout","lr","weight_decay",
        "val_f1_macro","test_f1_macro","cross_f1_macro"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()