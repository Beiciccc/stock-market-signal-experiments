from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ID_COL = "id"
TARGET_COL = "target"


def rank01(values: np.ndarray) -> np.ndarray:
    s = pd.Series(np.asarray(values, dtype=np.float64))
    return s.rank(method="average").to_numpy() / (len(s) + 1.0)


def load_pred(path: Path, sample: pd.DataFrame) -> np.ndarray:
    df = pd.read_csv(path)
    if ID_COL in df.columns and not np.array_equal(df[ID_COL].to_numpy(), sample[ID_COL].to_numpy()):
        df = sample[[ID_COL]].merge(df[[ID_COL, TARGET_COL]], on=ID_COL, how="left")
        if df[TARGET_COL].isna().any():
            raise ValueError(f"Could not align ids for {path}")
    return df[TARGET_COL].astype(float).to_numpy()


def write_submission(sample: pd.DataFrame, pred: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub = sample.copy()
    sub[TARGET_COL] = np.clip(np.asarray(pred, dtype=np.float64), 0.0, 1.0)
    sub.to_csv(out_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="data/raw/sample_submission.csv")
    parser.add_argument("--base", default="submissions/em675_contr324_rank.csv")
    parser.add_argument("--tabicl", required=True)
    parser.add_argument("--out-dir", default="submissions")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--prefix", default="best_tabicl_v2")
    parser.add_argument("--weights", default="0.03,0.05,0.08,0.10,0.15")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample = pd.read_csv(args.sample)
    base_path = Path(args.base)
    tabicl_path = Path(args.tabicl)
    out_dir = Path(args.out_dir)
    artifact_dir = Path(args.artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    base = rank01(load_pred(base_path, sample)) - 0.5
    tabicl = rank01(load_pred(tabicl_path, sample)) - 0.5
    corr = float(pd.Series(base).corr(pd.Series(tabicl), method="spearman"))

    weights = [float(x) for x in args.weights.split(",") if x.strip()]
    outputs = {}
    for w in weights:
        name = f"{args.prefix}_w{int(round(w * 1000)):03d}.csv"
        pred = rank01((1.0 - w) * base + w * tabicl)
        path = out_dir / name
        write_submission(sample, pred, path)
        outputs[name] = {
            "tabicl_weight": w,
            "submission": str(path),
            "spearman_to_base": corr,
        }

    metrics = {
        "base": str(base_path),
        "tabicl": str(tabicl_path),
        "spearman_to_base": corr,
        "outputs": outputs,
    }
    (artifact_dir / f"{args.prefix}_blend_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
