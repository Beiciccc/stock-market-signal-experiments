from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


COMPETITION = "stock-market-signal-predict-next-day-returns"
ID_COL = "id"
TARGET_COL = "target"
STOCK_COL = "stock_id"


def rank01(values: np.ndarray) -> np.ndarray:
    s = pd.Series(np.asarray(values, dtype=np.float64))
    return s.rank(method="average").to_numpy() / (len(s) + 1.0)


def zscore(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    sd = x.std()
    if sd == 0 or not np.isfinite(sd):
        return np.zeros_like(x)
    return (x - x.mean()) / sd


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40, 40)
    return 1.0 / (1.0 + np.exp(-x))


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    return train, test, sample


def numeric_features(train: pd.DataFrame) -> list[str]:
    return [c for c in train.columns if c not in {ID_COL, TARGET_COL, STOCK_COL}]


def encode_stock(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()
    all_stock = pd.concat([train[STOCK_COL], test[STOCK_COL]], axis=0).astype("category")
    codes = all_stock.cat.codes.to_numpy()
    train[STOCK_COL] = codes[: len(train)].astype(np.int16)
    test[STOCK_COL] = codes[len(train) :].astype(np.int16)
    return train, test


def write_submission(sample: pd.DataFrame, pred: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub = sample.copy()
    sub[TARGET_COL] = np.clip(np.asarray(pred, dtype=np.float64), 0.0, 1.0)
    sub.to_csv(out_path, index=False)


def save_metrics(metrics: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")


def time_holdout_auc(train: pd.DataFrame, pred_col: np.ndarray, holdout_frac: float = 0.05) -> float:
    cut = train[ID_COL].quantile(1.0 - holdout_frac)
    mask = train[ID_COL] > cut
    return float(roc_auc_score(train.loc[mask, TARGET_COL], np.asarray(pred_col)[mask]))


def contrarian_signal(df: pd.DataFrame) -> np.ndarray:
    """Low recent momentum/overextension has been the most stable late-train signal."""
    specs = [
        ("sma_ratio_5", -2.0),
        ("return_5d", -1.8),
        ("roc_5", -1.8),
        ("sma_ratio_10", -1.5),
        ("ema_ratio_12", -1.4),
        ("return_1d", -1.2),
        ("ema_ratio_26", -1.1),
        ("sma_ratio_20", -0.8),
        ("volatility_60d", -0.8),
    ]
    score = np.zeros(len(df), dtype=np.float64)
    for col, weight in specs:
        if col in df:
            score += weight * zscore(df[col].to_numpy())
    return rank01(score)


def experiment_contrarian(data_dir: Path, out_dir: Path, artifact_dir: Path) -> dict:
    train, test, sample = load_data(data_dir)
    train_pred = contrarian_signal(train)
    test_pred = contrarian_signal(test)
    metrics = {
        "experiment": "contrarian_signal",
        "time_holdout_auc_5pct": time_holdout_auc(train, train_pred, 0.05),
        "time_holdout_auc_10pct": time_holdout_auc(train, train_pred, 0.10),
        "all_train_auc": float(roc_auc_score(train[TARGET_COL], train_pred)),
        "submission": str(out_dir / "contrarian_signal.csv"),
    }
    write_submission(sample, test_pred, out_dir / "contrarian_signal.csv")
    save_metrics(metrics, artifact_dir / "contrarian_signal_metrics.json")
    return metrics


def public_submission_files(public_dir: Path) -> list[Path]:
    files = []
    for p in sorted(public_dir.glob("*/*.csv")):
        try:
            cols = pd.read_csv(p, nrows=2).columns.tolist()
        except Exception:
            continue
        if TARGET_COL in cols:
            files.append(p)
    return files


def load_public_predictions(public_dir: Path, sample: pd.DataFrame) -> dict[str, np.ndarray]:
    preds: dict[str, np.ndarray] = {}
    sample_ids = sample[ID_COL].to_numpy()
    for p in public_submission_files(public_dir):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if TARGET_COL not in df or len(df) != len(sample):
            continue
        if ID_COL in df and not np.array_equal(df[ID_COL].to_numpy(), sample_ids):
            df = sample[[ID_COL]].merge(df[[ID_COL, TARGET_COL]], on=ID_COL, how="left")
            if df[TARGET_COL].isna().any():
                continue
        name = p.parent.name + "__" + p.stem
        preds[name] = df[TARGET_COL].astype(float).to_numpy()
    return preds


PUBLIC_WEIGHTS = {
    "emanuellcs_stock_market_signal_predict_next_day_returns__submission": 1.15,
    "emanuellcs_stock_market_signal_predict_next_day_returns__submission_blend": 1.15,
    "emanuellcs_stock_market_signal_predict_next_day_returns__submission_lgbm": 1.05,
    "emanuellcs_stock_market_signal_predict_next_day_returns__submission_meta": 1.10,
    "emanuellcs_stock_market_signal_predict_next_day_returns__submission_xgb": 1.00,
    "lightningv08_stock_market_signal_blind_blend__submission": 1.20,
    "lightningv08_stock_market_signal_single_tabicl_model__submission": 0.95,
    "alanamonks_hgb_baseline_next_day_stock_pred__submission": 1.10,
    "ravi20076_stock2026_blindblend_v1__submission": 1.05,
    "ravi20076_stock2026_public_baseline_v2__submission": 0.95,
    "junaid512_stock_market_signal_predictday__submission": 0.85,
    "hmnshudhmn24_stock_market_signal_predict_next_day_returns__submission": 0.75,
    "suhanigupta04_stock_direction_prediction_next_day_returns__submission": 0.65,
    "nina2025_stock_market_signal_h_blend__submission": 0.60,
    "nina2025_stock_market_signal_hs_feature_engineering__submission": 0.55,
}


def weighted_public_blend(preds: dict[str, np.ndarray], use_all: bool = False) -> tuple[np.ndarray, dict[str, float]]:
    arrays = []
    weights = []
    used = {}
    for name, pred in preds.items():
        w = PUBLIC_WEIGHTS.get(name, 0.25 if use_all else 0.0)
        if w <= 0:
            continue
        arrays.append(rank01(pred) - 0.5)
        weights.append(w)
        used[name] = w
    if not arrays:
        raise ValueError("No public predictions matched the configured blend weights.")
    mat = np.column_stack(arrays)
    w = np.asarray(weights, dtype=np.float64)
    raw = mat @ w / w.sum()
    return rank01(raw), used


def experiment_public_blends(data_dir: Path, public_dir: Path, out_dir: Path, artifact_dir: Path) -> dict:
    _, test, sample = load_data(data_dir)
    preds = load_public_predictions(public_dir, sample)
    public_pred, used = weighted_public_blend(preds, use_all=False)
    contrarian = contrarian_signal(test)
    hybrid = rank01(0.82 * (public_pred - 0.5) + 0.18 * (contrarian - 0.5))
    write_submission(sample, public_pred, out_dir / "public_rank_blend_v1.csv")
    write_submission(sample, hybrid, out_dir / "public_rank_blend_plus_contrarian_v1.csv")

    stats = {
        name: {
            "min": float(np.min(pred)),
            "max": float(np.max(pred)),
            "mean": float(np.mean(pred)),
            "std": float(np.std(pred)),
        }
        for name, pred in preds.items()
    }
    corr_names = list(preds)
    if corr_names:
        corr_mat = np.column_stack([rank01(preds[n]) for n in corr_names])
        corr = pd.DataFrame(corr_mat, columns=corr_names).corr("spearman")
        corr.to_csv(artifact_dir / "public_prediction_spearman.csv")
    metrics = {
        "experiment": "public_blends",
        "n_public_predictions": len(preds),
        "used_weights": used,
        "prediction_stats": stats,
        "submissions": [
            str(out_dir / "public_rank_blend_v1.csv"),
            str(out_dir / "public_rank_blend_plus_contrarian_v1.csv"),
        ],
    }
    save_metrics(metrics, artifact_dir / "public_blends_metrics.json")
    return metrics


def experiment_narrow_blends(data_dir: Path, public_dir: Path, out_dir: Path, artifact_dir: Path) -> dict:
    _, test, sample = load_data(data_dir)
    preds = load_public_predictions(public_dir, sample)
    contrarian = contrarian_signal(test)
    base = {
        "em": preds["emanuellcs_stock_market_signal_predict_next_day_returns__submission"],
        "light": preds["lightningv08_stock_market_signal_blind_blend__submission"],
        "alan": preds["alanamonks_hgb_baseline_next_day_stock_pred__submission"],
        "ravi": preds["ravi20076_stock2026_blindblend_v1__submission"],
        "junaid": preds["junaid512_stock_market_signal_predictday__submission"],
        "contrarian": contrarian,
    }
    candidates = {
        "emanuellcs70_contrarian30_rank": {"em": 0.70, "contrarian": 0.30},
        "narrow_em_contr_light_alan_rank": {"em": 0.37, "contrarian": 0.19, "light": 0.18, "alan": 0.26},
        "narrow_em_contr_light_alan_junaid_rank": {
            "em": 0.34,
            "contrarian": 0.18,
            "light": 0.19,
            "alan": 0.24,
            "junaid": 0.05,
        },
        "narrow_em_contr_light_alan_ravi_rank": {
            "em": 0.36,
            "contrarian": 0.18,
            "light": 0.15,
            "alan": 0.22,
            "ravi": 0.09,
        },
    }
    outputs = []
    for name, weights in candidates.items():
        score = np.zeros(len(sample), dtype=np.float64)
        denom = sum(weights.values())
        for key, weight in weights.items():
            score += weight * (rank01(base[key]) - 0.5) / denom
        pred = rank01(score)
        path = out_dir / f"{name}.csv"
        write_submission(sample, pred, path)
        outputs.append(str(path))

    metrics = {"experiment": "narrow_blends", "submissions": outputs, "weights": candidates}
    save_metrics(metrics, artifact_dir / "narrow_blends_metrics.json")
    return metrics


def experiment_hgb(data_dir: Path, out_dir: Path, artifact_dir: Path) -> dict:
    train, test, sample = load_data(data_dir)
    train_enc, test_enc = encode_stock(train, test)
    features = [c for c in train_enc.columns if c not in {ID_COL, TARGET_COL}]
    x = train_enc[features].to_numpy()
    y = train_enc[TARGET_COL].to_numpy()
    x_test = test_enc[features].to_numpy()

    params = {
        "loss": "log_loss",
        "max_depth": 12,
        "learning_rate": 0.02,
        "max_iter": 500,
        "max_leaf_nodes": 53,
        "min_samples_leaf": 349,
        "l2_regularization": 0.05,
        "random_state": 42,
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(train_enc), dtype=np.float64)
    pred = np.zeros(len(test_enc), dtype=np.float64)
    folds = []
    for fold, (trn_idx, val_idx) in enumerate(skf.split(x, y), 1):
        model = HistGradientBoostingClassifier(**params)
        model.fit(x[trn_idx], y[trn_idx])
        val_pred = model.predict_proba(x[val_idx])[:, 1]
        oof[val_idx] = val_pred
        pred += model.predict_proba(x_test)[:, 1] / skf.n_splits
        folds.append(float(roc_auc_score(y[val_idx], val_pred)))
    metrics = {
        "experiment": "hgb_public_repro",
        "random_oof_auc": float(roc_auc_score(y, oof)),
        "fold_auc": folds,
        "submission": str(out_dir / "hgb_public_repro.csv"),
    }
    write_submission(sample, pred, out_dir / "hgb_public_repro.csv")
    save_metrics(metrics, artifact_dir / "hgb_public_repro_metrics.json")
    return metrics


def experiment_simple_logit(data_dir: Path, out_dir: Path, artifact_dir: Path) -> dict:
    """A low-variance calibrated version of the late-time contrarian signal."""
    train, test, sample = load_data(data_dir)
    train_signal = contrarian_signal(train) - 0.5
    test_signal = contrarian_signal(test) - 0.5
    # Use a small slope to avoid overconfident probabilities. AUC depends only on order.
    pred = sigmoid(1.2 * test_signal)
    metrics = {
        "experiment": "simple_logit_contrarian",
        "time_holdout_auc_5pct": time_holdout_auc(train, train_signal, 0.05),
        "time_holdout_auc_10pct": time_holdout_auc(train, train_signal, 0.10),
        "submission": str(out_dir / "simple_logit_contrarian.csv"),
    }
    write_submission(sample, pred, out_dir / "simple_logit_contrarian.csv")
    save_metrics(metrics, artifact_dir / "simple_logit_contrarian_metrics.json")
    return metrics


def run(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    public_dir = Path(args.public_dir)
    out_dir = Path(args.out_dir)
    artifact_dir = Path(args.artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    experiments = {
        "contrarian": lambda: experiment_contrarian(data_dir, out_dir, artifact_dir),
        "simple_logit": lambda: experiment_simple_logit(data_dir, out_dir, artifact_dir),
        "public_blends": lambda: experiment_public_blends(data_dir, public_dir, out_dir, artifact_dir),
        "narrow_blends": lambda: experiment_narrow_blends(data_dir, public_dir, out_dir, artifact_dir),
        "hgb": lambda: experiment_hgb(data_dir, out_dir, artifact_dir),
    }
    selected: Iterable[str] = experiments.keys() if args.experiment == "all" else [args.experiment]
    for name in selected:
        metrics = experiments[name]()
        print(json.dumps(metrics, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=["all", "contrarian", "simple_logit", "public_blends", "narrow_blends", "hgb"],
        default="all",
    )
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--public-dir", default="data/public_outputs")
    parser.add_argument("--out-dir", default="submissions")
    parser.add_argument("--artifact-dir", default="artifacts")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
