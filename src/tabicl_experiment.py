from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


ID_COL = "id"
TARGET_COL = "target"
STOCK_COL = "stock_id"


def write_submission(sample: pd.DataFrame, pred: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub = sample.copy()
    sub[TARGET_COL] = np.clip(np.asarray(pred, dtype=np.float64), 0.0, 1.0)
    sub.to_csv(out_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--out-dir", default="submissions")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--name", default="tabicl_v2_est16")
    parser.add_argument("--n-estimators", type=int, default=16)
    parser.add_argument("--checkpoint-version", default="tabicl-classifier-v2-20260212.ckpt")
    parser.add_argument("--predict-batch-size", type=int, default=8192)
    parser.add_argument("--random-state", type=int, default=777)
    parser.add_argument("--include-stock-id", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()

    try:
        from tabicl import TabICLClassifier
    except ImportError as exc:
        raise SystemExit(
            "tabicl is not installed. Install it with: python -m pip install tabicl"
        ) from exc

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    artifact_dir = Path(args.artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    print(
        f"loaded train={train.shape} test={test.shape} checkpoint={args.checkpoint_version} "
        f"n_estimators={args.n_estimators}",
        flush=True,
    )

    drop_cols = [TARGET_COL]
    if not args.include_stock_id:
        drop_cols.append(STOCK_COL)
    x = train.drop(columns=drop_cols)
    y = train[TARGET_COL]
    x_test = test.drop(columns=[] if args.include_stock_id else [STOCK_COL])
    x_test = x_test[x.columns]

    model = TabICLClassifier(
        n_estimators=args.n_estimators,
        norm_methods=["quantile"],
        feat_shuffle_method="latin",
        class_shuffle_method="shift",
        outlier_threshold=4.0,
        softmax_temperature=0.9,
        average_logits=True,
        support_many_classes=True,
        batch_size=1,
        model_path=None,
        allow_auto_download=True,
        checkpoint_version=args.checkpoint_version,
        device=None,
        use_amp="auto",
        use_fa3="auto",
        offload_mode="auto",
        disk_offload_dir=None,
        random_state=args.random_state,
        n_jobs=None,
        verbose=args.verbose,
        inference_config=None,
    )
    fit_started = time.time()
    print("fitting TabICL support set", flush=True)
    model.fit(x, y)
    fit_seconds = time.time() - fit_started
    print(f"fit complete in {fit_seconds:.1f}s", flush=True)

    chunks: list[np.ndarray] = []
    predict_started = time.time()
    for start in range(0, len(x_test), args.predict_batch_size):
        stop = min(start + args.predict_batch_size, len(x_test))
        print(f"predicting rows {start}:{stop}", flush=True)
        pred = model.predict_proba(x_test.iloc[start:stop])[:, 1]
        chunks.append(pred)
        print(f"predicted rows {start}:{stop}", flush=True)
    test_pred = np.concatenate(chunks)
    predict_seconds = time.time() - predict_started

    out_path = out_dir / f"{args.name}.csv"
    write_submission(sample, test_pred, out_path)

    metrics = {
        "experiment": args.name,
        "checkpoint_version": args.checkpoint_version,
        "n_estimators": args.n_estimators,
        "include_stock_id": args.include_stock_id,
        "predict_batch_size": args.predict_batch_size,
        "rows_train": int(len(train)),
        "rows_test": int(len(test)),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "total_seconds": time.time() - started,
        "prediction_stats": {
            "min": float(np.min(test_pred)),
            "max": float(np.max(test_pred)),
            "mean": float(np.mean(test_pred)),
            "std": float(np.std(test_pred)),
        },
        "submission": str(out_path),
    }
    (artifact_dir / f"{args.name}_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
