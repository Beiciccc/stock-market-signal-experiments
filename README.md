# Stock Market Signal Experiments

Public experiment archive for the Kaggle competition
[Stock Market Signal: Predict Next-Day Returns](https://www.kaggle.com/competitions/stock-market-signal-predict-next-day-returns).

This repository contains:

- Experiment notes and public leaderboard readouts.
- Reproducible modeling and blending scripts.
- Generated submission files.
- Metrics and diagnostic artifacts produced during iteration.

Raw competition data is not redistributed here. Download it from Kaggle:

```bash
kaggle competitions download -c stock-market-signal-predict-next-day-returns
```

## Current Best Public Score

Best public leaderboard score recorded in this archive: `0.52028`.

The strongest submitted family is a TabICL v2 single-estimator signal blended at
small rank weights over the prior best anchor. The useful range plateaued around
`3%` to `12%`; higher weights and public-output microblends did not improve.

## Repository Layout

- `src/` - modeling and blending scripts.
- `reports/experiment_log.md` - chronological experiment record.
- `artifacts/` - metrics, correlations, and blend metadata.
- `submissions/` - generated submission CSVs.

## Notes

The leaderboard results in `reports/experiment_log.md` are public leaderboard
scores observed at submission time. They should not be treated as validation
metrics for model selection beyond the public leaderboard context.
