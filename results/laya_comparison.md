# Why not Laya? Laya (zero-shot and fine-tuned) vs our models (eval, 1,200 questions)

Laya = a ready-made decision model (ModernBERT-large, 421M), asked 7 yes/no questions (one per
source). Fine-tuned = trained on the fit slice (the same data model B learned from). All of
Laya's thresholds and top-1 cuts were chosen on the threshold slice (train), never on eval.

| system | all exact | real-world | company | hard | all needed |
|---|---|---|---|---|---|
| Laya zero-shot (thresholds) | 7.4% | 13.8% | 2.9% | 2.9% | 72.3% |
| Laya zero-shot (top-1) | 39.1% | 43.8% | 52.1% | 31.6% | 45.4% |
| Laya FINE-TUNED laya_ft_v1, 1 epoch(s) (thresholds) | 57.9% | 72.6% | 69.3% | 42.0% | 64.9% |
| Laya FINE-TUNED laya_ft_v1, 1 epoch(s) (top-1) | 60.3% | 77.6% | 75.7% | 41.1% | 63.6% |
| Laya FINE-TUNED laya_ft_v2, 3 epoch(s) (thresholds) | 65.3% | 82.0% | 76.4% | 47.7% | 71.0% |
| Laya FINE-TUNED laya_ft_v2, 3 epoch(s) (top-1) | 66.2% | 85.8% | 85.0% | 43.9% | 68.3% |
| model B (tfidf_ftv2_bge_logreg_v2) | 63.7% | 84.0% | 84.3% | 40.4% | 68.2% |
| gpt-4o prompt v4 | 74.4% | 74.6% | 85.7% | 71.4% | 79.0% |
| FULL SYSTEM (model B + gpt-4o v4, Step 5.6) | 70.0% | 84.2% | 90.7% | 52.1% | 73.2% |

Fine-tuned versions (model B: ~7 ms; target < 50 ms; Laya cost $0):
- laya_ft_v1: 1 epoch(s) │ threshold-slice top-1 exact 68.8% (used to CHOOSE between versions) │ speed p50 46 ms, p95 56 ms
- laya_ft_v2: 3 epoch(s) │ threshold-slice top-1 exact 77.2% (used to CHOOSE between versions) │ speed p50 46 ms, p95 54 ms
