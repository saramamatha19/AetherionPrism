# Full system: Laya vs model B as the small model (eval, 1,200 questions)

Both: answer when sure (gate chosen on the gate slice, 80% rule), gpt-4o prompt v4 otherwise.

| full system | all exact | real-world | company | hard | all needed | small model answers | hard: confident mistakes / 100 | cost / 1M |
|---|---|---|---|---|---|---|---|---|
| Laya laya_ft_v2 + gpt-4o (no rule), gate 0.26 | 69.0% | 82.6% | 81.4% | 53.8% | 73.4% | 88% | 40 | $203 |
| Laya laya_ft_v2 + gpt-4o (+ negation rule), gate 0.3 | 70.8% | 82.2% | 85.0% | 57.1% | 75.3% | 77% | 33 | $404 |
| model B + gpt-4o (+ negation rule), gate 0.13 | 70.0% | 84.2% | 90.7% | 52.1% | 73.2% | 76% | 37 | $414 |
