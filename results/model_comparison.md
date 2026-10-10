# Model comparison (eval, 1,200 questions)

### Table 1: Accuracy, before the gate

| model | all exact | real-world exact | company exact | hard exact | all needed |
|---|---|---|---|---|---|
| tfidf_logreg_v3 | 55.4% | 68.8% | 69.3% | 40.0% | 65.1% |
| bge_logreg_v1 | 45.8% | 67.0% | 30.0% | 30.7% | 60.8% |
| tfidf_bge_logreg_v1 | 58.2% | 76.0% | 68.6% | 39.6% | 69.5% |

### Table 2: After the gate

| model | gate | coverage | right on kept | real-world coverage | real-world right on kept | company coverage | company right on kept | hard-case coverage | hard-case kept |
|---|---|---|---|---|---|---|---|---|---|
| tfidf_logreg_v3 | 0.11 | 61.0% | 69.5% | 67.4% | 82.5% | 69.3% | 82.5% | 53.2% | 50.7% |
| bge_logreg_v1 | 0.21 | 41.8% | 71.5% | 58.0% | 86.9% | 17.1% | 58.3% | 33.4% | 49.2% |
| tfidf_bge_logreg_v1 | 0.15 | 63.8% | 71.8% | 74.6% | 87.9% | 59.3% | 81.9% | 55.4% | 49.7% |

### Table 3: Speed and size

| model | p50 ms | p95 ms | size MB |
|---|---|---|---|
| tfidf_logreg_v3 | 0.6 | 0.6 | 4.67 |
| bge_logreg_v1 | 5.7 | 6.6 | 0.01 |
| tfidf_bge_logreg_v1 | 6.5 | 7.4 | 4.69 |

### Table 4: Exact accuracy on all 1,200 questions, any category

| model | before the gate (all exact) | after the gate (right on kept) |
|---|---|---|
| tfidf_logreg_v3 | 55.4% | 69.5% |
| bge_logreg_v1 | 45.8% | 71.5% |
| tfidf_bge_logreg_v1 | 58.2% | 71.8% |

### Table 5: Out of 100 questions (without the gate vs with the gate)

| model | questions | without gate: right | without gate: wrong | with gate: model right | with gate: model wrong | with gate: sent to LLM |
|---|---|---|---|---|---|---|
| tfidf_logreg_v3 | all | 55 | 45 | 42 | 19 | 39 |
| tfidf_logreg_v3 | real-world | 69 | 31 | 56 | 12 | 33 |
| tfidf_logreg_v3 | company | 69 | 31 | 57 | 12 | 31 |
| tfidf_logreg_v3 | hard cases | 40 | 60 | 27 | 26 | 47 |
| bge_logreg_v1 | all | 46 | 54 | 30 | 12 | 58 |
| bge_logreg_v1 | real-world | 67 | 33 | 50 | 8 | 42 |
| bge_logreg_v1 | company | 30 | 70 | 10 | 7 | 83 |
| bge_logreg_v1 | hard cases | 31 | 69 | 16 | 17 | 67 |
| tfidf_bge_logreg_v1 | all | 58 | 42 | 46 | 18 | 36 |
| tfidf_bge_logreg_v1 | real-world | 76 | 24 | 66 | 9 | 25 |
| tfidf_bge_logreg_v1 | company | 69 | 31 | 49 | 11 | 41 |
| tfidf_bge_logreg_v1 | hard cases | 40 | 60 | 28 | 28 | 45 |

