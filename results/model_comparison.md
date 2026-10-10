# Model comparison (eval, 1,200 questions)

### Table 1: Accuracy, before the gate

| model | all exact | real-world exact | company exact | hard exact | all needed |
|---|---|---|---|---|---|
| tfidf_logreg_v3 | 55.4% | 68.8% | 69.3% | 40.0% | 65.1% |
| bge_logreg_v1 | 45.8% | 67.0% | 30.0% | 30.7% | 60.8% |
| tfidf_bge_logreg_v1 | 58.2% | 76.0% | 68.6% | 39.6% | 69.5% |
| ft_bge_logreg_v1 | 56.2% | 77.8% | 70.0% | 33.6% | 65.3% |
| tfidf_ft_bge_logreg_v1 | 60.4% | 81.2% | 82.9% | 36.2% | 69.4% |
| tfidf_ftv2_bge_logreg_v1 | 63.7% | 84.0% | 84.3% | 40.4% | 68.2% |
| tfidf_ftsplit_bge_logreg_v1 | 57.7% | 75.8% | 72.1% | 37.9% | 68.4% |
| tfidf_ftv2_bge_logreg_v2 | 63.7% | 84.0% | 84.3% | 40.4% | 68.2% |

### Table 2: After the gate

| model | negation rule | gate | coverage | right on kept | real-world coverage | real-world right on kept | company coverage | company right on kept | hard-case coverage | hard-case kept |
|---|---|---|---|---|---|---|---|---|---|---|
| tfidf_logreg_v3 | off | 0.11 | 61.0% | 69.5% | 67.4% | 82.5% | 69.3% | 82.5% | 53.2% | 50.7% |
| bge_logreg_v1 | off | 0.21 | 41.8% | 71.5% | 58.0% | 86.9% | 17.1% | 58.3% | 33.4% | 49.2% |
| tfidf_bge_logreg_v1 | off | 0.15 | 63.8% | 71.8% | 74.6% | 87.9% | 59.3% | 81.9% | 55.4% | 49.7% |
| ft_bge_logreg_v1 | off | 0.15 | 75.2% | 67.4% | 83.0% | 88.4% | 70.7% | 84.8% | 69.3% | 40.5% |
| tfidf_ft_bge_logreg_v1 | off | 0.09 | 86.7% | 65.8% | 90.6% | 87.0% | 88.6% | 87.1% | 82.7% | 39.3% |
| tfidf_ftv2_bge_logreg_v1 | off | 0.11 | 85.8% | 69.2% | 92.0% | 88.0% | 87.9% | 91.1% | 79.8% | 43.8% |
| tfidf_ftsplit_bge_logreg_v1 | off | 0.16 | 68.6% | 71.0% | 78.4% | 87.8% | 70.7% | 85.9% | 59.3% | 46.7% |
| tfidf_ftv2_bge_logreg_v2 | on | 0.13 | 76.2% | 71.3% | 84.0% | 89.3% | 79.3% | 91.9% | 68.6% | 45.6% |

### Table 3: Speed and size

| model | p50 ms | p95 ms | size MB |
|---|---|---|---|
| tfidf_logreg_v3 | 0.6 | 0.6 | 4.67 |
| bge_logreg_v1 | 5.7 | 6.3 | 0.01 |
| tfidf_bge_logreg_v1 | 6.8 | 7.4 | 4.69 |
| ft_bge_logreg_v1 | 5.7 | 5.9 | 0.01 |
| tfidf_ft_bge_logreg_v1 | 6.3 | 7.3 | 4.69 |
| tfidf_ftv2_bge_logreg_v1 | 7.0 | 7.4 | 4.69 |
| tfidf_ftsplit_bge_logreg_v1 | 7.0 | 7.4 | 3.09 |
| tfidf_ftv2_bge_logreg_v2 | 7.0 | 7.4 | 4.69 |

### Table 4: Exact accuracy on all 1,200 questions, any category

| model | before the gate (all exact) | after the gate (right on kept) |
|---|---|---|
| tfidf_logreg_v3 | 55.4% | 69.5% |
| bge_logreg_v1 | 45.8% | 71.5% |
| tfidf_bge_logreg_v1 | 58.2% | 71.8% |
| ft_bge_logreg_v1 | 56.2% | 67.4% |
| tfidf_ft_bge_logreg_v1 | 60.4% | 65.8% |
| tfidf_ftv2_bge_logreg_v1 | 63.7% | 69.2% |
| tfidf_ftsplit_bge_logreg_v1 | 57.7% | 71.0% |
| tfidf_ftv2_bge_logreg_v2 | 63.7% | 71.3% |

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
| ft_bge_logreg_v1 | all | 56 | 44 | 51 | 24 | 25 |
| ft_bge_logreg_v1 | real-world | 78 | 22 | 73 | 10 | 17 |
| ft_bge_logreg_v1 | company | 70 | 30 | 60 | 11 | 29 |
| ft_bge_logreg_v1 | hard cases | 34 | 66 | 28 | 41 | 31 |
| tfidf_ft_bge_logreg_v1 | all | 60 | 40 | 57 | 30 | 13 |
| tfidf_ft_bge_logreg_v1 | real-world | 81 | 19 | 79 | 12 | 9 |
| tfidf_ft_bge_logreg_v1 | company | 83 | 17 | 77 | 11 | 11 |
| tfidf_ft_bge_logreg_v1 | hard cases | 36 | 64 | 32 | 50 | 17 |
| tfidf_ftv2_bge_logreg_v1 | all | 64 | 36 | 59 | 26 | 14 |
| tfidf_ftv2_bge_logreg_v1 | real-world | 84 | 16 | 81 | 11 | 8 |
| tfidf_ftv2_bge_logreg_v1 | company | 84 | 16 | 80 | 8 | 12 |
| tfidf_ftv2_bge_logreg_v1 | hard cases | 40 | 60 | 35 | 45 | 20 |
| tfidf_ftsplit_bge_logreg_v1 | all | 58 | 42 | 49 | 20 | 31 |
| tfidf_ftsplit_bge_logreg_v1 | real-world | 76 | 24 | 69 | 10 | 22 |
| tfidf_ftsplit_bge_logreg_v1 | company | 72 | 28 | 61 | 10 | 29 |
| tfidf_ftsplit_bge_logreg_v1 | hard cases | 38 | 62 | 28 | 32 | 41 |
| tfidf_ftv2_bge_logreg_v2 | all | 64 | 36 | 54 | 22 | 24 |
| tfidf_ftv2_bge_logreg_v2 | real-world | 84 | 16 | 75 | 9 | 16 |
| tfidf_ftv2_bge_logreg_v2 | company | 84 | 16 | 73 | 6 | 21 |
| tfidf_ftv2_bge_logreg_v2 | hard cases | 40 | 60 | 31 | 37 | 31 |

