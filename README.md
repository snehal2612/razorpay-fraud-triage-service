# Razorpay Merchant Risk Desk — Fraud Triage Service

## Overview

This project implements an offline fraud triage service for synthetic Razorpay merchant transactions.

- Cleans transaction data and engineers fraud-related features.
- Trains and evaluates a fraud classification model.
- Retrieves relevant policy documents using TF-IDF similarity.
- Generates and validates structured case notes using a mock LLM.
- Exposes scoring, case triage, health, and metrics endpoints through FastAPI.
- Applies guardrails to prevent unsafe automated approvals.
- Includes smoke tests and structured request logging.

## Reproduce

Run all commands from the project root using Python 3.11 or later.

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Generate the synthetic dataset

```powershell
python generate_data.py
```

Keep `generate_data.py` and `mock_llm.py` unchanged.

### 4. Run Part A: Data preparation

```powershell
python prepare.py
```

### 5. Run Part B: Model training and evaluation

```powershell
python train.py
```

### 6. Run Part C: Retrieval and case-note checks

```powershell
python retrieval.py
python casenote.py
```

### 7. Start the API

```powershell
uvicorn app:app --host 127.0.0.1 --port 8000
```

Keep the server running. Open a second PowerShell terminal in the project root.

### 8. Run smoke tests

```powershell
python smoke_test.py
```

The expected result is `OVERALL: 11/11 PASS`.

Useful API endpoints:

- `GET /health` — service health check
- `POST /score` — score a transaction
- `POST /case` — perform fraud triage and generate a case note
- `GET /metrics` — request and latency metrics

## Part A results

### 1. Raw data audit

| Audit item | Result |
|---|---:|
| Raw rows | 6,090 |
| Raw columns | 14 |
| Exact duplicate rows | 90 |
| Missing `amount_inr` after deduplication | 76 |
| Missing `distance_from_home_km` after deduplication | 193 |
| Out-of-range amounts after deduplication | 5 |
| Distinct raw `merchant_category` spellings | 24 |
| Distinct raw `city` spellings | 24 |

### 2. Cleaning results

| Checkpoint | Result |
|---|---:|
| Rows after cleaning | 5,919 |
| Duplicate rows remaining | 0 |
| Missing distances remaining | 191 |
| Minimum amount (INR) | 141.37 |
| Maximum amount (INR) | 105,883.95 |

Missing `distance_from_home_km` values are left unfilled in Part A because the imputation median must be learned from training rows only to avoid validation and test data leakage.

### 3. Merchant category, city and payment-mode counts

**Merchant category counts**

| Merchant category | Rows |
|---|---:|
| Grocery | 1,406 |
| Food Delivery | 1,326 |
| Fashion | 976 |
| Electronics | 845 |
| Utilities | 685 |
| Travel | 681 |

**City counts**

| City | Rows |
|---|---:|
| Bengaluru | 1,033 |
| Chennai | 1,025 |
| Kolkata | 996 |
| Delhi | 979 |
| Pune | 944 |
| Mumbai | 942 |

**Payment-mode counts**

| Payment mode | Rows |
|---|---:|
| UPI | 3,221 |
| Card | 1,450 |
| NetBanking | 731 |
| Wallet | 517 |

### 4. Feature engineering checkpoints

The `add_features(df)` function adds these three features:

- `amount_vs_avg_ratio` = `amount_inr / avg_amount_30d_inr`
- `is_night` = 1 when `hour_of_day` is 5 or less; otherwise 0
- `high_value` = 1 when `amount_inr` is greater than 50,000; otherwise 0

| Feature checkpoint | Result |
|---|---:|
| Number of features added | 3 |
| Sum of `is_night` | 365 |
| Sum of `high_value` | 17 |
| Maximum `amount_vs_avg_ratio` | 13.11 |

Both `clean_transactions()` and `add_features()` passed the one-row and input-immutability checks.

### 5. Fraud rate and why accuracy misleads

| Metric | Result |
|---|---:|
| Fraud rows | 366 |
| Cleaned rows | 5,919 |
| Fraud rate | 6.18% |
| Never-fraud rule accuracy | 93.82% |
| Never-fraud rule recall | 0.00% |

The never-fraud rule achieves 93.82% accuracy because most transactions are legitimate, but it misses every fraud case. Accuracy alone is therefore misleading for this imbalanced fraud-detection problem; recall and F1 also need to be considered.

### 6. Chargeback leakage audit

| `chargeback_filed` | Total rows | Fraud rows |
|---:|---:|---:|
| 0 | 5,546 | 62 |
| 1 | 373 | 304 |

The model excludes `chargeback_filed`, `txn_id`, `customer_id` and `is_fraud` from its inputs. `chargeback_filed` can reflect a dispute raised after settlement, so using it would leak information unavailable at prediction time. The transaction and customer identifiers are identifiers rather than legitimate behavioural predictors, and `is_fraud` is the target being predicted.

### 7. Model input columns before one-hot encoding

The 13 columns passed into the feature-encoding stage are:

```text
merchant_category
city
payment_mode
amount_inr
hour_of_day
txn_count_last_24h
avg_amount_30d_inr
distance_from_home_km
is_new_device
account_age_days
amount_vs_avg_ratio
is_night
high_value
```

After one-hot encoding the three categorical columns, the model input contains 26 columns.

## Part B results

### Model training and validation

- Training rows: **3,551**
- Validation rows: **1,184**
- Test rows: **1,184**
- Fraud cases in training, validation, and test sets: **220, 73, and 73**, respectively.
- Training-set distance median: **12.6**.
- Oversampled training rows: **6,662**. Oversampling was applied to training data only.
- The final feature matrix contains **26 features**.
- Balanced Logistic Regression cross-validation F1: **0.402 ± 0.015**.
- Untuned Random Forest cross-validation F1: **0.453 ± 0.052**.
- Grid search evaluated **12 combinations** and selected `max_depth=None` and `min_samples_leaf=5`.
- Best mean cross-validation F1: **0.465**.
- Selected decision threshold: **0.40**.

### Class imbalance comparison

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| Plain Logistic Regression | 0.625 | 0.342 | 0.442 |
| Balanced Logistic Regression | 0.252 | 0.781 | 0.381 |
| Oversampled Logistic Regression | 0.257 | 0.781 | 0.386 |

### Validation threshold analysis

| Threshold | Precision | Recall | F1 | Flagged |
|---:|---:|---:|---:|---:|
| 0.1 | 0.111 | 0.973 | 0.200 | 638 |
| 0.2 | 0.158 | 0.945 | 0.271 | 437 |
| 0.3 | 0.209 | 0.863 | 0.336 | 302 |
| 0.4 | 0.272 | 0.712 | 0.394 | 191 |
| 0.5 | 0.352 | 0.589 | 0.441 | 122 |
| 0.6 | 0.507 | 0.507 | 0.507 | 73 |
| 0.7 | 0.528 | 0.260 | 0.349 | 36 |
| 0.8 | 0.778 | 0.192 | 0.308 | 18 |
| 0.9 | 0.800 | 0.055 | 0.103 | 5 |

### Final test evaluation

- Test precision: **0.281**.
- Test recall: **0.781**.
- Test F1 score: **0.413**.
- Confusion matrix: **TN=965, FP=146, FN=16, TP=57**.
- Test transactions flagged at the selected threshold: **203**.
- Legitimate payments flagged per fraud caught: **2.561**.
- Frauds missed: **16**.

At threshold 0.40, the model caught 57 fraud cases but also flagged 146 legitimate payments as false positives. This reflects a trade-off between detecting fraud and avoiding unnecessary transaction reviews.

### Leakage experiment

- Clean tuned-forest validation F1: **0.441**.
- Leaky-forest validation F1: **0.797**.

The higher leaky validation score illustrates how including information unavailable at prediction time can make evaluation misleading. The target label, identifiers, and `chargeback_filed` are excluded from model inputs.

### Top five feature importances

| Feature | Importance |
|---|---:|
| `amount_vs_avg_ratio` | 0.269996 |
| `txn_count_last_24h` | 0.177916 |
| `amount_inr` | 0.133949 |
| `account_age_days` | 0.083521 |
| `distance_from_home_km` | 0.070884 |

### Reusable model builder

The `build_final_model()` function returns a dictionary containing `model`, `columns`, `median`, and `threshold`. The verified output contains **26 columns**, a training distance median of **12.6**, and a threshold of **0.4**.

### Limitations

The selected threshold prioritizes fraud recall but also flags legitimate transactions. The validation and test metrics reflect this dataset and split and do not guarantee equivalent real-world performance. The leakage experiment demonstrates why post-outcome information must not be used as a prediction feature.

## Part C results

### Retrieval

- Policy retrieval checks passed: **8/8**
- Configured minimum similarity score: **0.45**
- Highest out-of-scope similarity observed: **0.4251**
- Lowest in-scope similarity observed: **0.4862**

The observed scores are separated by the configured retrieval threshold for the tested examples.

### Case-note generation and validation

- Total mock LLM calls: **19**
- Successful first-call validations: **3**
- Retry recoveries: **6**
- Safe defaults: **1**
- Guardrail overrides: **1**
- Simulated outage: `ConnectionError` propagated as expected

Case notes are checked against the required schema and allowed enum values. Invalid responses can be retried, while unsafe or unavailable outcomes are handled without silently treating them as successful approvals.

## Part D results

### API smoke tests

**Overall result: 11/11 PASS**

| Test group | Passed | Total |
|---|---:|---:|
| Happy path | 3 | 3 |
| Escalation | 4 | 4 |
| Edge cases | 4 | 4 |
| **Overall** | **11** | **11** |

### Metrics from the clean smoke-test run

| Metric | Result |
|---|---:|
| Total requests | 10 |
| Successful requests | 7 |
| Error requests | 3 |
| Success rate | 0.70 |
| Error rate | 0.30 |
| Human-review total | 5 |
| Human-review rate | 0.50 |

Average latency for this run was approximately **53.48 ms**. Latency can vary between runs and machines.

### Structured logs

The final `service.log` contains the expected 10 POST request log lines:

- INFO: **5**
- WARNING: **4**
- ERROR: **1**

The log was checked after a clean server restart and one smoke-test run.

## Limitations

1. **Synthetic data:** Results are based on generated data and may not reflect real merchant fraud patterns.
2. **Limited generalization:** Model performance on this dataset does not guarantee performance on real-world transactions.
3. **Mock LLM:** Case-note generation uses a deterministic offline mock rather than a production language model.
4. **Lexical retrieval:** TF-IDF retrieval measures lexical similarity and may miss semantically relevant policy text.
5. **Threshold uncertainty:** Threshold tuning uses a limited validation set, including only 73 fraud cases.
6. **Operational hardening:** Authentication, rate limiting, persistent storage, and production-grade monitoring are outside this capstone implementation.
7. **Latency variability:** The reported latency reflects one local smoke-test run and is not a production service-level guarantee.
