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

### Data audit and cleaning

- Raw dataset size: **6,090 rows × 14 columns**
- Duplicate rows removed: **90**
- Rows after deduplication: **6,000**
- Rows with missing amount removed: **76**
- Rows with missing distance removed: **193**
- Out-of-range rows removed: **5**
- Final cleaned dataset: **5,919 rows**
- Fraud cases in cleaned data: **366 (6.18%)**

### Data quality observations

- Merchant category spelling variants were normalized.
- The dataset contains six cities.
- Missing distance values were handled during cleaning.
- Engineered features include `amount_vs_avg_ratio`, `is_night`, and `high_value`.
- `chargeback_filed` was excluded from model features to avoid target leakage.

### Important findings

- A majority-class-only classifier would achieve approximately **93.82% accuracy**, but **0% fraud recall**.
- Among rows with `chargeback_filed = 1`, 304 out of 373 were fraud cases.
- Among rows with `chargeback_filed = 0`, 62 out of 5,546 were fraud cases.

These findings demonstrate why accuracy alone is not a sufficient metric for fraud detection.

## Part B results

### Model training and validation

- Training rows: **3,551**
- Validation rows: **1,184**
- Test rows: **1,184**
- Fraud cases in training, validation, and test sets: **220, 73, and 73**, respectively.
- Missing-value imputation used the training-set median amount/distance statistics as appropriate.
- Oversampling was applied to the training data only.
- Grid search selected a Random Forest configuration with `max_depth=None` and `min_samples_leaf=5`.
- Best cross-validation F1: **0.465**
- Selected decision threshold: **0.40**

### Model evaluation

- Test F1 score: **0.413**
- Test transactions flagged at the selected threshold: **203**
- Clean validation F1: **0.441**
- Leaky validation F1: **0.797**

The higher leaky validation score illustrates how including information that would not legitimately be available at prediction time can make evaluation misleading.

### Feature handling

The model uses numeric and engineered features alongside one-hot encoded merchant category, city, and payment mode features. Identifiers, the target label, and `chargeback_filed` are excluded from model inputs.

The final feature matrix contains **26 features**.

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
