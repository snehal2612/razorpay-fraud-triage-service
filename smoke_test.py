
import time
from collections import defaultdict

import requests

BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 10

BENIGN = {
    "merchant_category": "Grocery",
    "city": "Pune",
    "payment_mode": "UPI",
    "amount_inr": 1500,
    "hour_of_day": 13,
    "txn_count_last_24h": 0,
    "avg_amount_30d_inr": 1600,
    "distance_from_home_km": 3.0,
    "is_new_device": 0,
    "account_age_days": 900,
}

RISKY = {
    "merchant_category": "Electronics",
    "city": "Mumbai",
    "payment_mode": "Card",
    "amount_inr": 45000,
    "hour_of_day": 2,
    "txn_count_last_24h": 9,
    "avg_amount_30d_inr": 3000,
    "distance_from_home_km": 60.0,
    "is_new_device": 1,
    "account_age_days": 40,
}

BIG = {
    **BENIGN,
    "merchant_category": "Utilities",
    "payment_mode": "NetBanking",
    "amount_inr": 150000,
    "avg_amount_30d_inr": 250000,
    "distance_from_home_km": 0.0,
    "is_new_device": 1,
}

def payload(base, txn_id, question=None):
    result = {"txn_id": txn_id, **base}
    if question is not None:
        result["question"] = question
    return result


def wait_for_service():
    deadline = time.time() + 120

    while time.time() < deadline:
        try:
            response = requests.get(
                f"{BASE_URL}/health",
                timeout=2,
            )
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass

        time.sleep(2)

    raise RuntimeError("Service did not become healthy within 120 seconds")


def check_response(response, expected_status, checks):
    passed = response.status_code == expected_status

    try:
        body = response.json()
    except ValueError:
        body = {}

    for key, expected in checks.items():
        actual = body

        for part in key.split("."):
            if not isinstance(actual, dict) or part not in actual:
                actual = object()
                break
            actual = actual[part]

        if actual != expected:
            passed = False

    return passed


def main():
    wait_for_service()

    tests = []

    # Q1: Health
    tests.append({
        "id": "Q1", "category": "happy", "method": "GET",
        "path": "/health", "body": None, "status": 200,
        "checks": {"status": "ok"},
    })

    # Q2: Benign score
    tests.append({
        "id": "Q2", "category": "happy", "method": "POST",
        "path": "/score",
        "body": payload(BENIGN, "T800001"),
        "status": 200,
        "checks": {
            "flagged": False,
            "decision": "approve",
            "human_review": False,
        },
    })

    # Q3: Risky score
    tests.append({
        "id": "Q3", "category": "escalation", "method": "POST",
        "path": "/score",
        "body": payload(RISKY, "T800002"),
        "status": 200,
        "checks": {
            "flagged": True,
            "decision": "hold_for_review",
            "human_review": True,
        },
    })

    # Q4: Large payment guardrail
    tests.append({
        "id": "Q4", "category": "escalation", "method": "POST",
        "path": "/score",
        "body": payload(BIG, "T800003"),
        "status": 200,
        "checks": {
            "flagged": False,
            "decision": "hold_for_review",
            "human_review": True,
        },
    })

    # Q5: Invalid amount => 400
    invalid_amount = payload(BENIGN, "T800004")
    invalid_amount["amount_inr"] = -5

    tests.append({
        "id": "Q5", "category": "edge", "method": "POST",
        "path": "/score", "body": invalid_amount,
        "status": 400,
        "checks": {
            "error": "amount_inr must be between 1 and 1000000",
            "field": "amount_inr",
        },
    })

    # Q6: Missing required field => 422
    missing_hour = payload(BENIGN, "T800005")
    del missing_hour["hour_of_day"]

    tests.append({
        "id": "Q6", "category": "edge", "method": "POST",
        "path": "/score", "body": missing_hour,
        "status": 422, "checks": {},
    })

    # Q7: Retry succeeds
    tests.append({
        "id": "Q7", "category": "escalation", "method": "POST",
        "path": "/case",
        "body": payload(
            RISKY, "T900002",
            "more than 6 UPI payments in 24 hours velocity alert",
        ),
        "status": 200,
        "checks": {
            "flagged": True,
            "attempts": 2,
            "safe_default": False,
            "case_note.policy_id": "P2",
            "human_review": True,
        },
    })

    # Q8: Safe default after three failures
    tests.append({
        "id": "Q8", "category": "escalation", "method": "POST",
        "path": "/case",
        "body": payload(
            RISKY, "T900005",
            "electronics merchant payment on an account younger than 30 days",
        ),
        "status": 200,
        "checks": {
            "attempts": 3,
            "safe_default": True,
            "decision": "hold_for_review",
            "human_review": True,
        },
    })

    # Q9: Simulated model outage => 503
    tests.append({
        "id": "Q9", "category": "edge", "method": "POST",
        "path": "/case",
        "body": payload(
            RISKY, "T900099",
            "card payment above INR 50000 from a new device",
        ),
        "status": 503,
        "checks": {
            "error": "case-note service temporarily unavailable, try again later",
        },
    })

    # Q10: Benign case request; no case note
    tests.append({
        "id": "Q10", "category": "happy", "method": "POST",
        "path": "/case",
        "body": payload(
            BENIGN, "T800001",
            "card payment above INR 50000 from a new device",
        ),
        "status": 200,
        "checks": {
            "flagged": False,
            "case_note": None,
            "attempts": 0,
            "decision": "approve",
            "human_review": False,
        },
    })

    # Q11: Out-of-scope question; no matching policy
    tests.append({
        "id": "Q11", "category": "edge", "method": "POST",
        "path": "/case",
        "body": payload(
            RISKY, "T800006",
            "what is the cafeteria menu today",
        ),
        "status": 200,
        "checks": {
            "case_note": None,
            "attempts": 0,
            "decision": "hold_for_review",
            "human_review": True,
        },
    })

    category_totals = defaultdict(lambda: [0, 0])
    all_passed = True

    print("ID  CATEGORY    EXPECTED  OBSERVED  RESULT")

    for test in tests:
        try:
            if test["method"] == "GET":
                response = requests.get(
                    BASE_URL + test["path"],
                    timeout=TIMEOUT,
                )
            else:
                response = requests.post(
                    BASE_URL + test["path"],
                    json=test["body"],
                    timeout=TIMEOUT,
                )

            passed = check_response(
                response,
                test["status"],
                test["checks"],
            )
            observed = response.status_code

        except requests.RequestException as exc:
            passed = False
            observed = f"ERROR: {exc}"

        category_totals[test["category"]][1] += 1
        if passed:
            category_totals[test["category"]][0] += 1

        all_passed = all_passed and passed

        print(
            f"{test['id']:<4}{test['category']:<12}"
            f"{test['status']!s:<10}{observed!s:<10}"
            f"{'PASS' if passed else 'FAIL'}"
        )

    print("\n========== CATEGORY RESULTS ==========")
    for category, (passed, total) in category_totals.items():
        print(f"{category}: {passed}/{total}")

    print("\n========== METRICS ==========")
    metrics = requests.get(
        f"{BASE_URL}/metrics",
        timeout=TIMEOUT,
    )
    print(metrics.json())

    print(
        "\nOVERALL:",
        "11/11 PASS" if all_passed else "SOME TESTS FAILED",
    )


if __name__ == "__main__":
    main()
