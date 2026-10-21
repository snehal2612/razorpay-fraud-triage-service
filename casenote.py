
import json

from retrieval import retrieve, MIN_SCORE
from mock_llm import generate, reset


# Define the rules once using JSON Schema vocabulary.
CASE_NOTE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "priority": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
        "recommended_team": {
            "type": "string",
            "enum": [
                "card_risk",
                "upi_risk",
                "merchant_ops",
                "customer_care",
            ],
        },
        "recommended_action": {
            "type": "string",
            "enum": [
                "approve",
                "hold_for_review",
                "block_and_notify",
            ],
        },
        "policy_id": {
            "type": "string",
            "enum": ["P1", "P2", "P3", "P4", "P5", "P6"],
        },
    },
    "required": [
        "summary",
        "priority",
        "recommended_team",
        "recommended_action",
        "policy_id",
    ],
}


def build_prompt(
    txn_id,
    fraud_probability,
    amount_inr,
    policy_id,
    policy_text,
):
    """Build the structured policy-grounded prompt."""

    return f"""ROLE:
You are a payment fraud risk analyst assistant.

TASK:
Prepare a concise case note for the transaction using the supplied policy.
Transaction details:
{policy_text}

CONSTRAINTS:
Use only the supplied policy text as the policy source.
Do not invent policy rules or unsupported facts.
Reply with one JSON object and nothing else.
All five required fields must be present and contain strings.

OUTPUT FORMAT:
Return a JSON object with exactly these five keys:
summary: a concise transaction case summary (string).
priority: low, medium, or high.
recommended_team: card_risk, upi_risk, merchant_ops, or customer_care.
recommended_action: approve, hold_for_review, or block_and_notify.
policy_id: P1, P2, P3, P4, P5, or P6.

TXN_ID: {txn_id}
FRAUD_PROBABILITY: {fraud_probability}
AMOUNT_INR: {amount_inr}
POLICY_ID: {policy_id}"""


def validate(text, retrieved_policy_id):
    """
    Return (parsed_note, gate).
    Gate 0 means valid; gates 1, 2 and 3 identify failures.
    """

    # Gate 1: Valid JSON object.
    try:
        note = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None, 1

    if not isinstance(note, dict):
        return None, 1

    # Gate 2: Required keys exist and all values are strings.
    required = CASE_NOTE_SCHEMA["required"]
    properties = CASE_NOTE_SCHEMA["properties"]

    for key in required:
        if key not in note:
            return None, 2
        if not isinstance(note[key], str):
            return None, 2

    # Gate 3: Enum restrictions and policy consistency.
    for key, rules in properties.items():
     if "enum" in rules:
        if note[key] not in rules["enum"]:
            return None, 3

    if note["policy_id"] != retrieved_policy_id:
        return None, 3

    return note, 0


def run_case(
    txn_id,
    fraud_probability,
    amount_inr,
    question,
    max_attempts=3,
):
    """Retrieve policy, generate a validated note and apply guardrails."""

    policy_id, policy_text, score = retrieve(question)

    # No model call when the best policy match is too weak.
    if score < MIN_SCORE:
        return {
            "status": "no_policy_match",
            "attempts": 0,
            "failures": [],
            "case_note": None,
            "safe_default": False,
            "guardrail_override": False,
        }

    prompt = build_prompt(
        txn_id,
        fraud_probability,
        amount_inr,
        policy_id,
        policy_text,
    )

    failures = []
    note = None
    attempts = 0
    safe_default = False

    for _ in range(max_attempts):
        attempts += 1

        # ConnectionError intentionally propagates to the caller.
        response = generate(prompt)
        note, gate = validate(response, policy_id)

        if gate == 0:
            break

        failures.append(gate)
        note = None

    if note is None:
        safe_default = True
        note = {
            "summary": (
                "Automated case note unavailable; "
                "manual review required."
            ),
            "priority": "high",
            "recommended_team": "customer_care",
            "recommended_action": "hold_for_review",
            "policy_id": policy_id,
        }

    # Business guardrail: high fraud probability must not be approved.
    guardrail_override = False

    if (
        note["recommended_action"] == "approve"
        and fraud_probability >= 0.5
    ):
        note["recommended_action"] = "hold_for_review"
        guardrail_override = True

    return {
        "status": "ok",
        "attempts": attempts,
        "failures": failures,
        "case_note": note,
        "safe_default": safe_default,
        "guardrail_override": guardrail_override,
    }


def main():
    with open("case_scenarios.json", "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    # Ensure deterministic scenario results on each run.
    reset()

    print("========== CASE NOTE SCENARIOS ==========")

    rows = []
    total_calls = 0
    first_call_success = 0
    recovered = 0
    safe_defaults = 0
    overrides = 0

    for index, scenario in enumerate(scenarios, start=1):
        result = run_case(
            txn_id=scenario["txn_id"],
            fraud_probability=scenario["fraud_probability"],
            amount_inr=scenario["amount_inr"],
            question=scenario["question"],
        )

        note = result["case_note"]
        action = note["recommended_action"] if note else None

        rows.append({
            "scenario": f"S{index} ({scenario['txn_id']})",
            "retrieved_policy": note["policy_id"] if note else "-",
            "attempts": result["attempts"],
            "failures": result["failures"],
            "status": result["status"],
            "safe_default": result["safe_default"],
            "final_action": action,
            "override": result["guardrail_override"],
        })

        total_calls += result["attempts"]

        if result["attempts"] == 1 and not result["failures"]:
            first_call_success += 1

        if result["failures"] and not result["safe_default"]:
            recovered += 1

        safe_defaults += int(result["safe_default"])
        overrides += int(result["guardrail_override"])

    # Print a readable table without requiring another dependency.
    headers = list(rows[0].keys())
    print(" | ".join(headers))
    print("-" * 140)

    for row in rows:
        print(" | ".join(str(row[h]) for h in headers))

    print("\n========== TOTALS ==========")
    print("Total model calls:", total_calls)
    print("Valid on first call:", first_call_success)
    print("Recovered by retry:", recovered)
    print("Safe defaults:", safe_defaults)
    print("Guardrail overrides:", overrides)

    # Verify that the simulated outage propagates.
    print("\n========== OUTAGE CHECK ==========")
    try:
        run_case(
            "T900099",
            0.96,
            82000,
            "card payment above INR 50000 from a new device",
        )
    except ConnectionError as exc:
        print("ConnectionError propagated:", str(exc))


if __name__ == "__main__":
    main()
