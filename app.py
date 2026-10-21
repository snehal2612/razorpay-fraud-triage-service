
import logging
import time

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from train import build_final_model
from prepare import add_features
from casenote import run_case


# Build the model once when the service starts.
MODEL_BUNDLE = build_final_model()
MODEL = MODEL_BUNDLE["model"]
MODEL_COLUMNS = MODEL_BUNDLE["columns"]
TRAINING_MEDIAN = MODEL_BUNDLE["median"]
THRESHOLD = MODEL_BUNDLE["threshold"]

KNOWN_CATEGORIES = {
    "Grocery",
    "Electronics",
    "Travel",
    "Food Delivery",
    "Fashion",
    "Utilities",
}

app = FastAPI(title="Merchant Risk Desk — Fraud Triage Service")

# Required: one log line per POST request.
logging.basicConfig(
    filename="service.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    force=True,
)
logger = logging.getLogger("fraud_triage")


# In-memory metrics for the current process.
METRICS = {
    "requests_total": 0,
    "success_total": 0,
    "error_total": 0,
    "latency_total_ms": 0.0,
    "human_review_total": 0,
}


class Txn(BaseModel):
    txn_id: str
    merchant_category: str
    city: str
    payment_mode: str
    amount_inr: float
    hour_of_day: int
    txn_count_last_24h: int
    avg_amount_30d_inr: float
    distance_from_home_km: float | None = None
    is_new_device: int
    account_age_days: int


class CaseIn(Txn):
    question: str


def validate_business_input(txn: Txn):
    """Reject impossible values before model inference."""

    if not 1 <= txn.amount_inr <= 1_000_000:
        return {
            "error": "amount_inr must be between 1 and 1000000",
            "field": "amount_inr",
        }

    if not 0 <= txn.hour_of_day <= 23:
        return {
            "error": "hour_of_day must be between 0 and 23",
            "field": "hour_of_day",
        }

    if txn.txn_count_last_24h < 0:
        return {
            "error": "txn_count_last_24h must be non-negative",
            "field": "txn_count_last_24h",
        }

    if txn.avg_amount_30d_inr <= 0:
        return {
            "error": "avg_amount_30d_inr must be greater than 0",
            "field": "avg_amount_30d_inr",
        }

    category = txn.merchant_category.strip().title()
    if category not in KNOWN_CATEGORIES:
        return {
            "error": "Unknown merchant_category",
            "field": "merchant_category",
        }

    return None


def score_transaction(txn: Txn):
    """Normalize, engineer, align features and score one payment."""

    row = txn.model_dump() if hasattr(txn, "model_dump") else txn.dict()

    # Normalize the category and city.
    row["merchant_category"] = txn.merchant_category.strip().title()
    row["city"] = txn.city.strip().title()

    # Missing distance is filled using the training median only.
    if row["distance_from_home_km"] is None:
        row["distance_from_home_km"] = TRAINING_MEDIAN

    df = pd.DataFrame([row])
    df = add_features(df)

    # Match the exact training feature names and column order.
    df = pd.get_dummies(
        df,
        columns=["merchant_category", "city", "payment_mode"],
        dtype=int,
    )

    df = df.reindex(columns=MODEL_COLUMNS, fill_value=0)

    probability = float(MODEL.predict_proba(df)[0, 1])
    flagged = probability >= THRESHOLD

    decision = "hold_for_review" if flagged else "approve"

    # Extra business guardrail for a very large new-device payment.
    large_payment_hold = (
        not flagged
        and txn.amount_inr >= 100_000
        and txn.is_new_device == 1
    )

    if large_payment_hold:
        decision = "hold_for_review"

    return {
        "txn_id": txn.txn_id,
        "fraud_probability": round(probability, 4),
        "flagged": bool(flagged),
        "decision": decision,
        "human_review": decision == "hold_for_review",
        "large_payment_hold": large_payment_hold,
    }


def error_response(status_code, message, field=None):
    body = {"error": message}
    if field is not None:
        body["field"] = field
    return JSONResponse(status_code=status_code, content=body)


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    """Log and count every POST response, including framework errors."""

    started = time.perf_counter()
    request.state.info = {}

    try:
        response = await call_next(request)
    except Exception:
        # Prevent an unexpected exception from bypassing request accounting.
        response = JSONResponse(
            status_code=500,
            content={"error": "internal server error"},
        )

    if request.method == "POST" and request.url.path in {"/score", "/case"}:
        latency_ms = (time.perf_counter() - started) * 1000
        info = request.state.info
        status = response.status_code

        METRICS["requests_total"] += 1
        METRICS["latency_total_ms"] += latency_ms

        if status < 400:
            METRICS["success_total"] += 1
        else:
            METRICS["error_total"] += 1

        if status < 400 and info.get("human_review", False):
            METRICS["human_review_total"] += 1

        if status >= 500:
            level = logging.ERROR
        elif (
            status >= 400
            or info.get("safe_default", False)
            or info.get("guardrail_override", False)
            or info.get("large_payment_hold", False)
        ):
            level = logging.WARNING
        else:
            level = logging.INFO

        message = (
            f"endpoint={request.url.path} "
            f"status={status} latency_ms={latency_ms:.2f}"
        )

        if info.get("txn_id") is not None:
            message += f" txn_id={info['txn_id']}"

        if info.get("fraud_probability") is not None:
            message += (
                f" fraud_probability={info['fraud_probability']:.4f}"
            )

        logger.log(level, message)

    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score")
def score(txn: Txn, request: Request):
    invalid = validate_business_input(txn)

    if invalid:
        return JSONResponse(status_code=400, content=invalid)

    result = score_transaction(txn)

    request.state.info.update({
        "txn_id": result["txn_id"],
        "fraud_probability": result["fraud_probability"],
        "human_review": result["human_review"],
        "large_payment_hold": result["large_payment_hold"],
    })

    # Public /score response has exactly the five required fields.
    return {
        "txn_id": result["txn_id"],
        "fraud_probability": result["fraud_probability"],
        "flagged": result["flagged"],
        "decision": result["decision"],
        "human_review": result["human_review"],
    }


@app.post("/case")
def create_case(payload: CaseIn, request: Request):
    invalid = validate_business_input(payload)

    if invalid:
        return JSONResponse(status_code=400, content=invalid)

    result = score_transaction(payload)

    case_note = None
    attempts = 0
    safe_default = False
    guardrail_override = False
    decision = result["decision"]

    if result["flagged"]:
        try:
            case_result = run_case(
                txn_id=payload.txn_id,
                fraud_probability=result["fraud_probability"],
                amount_inr=payload.amount_inr,
                question=payload.question,
            )
        except ConnectionError:
            request.state.info.update({
                "txn_id": payload.txn_id,
                "fraud_probability": result["fraud_probability"],
            })
            return error_response(
                503,
                "case-note service temporarily unavailable, try again later",
            )

        attempts = case_result["attempts"]
        safe_default = case_result["safe_default"]
        guardrail_override = case_result["guardrail_override"]
        case_note = case_result["case_note"]

        if case_result["status"] == "no_policy_match":
            decision = "hold_for_review"
        elif case_note is not None:
            decision = case_note["recommended_action"]

            # A flagged payment must never be auto-approved.
            if decision == "approve":
                decision = "hold_for_review"
                case_note["recommended_action"] = "hold_for_review"
                guardrail_override = True
    else:
        # Unflagged payment: no policy note is generated.
        case_note = None
        attempts = 0

    human_review = (
        decision in {"hold_for_review", "block_and_notify"}
        or safe_default
    )

    request.state.info.update({
        "txn_id": payload.txn_id,
        "fraud_probability": result["fraud_probability"],
        "human_review": human_review,
        "safe_default": safe_default,
        "guardrail_override": guardrail_override,
        "large_payment_hold": result["large_payment_hold"],
    })

    return {
        "txn_id": result["txn_id"],
        "fraud_probability": result["fraud_probability"],
        "flagged": result["flagged"],
        "decision": decision,
        "human_review": human_review,
        "case_note": case_note,
        "attempts": attempts,
        "safe_default": safe_default,
        "guardrail_override": guardrail_override,
    }


@app.get("/metrics")
def metrics():
    total = METRICS["requests_total"]
    success = METRICS["success_total"]
    errors = METRICS["error_total"]
    reviews = METRICS["human_review_total"]

    return {
        "requests_total": total,
        "success_total": success,
        "error_total": errors,
        "success_rate": round(success / total, 4) if total else 0.0,
        "error_rate": round(errors / total, 4) if total else 0.0,
        "avg_latency_ms": (
            round(METRICS["latency_total_ms"] / total, 2)
            if total else 0.0
        ),
        "human_review_total": reviews,
        "human_review_rate": round(reviews / total, 4) if total else 0.0,
    }
