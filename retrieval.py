
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


with open("fraud_policies.json", "r", encoding="utf-8") as f:
    POLICIES = json.load(f)

with open("retrieval_queries.json", "r", encoding="utf-8") as f:
    QUERIES = json.load(f)

# Fit TF-IDF on the six policy texts only.
VECTORIZER = TfidfVectorizer()
POLICY_TEXTS = [policy["text"] for policy in POLICIES]
POLICY_MATRIX = VECTORIZER.fit_transform(POLICY_TEXTS)

# Based on the assignment's measured scores:
# highest out-of-scope score = 0.4251
# lowest in-scope score = 0.4862
MIN_SCORE = 0.45


def retrieve(query):
    """Return (policy_id, policy_text, similarity_score)."""
    query_vector = VECTORIZER.transform([query])
    scores = cosine_similarity(query_vector, POLICY_MATRIX)[0]

    best_index = int(scores.argmax())
    policy = POLICIES[best_index]

    return (
        policy["policy_id"],
        policy["text"],
        float(scores[best_index]),
    )


def main():
    print("========== POLICY RETRIEVAL RESULTS ==========")

    hits = 0
    in_scope = 0
    out_of_scope_scores = []
    in_scope_scores = []

    rows = []

    for item in QUERIES:
        query = item["query"]
        expected = item["expected_policy"]

        policy_id, _, score = retrieve(query)

        # Reject queries below the calibrated threshold.
        accepted_policy = policy_id if score >= MIN_SCORE else None

        if expected is not None:
            in_scope += 1
            in_scope_scores.append(score)
            if accepted_policy == expected:
                hits += 1
        else:
            out_of_scope_scores.append(score)

        rows.append({
            "query": query,
            "expected_policy": expected,
            "retrieved_policy": policy_id,
            "score": round(score, 4),
            "accepted_policy": accepted_policy,
        })

    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nIn-scope top-1 hits: {hits}/{in_scope}")
    print("MIN_SCORE:", MIN_SCORE)

    if out_of_scope_scores:
        print(
            "Highest out-of-scope score:",
            f"{max(out_of_scope_scores):.4f}",
        )

    if in_scope_scores:
        print(
            "Lowest in-scope score:",
            f"{min(in_scope_scores):.4f}",
        )


if __name__ == "__main__":
    main()
