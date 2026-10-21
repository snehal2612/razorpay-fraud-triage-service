
import pandas as pd

RAW_FILE = "transactions_raw.csv"


def clean_transactions(raw_df):
    """Clean transaction data without modifying the input DataFrame."""
    df = raw_df.copy()

    # A. Remove exact duplicate rows
    df = df.drop_duplicates()

    # B. Standardize text columns
    for col in ["merchant_category", "city"]:
        df[col] = df[col].str.strip().str.title()

    # C. Remove missing amounts
    df = df.dropna(subset=["amount_inr"])

    # D. Remove impossible amounts
    df = df[
        (df["amount_inr"] >= 1)
        & (df["amount_inr"] <= 1_000_000)
    ]

    # E. Reset index
    df = df.reset_index(drop=True)

    return df


def add_features(df):
    """Add three engineered features without modifying the input."""
    result = df.copy()

    result["amount_vs_avg_ratio"] = (
        result["amount_inr"] / result["avg_amount_30d_inr"]
    )

    result["is_night"] = (
        result["hour_of_day"] <= 5
    ).astype(int)

    result["high_value"] = (
        result["amount_inr"] > 50_000
    ).astype(int)

    return result


def main():
    raw = pd.read_csv(RAW_FILE)

    # Raw data audit
    print("\n========== RAW DATA AUDIT ==========")
    print(f"Rows: {raw.shape[0]}")
    print(f"Columns: {raw.shape[1]}")
    print(f"Exact duplicate rows: {raw.duplicated().sum()}")

    deduplicated = raw.drop_duplicates()

    print(
        "Missing amount_inr after deduplication:",
        deduplicated["amount_inr"].isna().sum()
    )
    print(
        "Missing distance_from_home_km after deduplication:",
        deduplicated["distance_from_home_km"].isna().sum()
    )

    amounts = pd.to_numeric(deduplicated["amount_inr"], errors="coerce")
    print(
        "Out-of-range amounts after deduplication:",
        ((amounts < 1) | (amounts > 1_000_000)).sum()
    )

    print(
        "Distinct raw merchant_category spellings:",
        raw["merchant_category"].nunique()
    )
    print(
        "Distinct raw city spellings:",
        raw["city"].nunique()
    )

    # Clean and engineer features
    cleaned = clean_transactions(raw)
    featured = add_features(cleaned)

    print("\n========== CLEANING CHECKPOINTS ==========")
    print(f"Cleaned rows: {len(cleaned)}")
    print(f"Duplicate rows remaining: {cleaned.duplicated().sum()}")
    print(
        "Missing distances remaining:",
        cleaned["distance_from_home_km"].isna().sum()
    )

    print("\nMerchant category counts:")
    print(cleaned["merchant_category"].value_counts())

    print("\nCity counts:")
    print(cleaned["city"].value_counts())

    print("\nPayment mode counts:")
    print(cleaned["payment_mode"].value_counts())

    print("\nMinimum amount:", round(cleaned["amount_inr"].min(), 2))
    print("Maximum amount:", round(cleaned["amount_inr"].max(), 2))

    print("\n========== FEATURE CHECKPOINTS ==========")
    print("Number of features added:", len(
        set(featured.columns) - set(cleaned.columns)
    ))
    print("is_night sum:", featured["is_night"].sum())
    print("high_value sum:", featured["high_value"].sum())
    print(
        "Maximum amount_vs_avg_ratio:",
        round(featured["amount_vs_avg_ratio"].max(), 2)
    )

    print("\n========== FRAUD AND LEAKAGE AUDIT ==========")
    fraud_count = int(cleaned["is_fraud"].sum())
    fraud_rate = cleaned["is_fraud"].mean() * 100

    print(f"Fraud rows: {fraud_count}")
    print(f"Fraud rate: {fraud_rate:.2f}%")
    print(
        "Never-fraud rule accuracy:",
        f"{(1 - cleaned['is_fraud'].mean()) * 100:.2f}%"
    )
    print("Never-fraud rule recall: 0.00")

    print("\nChargeback leakage table:")
    print(
        cleaned.groupby("chargeback_filed")["is_fraud"]
        .agg(rows="count", fraud_rows="sum")
    )

    excluded = [
        "txn_id", "customer_id", "chargeback_filed", "is_fraud"
    ]
    model_columns = [
        col for col in featured.columns if col not in excluded
    ]

    print("\nModel input columns before one-hot encoding:")
    print(model_columns)
    print("Number of columns before one-hot encoding:", len(model_columns))

    # One-row and input-immutability checks
    one_row = raw.iloc[[0]].copy()
    original = one_row.copy(deep=True)

    clean_one = clean_transactions(one_row)
    assert one_row.equals(original), "Input DataFrame was modified!"

    if not clean_one.empty:
        add_features(clean_one)

    print("\nOne-row and input-immutability checks passed.")


if __name__ == "__main__":
    main()
