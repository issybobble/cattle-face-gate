
from pathlib import Path
import csv

RESULTS_FILE = Path("experiments/evaluation_results.csv")

ALLOWED_COWS = {"cow_1", "cow_2"}
THRESHOLDS = [0.30, 0.35, 0.40, 0.45]


def rate(count, total):
    return f"{count}/{total} ({count / total:.1%})" if total else "N/A"


def main():
    with RESULTS_FILE.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        records = list(csv.DictReader(file))

    # Use held-out test images for this diagnostic comparison.
    test_records = [
        row for row in records
        if row["split"] in {"known_test", "unknown_test"}
    ]

    authorised = [
        row for row in test_records
        if row["actual"] in ALLOWED_COWS
    ]

    unauthorised_known = [
        row for row in test_records
        if row["split"] == "known_test"
        and row["actual"] not in ALLOWED_COWS
    ]

    unknown = [
        row for row in test_records
        if row["split"] == "unknown_test"
    ]

    print("THRESHOLD COMPARISON")
    print("Diagnostic only: do not select a threshold based on test results.")
    print()
    print(
        "Threshold | Authorised accepted | "
        "Unauthorised known accepted | Unknown accepted"
    )

    for threshold in THRESHOLDS:
        true_accepts = sum(
            float(r["confidence"]) >= threshold
            and r["predicted"] in ALLOWED_COWS
            for r in authorised
        )

        false_accepts_known = sum(
            float(r["confidence"]) >= threshold
            and r["predicted"] in ALLOWED_COWS
            for r in unauthorised_known
        )

        false_accepts_unknown = sum(
            float(r["confidence"]) >= threshold
            and r["predicted"] in ALLOWED_COWS
            for r in unknown
        )

        print(
            f"{threshold:>9.2f} | "
            f"{rate(true_accepts, len(authorised)):>19} | "
            f"{rate(false_accepts_known, len(unauthorised_known)):>26} | "
            f"{rate(false_accepts_unknown, len(unknown)):>16}"
        )


if __name__ == "__main__":
    main()
