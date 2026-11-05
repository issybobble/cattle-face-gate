
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

    validation = [
        row for row in records
        if row["split"] == "validation"
        and row["actual"] in ALLOWED_COWS
    ]

    unseen_validation = [
        row for row in records
        if row["split"] == "unseen_validation"
    ]

    unauthorised_validation = [
        row for row in records
        if row["split"] == "validation"
        and row["actual"] not in ALLOWED_COWS
    ]

    if not validation or not unseen_validation:
        raise ValueError("Required validation groups are empty.")

    print("VALIDATION-BASED THRESHOLD COMPARISON")
    print(f"Authorised validation images: {len(validation)}")
    print(f"Unauthorised enrolled validation images: {len(unauthorised_validation)}")
    print(f"Unseen validation images: {len(unseen_validation)}")
    print()
    print(
        "Threshold | Authorised accepted | "
        "Unauthorised known accepted | Unseen cows accepted"
    )

    for threshold in THRESHOLDS:
        authorised_accepted = sum(
            float(r["confidence"]) >= threshold
            and r["predicted"] in ALLOWED_COWS
            for r in validation
        )

        unauthorised_accepted = sum(
            float(r["confidence"]) >= threshold
            and r["predicted"] in ALLOWED_COWS
            for r in unauthorised_validation
        )

        unseen_accepted = sum(
            float(r["confidence"]) >= threshold
            and r["predicted"] in ALLOWED_COWS
            for r in unseen_validation
        )

        print(
            f"{threshold:>9.2f} | "
            f"{rate(authorised_accepted, len(validation)):>19} | "
            f"{rate(unauthorised_accepted, len(unauthorised_validation)):>26} | "
            f"{rate(unseen_accepted, len(unseen_validation)):>19}"
        )


if __name__ == "__main__":
    main()
