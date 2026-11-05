
from pathlib import Path
import csv

RESULTS_FILE = Path("experiments/evaluation_results.csv")

ALLOWED_COWS = {"cow_1", "cow_2"}
THRESHOLD = 0.30


def rate(count, total):
    if total == 0:
        return "N/A (no images)"
    return f"{count}/{total} ({count / total:.1%})"


def main():
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Results file not found: {RESULTS_FILE.resolve()}\n"
            "Run evaluate_model.py first."
        )

    with RESULTS_FILE.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        records = list(csv.DictReader(file))

    # Use test sets only. Validation data is not included here.
    test_records = [
        row for row in records
        if row["split"] in {"known_test", "unknown_test"}
    ]

    for row in test_records:
        confidence = float(row["confidence"])
        predicted = row["predicted"]

        row["gate_open"] = (
            confidence >= THRESHOLD
            and predicted in ALLOWED_COWS
        )

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

    true_accepts = sum(row["gate_open"] for row in authorised)
    false_rejects = len(authorised) - true_accepts

    false_accepts_known = sum(
        row["gate_open"] for row in unauthorised_known
    )
    true_rejects_known = len(unauthorised_known) - false_accepts_known

    false_accepts_unknown = sum(
        row["gate_open"] for row in unknown
    )
    true_rejects_unknown = len(unknown) - false_accepts_unknown

    print("\nVIRTUAL GATE EVALUATION")
    print(f"Authorised identities: {sorted(ALLOWED_COWS)}")
    print(f"Confidence threshold: {THRESHOLD:.2f}")

    print("\nAuthorised cows")
    print(f"Test images: {len(authorised)}")
    print(f"True accepts: {rate(true_accepts, len(authorised))}")
    print(f"False rejects: {rate(false_rejects, len(authorised))}")

    print("\nUnauthorised enrolled cows")
    print(f"Test images: {len(unauthorised_known)}")
    print(
        "False accepts (gate opened): "
        f"{rate(false_accepts_known, len(unauthorised_known))}"
    )
    print(
        "True rejects (gate stayed closed): "
        f"{rate(true_rejects_known, len(unauthorised_known))}"
    )

    print("\nUnseen cows")
    print(f"Test images: {len(unknown)}")
    print(
        "False accepts (gate opened): "
        f"{rate(false_accepts_unknown, len(unknown))}"
    )
    print(
        "True rejects (gate stayed closed): "
        f"{rate(true_rejects_unknown, len(unknown))}"
    )

    # Save each test decision for later inspection.
    output_file = Path("experiments/gate_test_results.csv")
    with output_file.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        fields = [
            "split", "path", "actual", "predicted",
            "confidence", "gate_open"
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for row in test_records:
            writer.writerow({
                "split": row["split"],
                "path": row["path"],
                "actual": row["actual"],
                "predicted": row["predicted"],
                "confidence": row["confidence"],
                "gate_open": row["gate_open"]
            })

    print(f"\nDetailed gate decisions saved to: {output_file.resolve()}")


if __name__ == "__main__":
    main()
