
from pathlib import Path
import csv

import joblib
import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.transform import resize
from sklearn.metrics import accuracy_score, classification_report


SPLIT_FILE = Path("experiments/image_splits.csv")
MODEL_FILE = Path("models/cow_svm_model.joblib")
IMAGE_SIZE = (128, 128)

# Try candidate thresholds using validation data only.
THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def extract_features(image_path):
    with Image.open(image_path) as image:
        image = np.asarray(image.convert("RGB"))

    gray_image = rgb2gray(image)
    gray_image = resize(
        gray_image,
        IMAGE_SIZE,
        anti_aliasing=True
    )

    return hog(
        gray_image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )


def load_records():
    with SPLIT_FILE.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def get_predictions(model, records):
    results = []

    for index, record in enumerate(records, start=1):
        path = Path(record["path"])

        if not path.exists():
            print(f"Skipping missing image: {path}")
            continue

        try:
            features = extract_features(path).reshape(1, -1)
            probabilities = model.predict_proba(features)[0]
            best_index = int(np.argmax(probabilities))

            results.append({
                "actual": record["cow_id"],
                "predicted": str(model.classes_[best_index]),
                "confidence": float(probabilities[best_index]),
                "split": record["split"],
                "path": str(path)
            })

        except Exception as error:
            print(f"Could not evaluate {path}: {error}")

        if index % 50 == 0:
            print(f"Evaluated {index}/{len(records)} listed images")

    return results


def gate_decision(result, threshold):
    # A cow not enrolled in the trained model cannot be identified
    # as an authorised identity simply because it has a high score.
    if result["confidence"] < threshold:
        return "rejected"
    return "identified"


def main():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE.resolve()}"
        )
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"Split manifest not found: {SPLIT_FILE.resolve()}"
        )

    model_data = joblib.load(MODEL_FILE)
    model = model_data["model"]
    records = load_records()

    split_names = [
        "validation",
        "unseen_validation",
        "known_test",
        "unknown_test"
    ]

    predictions = {}

    for split_name in split_names:
        split_records = [
            row for row in records if row["split"] == split_name
        ]
        print(f"\nEvaluating {split_name}: {len(split_records)} images")
        predictions[split_name] = get_predictions(model, split_records)

    # Choose threshold on validation data only.
    # The criterion balances known-cow acceptance with rejection
    # of cows absent from the training identities.
    known_val = predictions["validation"]
    unknown_val = predictions["unseen_validation"]

    if not known_val or not unknown_val:
        raise ValueError(
            "Validation groups are empty. Check experiments/image_splits.csv."
        )

    threshold_results = []

    print("\nThreshold selection (validation data only)")
    print("Threshold | Known accepted correctly | Unknown rejected")

    for threshold in THRESHOLDS:
        known_correct = sum(
            r["predicted"] == r["actual"]
            and r["confidence"] >= threshold
            for r in known_val
        ) / len(known_val)

        unknown_rejected = sum(
            r["confidence"] < threshold
            for r in unknown_val
        ) / len(unknown_val)

        # Simple balanced objective: give both validation groups equal weight.
        score = (known_correct + unknown_rejected) / 2

        threshold_results.append((score, threshold, known_correct, unknown_rejected))
        print(
            f"{threshold:>9.2f} | "
            f"{known_correct:>25.1%} | "
            f"{unknown_rejected:>15.1%}"
        )

    # If scores tie, prefer the higher threshold to be more cautious.
    _, selected_threshold, _, _ = max(
        threshold_results,
        key=lambda item: (item[0], item[1])
    )

    print(f"\nSelected threshold: {selected_threshold:.2f}")
    print("This threshold was selected without using the test sets.")

    # Evaluate the final threshold on held-out test data.
    known_test = predictions["known_test"]
    unknown_test = predictions["unknown_test"]

    if not known_test or not unknown_test:
        raise ValueError("A test group is empty. Check the split manifest.")

    # Known-cow performance: correct identity AND confidence meets threshold.
    correct_accepted = sum(
        r["predicted"] == r["actual"]
        and r["confidence"] >= selected_threshold
        for r in known_test
    )
    false_reject_known = sum(
        r["confidence"] < selected_threshold
        for r in known_test
    )
    wrong_identity_accepted = sum(
        r["predicted"] != r["actual"]
        and r["confidence"] >= selected_threshold
        for r in known_test
    )

    # Unknown cows are not in the enrolled model identities.
    unknown_accepted = sum(
        r["confidence"] >= selected_threshold
        for r in unknown_test
    )
    unknown_rejected = len(unknown_test) - unknown_accepted

    print("\nFINAL TEST RESULTS")
    print(f"Known test images: {len(known_test)}")
    print(
        "Correctly identified and accepted: "
        f"{correct_accepted} ({correct_accepted / len(known_test):.1%})"
    )
    print(
        "Rejected known images (confidence below threshold): "
        f"{false_reject_known} ({false_reject_known / len(known_test):.1%})"
    )
    print(
        "Wrong identity accepted: "
        f"{wrong_identity_accepted} "
        f"({wrong_identity_accepted / len(known_test):.1%})"
    )

    print(f"\nUnknown test images: {len(unknown_test)}")
    print(
        "Unknown cows rejected: "
        f"{unknown_rejected} ({unknown_rejected / len(unknown_test):.1%})"
    )
    print(
        "Unknown cows accepted by the confidence rule: "
        f"{unknown_accepted} ({unknown_accepted / len(unknown_test):.1%})"
    )

    # Identify known cows only, without applying the gate threshold.
    actual = [r["actual"] for r in known_test]
    predicted = [r["predicted"] for r in known_test]

    print("\nKnown-test identity classification report (before threshold):")
    print(classification_report(
        actual,
        predicted,
        zero_division=0
    ))

    output_file = Path("experiments/evaluation_results.csv")
    with output_file.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "split", "path", "actual", "predicted", "confidence",
            "selected_threshold"
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for split_name in split_names:
            for result in predictions[split_name]:
                writer.writerow({
                    "split": result["split"],
                    "path": result["path"],
                    "actual": result["actual"],
                    "predicted": result["predicted"],
                    "confidence": f'{result["confidence"]:.6f}',
                    "selected_threshold": f"{selected_threshold:.2f}"
                })

    print(f"\nDetailed predictions saved to: {output_file.resolve()}")


if __name__ == "__main__":
    main()
