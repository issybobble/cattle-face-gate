
from pathlib import Path
import csv
from collections import Counter

import joblib
import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.transform import resize
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Project settings
SPLIT_FILE = Path("experiments/image_splits.csv")
MODEL_DIR = Path("models")
MODEL_FILE = MODEL_DIR / "cow_svm_model.joblib"

IMAGE_SIZE = (128, 128)


def extract_features(image_path):
    """Convert a cattle image into a HOG feature vector."""
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = np.asarray(image)

    gray_image = rgb2gray(image)
    gray_image = resize(
        gray_image,
        IMAGE_SIZE,
        anti_aliasing=True
    )

    features = hog(
        gray_image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features


def main():
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"Split manifest not found: {SPLIT_FILE.resolve()}\n"
            "Run make_splits.py first."
        )

    # Load only the designated training images
    with SPLIT_FILE.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        records = list(csv.DictReader(file))

    training_records = [
        record for record in records
        if record["split"] == "train"
    ]

    if not training_records:
        raise ValueError("No training images were found in the manifest.")

    features_list = []
    labels = []
    skipped = 0

    print(f"Training images listed: {len(training_records)}")
    print("Extracting HOG features...")

    for index, record in enumerate(training_records, start=1):
        image_path = Path(record["path"])
        cow_id = record["cow_id"]

        if not image_path.exists():
            print(f"Skipping missing image: {image_path}")
            skipped += 1
            continue

        try:
            features_list.append(extract_features(image_path))
            labels.append(cow_id)
        except Exception as error:
            print(f"Could not process {image_path}: {error}")
            skipped += 1

        if index % 100 == 0:
            print(f"Processed {index}/{len(training_records)} listed images")

    if len(set(labels)) < 2:
        raise ValueError("At least two cow identities are required to train.")

    X = np.asarray(features_list)
    y = np.asarray(labels)

    print(f"\nImages successfully processed: {len(y)}")
    print(f"Images skipped: {skipped}")
    print(f"Number of training identities: {len(set(y))}")
    print(f"Feature vector size: {X.shape[1]}")

    print("\nTraining the SVM model...")

    model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="linear",
            probability=True,
            random_state=42
        )
    )

    model.fit(X, y)

    # Keep the saved structure compatible with recognise_cow.py
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_data = {
        "model": model,
        "labels": sorted(set(y)),
        "image_size": IMAGE_SIZE,
        "feature_method": "HOG",
        "training_images": len(y),
        "training_identities": sorted(set(y)),
        "split_manifest": str(SPLIT_FILE)
    }

    joblib.dump(model_data, MODEL_FILE)

    print("\nTraining complete.")
    print(f"Saved model to: {MODEL_FILE.resolve()}")
    print("\nTraining images per identity:")

    for cow_id, count in sorted(Counter(y).items()):
        print(f"  {cow_id}: {count}")

    print(
        "\nImportant: validation and test images were not used "
        "to fit the model."
    )


if __name__ == "__main__":
    main()
