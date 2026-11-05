
from pathlib import Path
import csv
import random

DATASET = Path("data/Cattely-Cattle-Face-Images-Dataset")
OUTPUT_DIR = Path("experiments")
OUTPUT_DIR.mkdir(exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Reproducible results
random.seed(42)

# Known identities used to train the recognition model
TRAIN_COWS = {
    f"cow_{i}" for i in range(1, 41)
}

# Known identities held out for validation/testing
KNOWN_HOLDOUT_COWS = {
    "cow_41", "cow_42", "cow_43"
}

# Unseen identities reserved for final unknown-cow testing
UNKNOWN_TEST_COWS = {
    "cow_44", "cow_45", "cow_46"
}

if not DATASET.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET.resolve()}"
    )

records = []
counts = {}

for folder in sorted(DATASET.iterdir()):
    if not folder.is_dir():
        continue

    cow = folder.name.lower()

    # Skip non-cow folders such as "valid"
    if not cow.startswith("cow_"):
        continue

    images = sorted(
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    counts[cow] = len(images)

    if cow in TRAIN_COWS:
        # Very small classes cannot support a useful split.
        # Keep the one-image cow out of training.
        if len(images) < 2:
            print(f"Skipping {cow}: only {len(images)} image(s)")
            continue

        shuffled = images.copy()
        random.shuffle(shuffled)

        # 60% train, 20% validation, 20% known-cow test
        n = len(shuffled)
        n_train = max(1, int(n * 0.60))
        n_val = max(1, int(n * 0.20))

        # Ensure at least one test image remains
        if n_train + n_val >= n:
            n_train = n - 2
            n_val = 1

        for i, path in enumerate(shuffled):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "validation"
            else:
                split = "known_test"

            records.append({
                "path": str(path),
                "cow_id": cow,
                "split": split,
            })

    elif cow in KNOWN_HOLDOUT_COWS:
        # These identities are not enrolled in the model.
        # Use for an exploratory known-identity holdout only.
        for path in images:
            records.append({
                "path": str(path),
                "cow_id": cow,
                "split": "unseen_validation",
            })

    elif cow in UNKNOWN_TEST_COWS:
        for path in images:
            records.append({
                "path": str(path),
                "cow_id": cow,
                "split": "unknown_test",
            })

output_file = OUTPUT_DIR / "image_splits.csv"

with output_file.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["path", "cow_id", "split"],
    )
    writer.writeheader()
    writer.writerows(records)

print("\nSplit summary:")
summary = {}

for record in records:
    key = (record["split"], record["cow_id"])
    summary[key] = summary.get(key, 0) + 1

for split in [
    "train",
    "validation",
    "known_test",
    "unseen_validation",
    "unknown_test",
]:
    subset = [
        record for record in records
        if record["split"] == split
    ]
    print(
        f"{split}: {len(subset)} images, "
        f"{len(set(r['cow_id'] for r in subset))} identities"
    )

print(f"\nSaved split manifest: {output_file.resolve()}")
print("Review the summary before proceeding.")
