
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageOps
from skimage.feature import hog

MODEL_PATH = Path("models/cow_svm_model.joblib")

# Example policy: only these cows are authorised.
# Change this list to match the intended farm policy.
ALLOWED_COWS = {"cow_1", "cow_2"}

# Demonstration threshold only; it needs proper validation (see select_threshold.py and compare_thresholds.py).
CONFIDENCE_THRESHOLD = 0.30


def extract_features(image_path, image_size):
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB").resize(tuple(image_size))
        gray = np.asarray(image.convert("L"))

    return hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH.resolve()}. "
        "Run train_model.py first."
    )

saved = joblib.load(MODEL_PATH)
model = saved["model"]
image_size = saved["image_size"]

image_input = input(
    "Enter the path to a cow image: "
).strip().strip('"')

image_path = Path(image_input)

if not image_path.is_file():
    raise FileNotFoundError(
        f"Image not found: {image_path.resolve()}"
    )

features = extract_features(image_path, image_size)
features = features.reshape(1, -1)


probabilities = model.predict_proba(features)[0]

# Show the three highest-scoring identities
top_indices = np.argsort(probabilities)[::-1][:3]

print("\n--- Top predictions ---")
for index in top_indices:
    print(
        f"{model.classes_[index]}: "
        f"{probabilities[index]:.1%}"
    )

best_index = int(top_indices[0])
predicted_cow = str(model.classes_[best_index])
confidence = float(probabilities[best_index])

print("\n--- Recognition result ---")
print(f"Predicted identity: {predicted_cow}")
print(f"Top-class score: {confidence:.1%}")

print("\n--- Virtual gate ---")

if confidence < CONFIDENCE_THRESHOLD:
    print("GATE CLOSED")
    print("Reason: uncertain recognition.")
elif predicted_cow not in ALLOWED_COWS:
    print("GATE CLOSED")
    print("Reason: cow is not authorised.")
else:
    print("GATE OPEN")
    print("Reason: recognised cow is authorised.")

print("\nSimulation only — no physical gate is controlled.")
