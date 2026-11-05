# Additional help for checking the dataset before use

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import random

DATASET_PATH = Path(
    "data/Cattely-Cattle-Face-Images-Dataset"
)

random.seed(42)
extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Find folders that contain images directly
cow_folders = []

for folder in DATASET_PATH.iterdir():
    if not folder.is_dir() or folder.name.lower() == "valid":
        continue

    images = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ]

    if images:
        cow_folders.append((folder.name, images))

# Choose up to 6 cows to preview
selected = random.sample(
    cow_folders, min(6, len(cow_folders))
)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for ax, (cow_name, images) in zip(axes, selected):
    image_path = random.choice(images)

    with Image.open(image_path) as image:
        ax.imshow(image.convert("RGB"))

    ax.set_title(f"{cow_name} ({len(images)} images)")
    ax.axis("off")

for ax in axes[len(selected):]:
    ax.axis("off")

plt.tight_layout()
plt.show()
