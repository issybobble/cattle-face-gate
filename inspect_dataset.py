# Use to inspect the contents of the dataset before use

from pathlib import Path

ROOT = Path("data")
extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if not ROOT.exists():
    print("Cannot find data folder:", ROOT.resolve())
    raise SystemExit

total = 0

for folder in sorted(p for p in ROOT.rglob("*") if p.is_dir()):
    images = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ]

    if images:
        print(f"{folder}: {len(images)} images")
        total += len(images)

print(f"\nTotal images: {total}")