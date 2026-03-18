import os
import shutil
from pathlib import Path


def get_webp_files(source_dir):
    """Return all .webp files recursively"""
    return list(Path(source_dir).rglob("*.webp"))


# ✅ CHANGE THESE PATHS ONLY
SOURCE_DIR = r"D:\Downloads\giphy.com"
DEST_DIR = r"D:\Project1\two-way-sign-language-translator\filtered_data"

# Create destination folder if not exists
os.makedirs(DEST_DIR, exist_ok=True)

files = get_webp_files(SOURCE_DIR)

print(f"Found {len(files)} webp files\n")

count = 0

for file_path in files:

    # keep original filename
    filename = file_path.name
    dest_path = os.path.join(DEST_DIR, filename)

    # avoid overwrite
    if os.path.exists(dest_path):
        name, ext = os.path.splitext(filename)
        dest_path = os.path.join(DEST_DIR, f"{name}_{count}{ext}")

    shutil.copy2(file_path, dest_path)

    print(f"Copied -> {dest_path}")
    count += 1

print("\n✅ Extraction completed!")