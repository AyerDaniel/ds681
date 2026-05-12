from huggingface_hub import snapshot_download
import zipfile
from pathlib import Path

REPO_ID = "lgrzybowski/seraphim-drone-detection-dataset"
LOCAL_DIR = Path("seraphim-drone-detection-dataset")  # <-- change this

repo_path = Path(snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=LOCAL_DIR))

# Unzip all batch zip files
zip_files = list(repo_path.rglob("*.zip"))
print(f"Found {len(zip_files)} zip files to extract")

for zip_path in zip_files:
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(zip_path.parent)
        zip_path.unlink()  # delete the zip after extracting
        print(f"Extracted: {zip_path.relative_to(repo_path)}")
    except zipfile.BadZipFile:
        print(f"Skipping invalid zip: {zip_path}")

print(f"Done. Dataset ready at: {repo_path.resolve()}")