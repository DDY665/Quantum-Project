from pathlib import Path
from collections import Counter

import pandas as pd
from PIL import Image


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_ROOT = PROJECT_ROOT / "data" / "raw"

KERMANY_DIR = RAW_ROOT / "kermany"
RSNA_DIR = RAW_ROOT / "rsna"

RESULTS_DIR = PROJECT_ROOT / "results" / "logs"
REPORT_FILE = RESULTS_DIR / "dataset_inspection.txt"


# ============================================================
# CONFIGURATION
# ============================================================

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}

VALID_KERMANY_CLASSES = {
    "NORMAL",
    "PNEUMONIA",
}

VALID_SPLITS = {
    "train",
    "val",
    "test",
}


# ============================================================
# KERMANY IMAGE DISCOVERY
# ============================================================


def find_kermany_images():

    if not KERMANY_DIR.exists():
        print(
            f"[WARNING] Kermany directory does not exist:\n"
            f"          {KERMANY_DIR}"
        )
        return []

    records = []

    ignored_macos = 0

    for image_path in KERMANY_DIR.rglob("*"):

        if not image_path.is_file():
            continue

        if "__MACOSX" in image_path.parts:
            ignored_macos += 1
            continue

        if image_path.name.startswith("._"):
            ignored_macos += 1
            continue

        if not is_valid_image_candidate(image_path):
            continue

        label = None

        for parent in image_path.parents:
            if parent.name.upper() in VALID_KERMANY_CLASSES:
                label = parent.name.upper()
                break

        split = None

        for parent in image_path.parents:
            if parent.name.lower() in VALID_SPLITS:
                split = parent.name.lower()
                break

        records.append(
            {
                "path": image_path,
                "split": split,
                "original_label": label,
            }
        )

    print(f"\nIgnored macOS metadata files: {ignored_macos}")

    return records

# ============================================================
# KERMANY INSPECTION
# ============================================================


def inspect_kermany():

    print("\n")
    print("=" * 70)
    print("KERMANY DATASET INSPECTION")
    print("=" * 70)

    records = find_kermany_images()

    if not records:
        print("No Kermany images found.")
        return {
            "records": [],
            "total": 0,
            "corrupted": [],
        }

    print(f"\nTotal image files found: {len(records)}")

    split_counts = Counter()
    class_counts = Counter()
    split_class_counts = Counter()
    format_counts = Counter()
    size_counts = Counter()

    corrupted = []
    unknown_split = []
    unknown_class = []

    # --------------------------------------------------------
    # Inspect every image
    # --------------------------------------------------------

    for record in records:
        image_path = record["path"]

        split = record["split"]
        label = record["original_label"]

        # Split
        if split is not None:
            split_counts[split] += 1
        else:
            unknown_split.append(image_path)

        # Class
        if label is not None:
            class_counts[label] += 1

        else:
            unknown_class.append(image_path)

        if split is not None and label is not None:
            split_class_counts[(split, label)] += 1

        format_counts[image_path.suffix.lower()] += 1

        # Image validation
        try:
            with Image.open(image_path) as image:
                image.verify()

            with Image.open(image_path) as image:
                image.load()

                size_counts[image.size] += 1

        except Exception as error:
            corrupted.append(
                {
                    "path": image_path,
                    "error": str(error),
                }
            )

    # --------------------------------------------------------
    # Print split distribution
    # --------------------------------------------------------

    print("\nSplit distribution:")

    for split in [
        "train",
        "val",
        "test",
    ]:
        print(f"  {split.upper():5}: {split_counts.get(split, 0)}")

    # --------------------------------------------------------
    # Print overall class distribution
    # --------------------------------------------------------

    print("\nOverall class distribution:")

    print(f"  NORMAL:    {class_counts.get('NORMAL', 0)}")

    print(f"  PNEUMONIA: {class_counts.get('PNEUMONIA', 0)}")

    # --------------------------------------------------------
    # Print split + class
    # --------------------------------------------------------

    print("\nSplit / class distribution:")

    for split in [
        "train",
        "val",
        "test",
    ]:
        print(f"\n  {split.upper()}")

        print(f"    NORMAL:    {split_class_counts.get((split, 'NORMAL'), 0)}")

        print(f"    PNEUMONIA: {split_class_counts.get((split, 'PNEUMONIA'), 0)}")

    # --------------------------------------------------------
    # Formats
    # --------------------------------------------------------

    print("\nImage formats:")

    for extension, count in sorted(format_counts.items()):
        print(f"  {extension}: {count}")

    # --------------------------------------------------------
    # Dimensions
    # --------------------------------------------------------

    print("\nMost common image dimensions:")

    for size, count in size_counts.most_common(10):
        print(f"  {size[0]} x {size[1]}: {count}")

    # --------------------------------------------------------
    # Corrupted images
    # --------------------------------------------------------

    print(f"\nCorrupted/unreadable images: {len(corrupted)}")

    if corrupted:
        print("\nFirst 10 corrupted files:")

        for item in corrupted[:10]:
            print(f"  {item['path']}")

            print(f"    Error: {item['error']}")

    # --------------------------------------------------------
    # Unknown structure
    # --------------------------------------------------------

    print(f"\nImages with unknown split: {len(unknown_split)}")

    print(f"Images with unknown class: {len(unknown_class)}")

    return {
        "records": records,
        "total": len(records),
        "split_counts": split_counts,
        "class_counts": class_counts,
        "split_class_counts": split_class_counts,
        "corrupted": corrupted,
        "unknown_split": unknown_split,
        "unknown_class": unknown_class,
    }


# ============================================================
# RSNA INSPECTION
# ============================================================


def inspect_rsna():

    print("\n")
    print("=" * 70)
    print("RSNA DATASET INSPECTION")
    print("=" * 70)

    if not RSNA_DIR.exists():
        print(f"[WARNING] RSNA directory does not exist:\n          {RSNA_DIR}")

        return {
            "dicom_count": 0,
            "csv_files": [],
        }

    # --------------------------------------------------------
    # DICOM files
    # --------------------------------------------------------

    dicom_files = list(RSNA_DIR.rglob("*.dcm"))

    print(f"\nDICOM files found: {len(dicom_files)}")

    # --------------------------------------------------------
    # CSV files
    # --------------------------------------------------------

    csv_files = list(RSNA_DIR.rglob("*.csv"))

    print(f"CSV files found: {len(csv_files)}")

    if csv_files:
        print("\nCSV files:")

        for csv_file in csv_files:
            print(f"  {csv_file.relative_to(RSNA_DIR)}")

    # --------------------------------------------------------
    # Find label CSV
    # --------------------------------------------------------

    label_file = None

    for csv_file in csv_files:
        if (
            "label" in csv_file.name.lower()
            or "stage_2_train_labels" in csv_file.name.lower()
        ):
            label_file = csv_file
            break

    if label_file is None:
        print("\n[WARNING] Could not identify the RSNA label CSV.")

        return {
            "dicom_count": len(dicom_files),
            "csv_files": csv_files,
        }

    # --------------------------------------------------------
    # Read labels
    # --------------------------------------------------------

    try:
        df = pd.read_csv(label_file)

    except Exception as error:
        print(f"\n[ERROR] Could not read RSNA CSV:\n{error}")

        return {
            "dicom_count": len(dicom_files),
            "csv_files": csv_files,
        }

    print(f"\nLabel file:\n  {label_file.name}")

    print(f"\nLabel rows: {len(df)}")

    print(f"CSV columns:")

    for column in df.columns:
        print(f"  {column}")

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = {
        "patientId",
        "Target",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        print(f"\n[WARNING] Missing expected columns: {sorted(missing_columns)}")

        return {
            "dicom_count": len(dicom_files),
            "csv_files": csv_files,
        }

    # --------------------------------------------------------
    # Patient count
    # --------------------------------------------------------

    unique_patients = df["patientId"].nunique()

    print(f"\nUnique patients: {unique_patients}")

    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------

    target_counts = df["Target"].value_counts().sort_index()

    print("\nTarget distribution:")

    for target, count in target_counts.items():
        print(f"  Target {target}: {count}")

    # --------------------------------------------------------
    # Patient-level target distribution
    # --------------------------------------------------------

    patient_targets = df.groupby("patientId")["Target"].max()

    patient_target_counts = patient_targets.value_counts().sort_index()

    print("\nPatient-level target distribution:")

    for target, count in patient_target_counts.items():
        print(f"  Target {target}: {count}")

    return {
        "dicom_count": len(dicom_files),
        "csv_files": csv_files,
        "label_file": label_file,
        "label_rows": len(df),
        "unique_patients": unique_patients,
        "target_counts": target_counts,
        "patient_target_counts": (patient_target_counts),
    }


# ============================================================
# SAVE REPORT
# ============================================================


def save_report(
    kermany_result,
    rsna_result,
):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write("PNEUMONIA DATASET INSPECTION REPORT\n")

        file.write("=" * 70 + "\n\n")

        # ----------------------------------------------------
        # Kermany
        # ----------------------------------------------------

        file.write("KERMANY DATASET\n")

        file.write("-" * 50 + "\n")

        file.write(f"Total images: {kermany_result['total']}\n\n")

        file.write("Split distribution:\n")

        for split in [
            "train",
            "val",
            "test",
        ]:
            file.write(f"  {split}: {kermany_result['split_counts'].get(split, 0)}\n")

        file.write("\nClass distribution:\n")

        for label in [
            "NORMAL",
            "PNEUMONIA",
        ]:
            file.write(f"  {label}: {kermany_result['class_counts'].get(label, 0)}\n")

        file.write("\nSplit/class distribution:\n")

        for split in [
            "train",
            "val",
            "test",
        ]:
            file.write(f"\n  {split.upper()}\n")

            for label in [
                "NORMAL",
                "PNEUMONIA",
            ]:
                file.write(
                    f"    {label}: "
                    f"{kermany_result['split_class_counts'].get((split, label), 0)}\n"
                )

        file.write(f"\nCorrupted images: {len(kermany_result['corrupted'])}\n")

        file.write(f"Unknown split: {len(kermany_result['unknown_split'])}\n")

        file.write(f"Unknown class: {len(kermany_result['unknown_class'])}\n")

        # ----------------------------------------------------
        # RSNA
        # ----------------------------------------------------

        file.write("\n\nRSNA DATASET\n")

        file.write("-" * 50 + "\n")

        file.write(f"DICOM files: {rsna_result.get('dicom_count', 0)}\n")

        file.write(f"Label rows: {rsna_result.get('label_rows', 0)}\n")

        file.write(f"Unique patients: {rsna_result.get('unique_patients', 0)}\n")

        file.write("\nInspection complete.\n")

    print(f"\nInspection report saved to:\n{REPORT_FILE}")


def is_valid_image_candidate(path: Path) -> bool:
    """Return True only for actual image candidates."""
    if not path.is_file():
        return False

    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        return False

    # Ignore macOS metadata
    if "__MACOSX" in path.parts:
        return False

    # Ignore AppleDouble files such as ._IM-0001.jpeg
    if path.name.startswith("._"):
        return False
    return True


# ============================================================
# MAIN
# ============================================================


def main():

    print("\n")
    print("=" * 70)
    print("        PNEUMONIA DATASET INSPECTION")
    print("=" * 70)

    print(f"\nProject root:\n{PROJECT_ROOT}")

    kermany_result = inspect_kermany()

    rsna_result = inspect_rsna()

    save_report(
        kermany_result,
        rsna_result,
    )

    print("\n")
    print("=" * 70)
    print("        INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
