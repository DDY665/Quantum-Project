from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_ROOT = PROJECT_ROOT / "data" / "raw"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"

KERMANY_DIR = RAW_ROOT / "kermany"

RSNA_SELECTED_DIR = (
    PROCESSED_ROOT / "rsna_selected"
)

DATASET_A_DIR = (
    PROCESSED_ROOT / "dataset_A"
)

DATASET_B_DIR = (
    PROCESSED_ROOT / "dataset_B"
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

RANDOM_SEED = 42


# --------------------------------------------------
# Find Kermany images
# --------------------------------------------------

def find_kermany_images():

    if not KERMANY_DIR.exists():
        raise FileNotFoundError(
            f"Kermany dataset not found:\n"
            f"{KERMANY_DIR}"
        )

    records = []

    for image_path in KERMANY_DIR.rglob("*"):

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
        }:
            continue

        # Ignore macOS metadata directories.
        if "__MACOSX" in image_path.parts:
            continue

        # Ignore AppleDouble metadata files.
        if image_path.name.startswith("._"):
            continue

        class_name = (
            image_path.parent.name.upper()
        )

        if class_name not in {
            "NORMAL",
            "PNEUMONIA",
        }:
            continue

        # Standardize Kermany NORMAL
        # to the project's final negative class.
        final_label = (
            "NON_PNEUMONIA"
            if class_name == "NORMAL"
            else "PNEUMONIA"
        )

        records.append(
            {
                "path": str(
                    image_path.resolve()
                ),
                "label": final_label,
                "source": "KERMANY",
                "original_label": class_name,
            }
        )

    return pd.DataFrame(records)


# --------------------------------------------------
# Find selected RSNA images
# --------------------------------------------------

def find_rsna_images():

    if not RSNA_SELECTED_DIR.exists():
        raise FileNotFoundError(
            f"Selected RSNA dataset not found:\n"
            f"{RSNA_SELECTED_DIR}"
        )

    records = []

    for image_path in RSNA_SELECTED_DIR.rglob("*.png"):

        if not image_path.is_file():
            continue

        class_name = (
            image_path.parent.name.upper()
        )

        if class_name not in {
            "NON_PNEUMONIA",
            "PNEUMONIA",
        }:
            continue

        records.append(
            {
                "path": str(
                    image_path.resolve()
                ),
                "label": class_name,
                "source": "RSNA",
                "original_label": class_name,
            }
        )

    return pd.DataFrame(records)

# --------------------------------------------------
# Create Dataset A
# --------------------------------------------------

def create_dataset_a(
    kermany_df: pd.DataFrame,
):

    output_dir = DATASET_A_DIR
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = kermany_df.copy()

    manifest["dataset"] = "DATASET_A"

    output_file = (
        output_dir / "manifest.csv"
    )

    manifest.to_csv(
        output_file,
        index=False,
    )

    print(
        f"\nDataset A created:"
        f"\n{output_file}"
    )

    return manifest


# --------------------------------------------------
# Create Dataset B
# --------------------------------------------------

def create_dataset_b(
    kermany_df: pd.DataFrame,
    rsna_df: pd.DataFrame,
):

    output_dir = DATASET_B_DIR

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined = pd.concat(
        [
            kermany_df,
            rsna_df,
        ],
        ignore_index=True,
    )

    combined["dataset"] = "DATASET_B"

    output_file = (
        output_dir / "manifest.csv"
    )

    combined.to_csv(
        output_file,
        index=False,
    )

    print(
        f"\nDataset B created:"
        f"\n{output_file}"
    )

    return combined


# --------------------------------------------------
# Dataset summary
# --------------------------------------------------

def print_summary(
    name: str,
    dataframe: pd.DataFrame,
):

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(
        f"Total images: "
        f"{len(dataframe)}"
    )

    print("\nClass distribution:")

    print(
        dataframe["label"]
        .value_counts()
        .to_string()
    )

    print("\nSource distribution:")

    print(
        dataframe["source"]
        .value_counts()
        .to_string()
    )


# --------------------------------------------------
# Main builder
# --------------------------------------------------

def build_datasets():

    print("=" * 60)
    print("DATASET CONSTRUCTION")
    print("=" * 60)

    # ----------------------------------------------
    # Kermany
    # ----------------------------------------------

    print("\nScanning Kermany...")

    kermany_df = (
        find_kermany_images()
    )

    print(
        f"Kermany images found: "
        f"{len(kermany_df)}"
    )

    # ----------------------------------------------
    # RSNA
    # ----------------------------------------------

    print("\nScanning selected RSNA...")

    rsna_df = find_rsna_images()

    print(
        f"RSNA images found: "
        f"{len(rsna_df)}"
    )

    # ----------------------------------------------
    # Dataset A
    # ----------------------------------------------

    dataset_a = create_dataset_a(
        kermany_df
    )

    # ----------------------------------------------
    # Dataset B
    # ----------------------------------------------

    dataset_b = create_dataset_b(
        kermany_df,
        rsna_df,
    )

    # ----------------------------------------------
    # Print summaries
    # ----------------------------------------------

    print_summary(
        "DATASET A",
        dataset_a,
    )

    print_summary(
        "DATASET B",
        dataset_b,
    )

    print("\n" + "=" * 60)
    print("DATASET CONSTRUCTION COMPLETE")
    print("=" * 60)

    return dataset_a, dataset_b


if __name__ == "__main__":
    build_datasets()