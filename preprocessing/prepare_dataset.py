from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_ROOT = (
    PROJECT_ROOT / "data" / "processed"
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

TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15


# --------------------------------------------------
# Validate configuration
# --------------------------------------------------

def validate_configuration():

    total = (
        TRAIN_SIZE
        + VAL_SIZE
        + TEST_SIZE
    )

    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            "Train/validation/test ratios "
            "must sum to 1.0."
        )


# --------------------------------------------------
# Load manifest
# --------------------------------------------------

def load_manifest(
    dataset_dir: Path,
) -> pd.DataFrame:

    manifest_file = (
        dataset_dir / "manifest.csv"
    )

    if not manifest_file.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n"
            f"{manifest_file}"
        )

    df = pd.read_csv(
        manifest_file
    )

    required_columns = {
        "path",
        "label",
        "source",
        "original_label",
        "dataset",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns in "
            f"{manifest_file}:\n"
            f"{sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError(
            f"Manifest is empty:\n"
            f"{manifest_file}"
        )

    return df


# --------------------------------------------------
# Validate image paths
# --------------------------------------------------

def validate_image_paths(
    df: pd.DataFrame,
    dataset_name: str,
):

    missing_paths = []

    for image_path in df["path"]:

        if not Path(image_path).exists():
            missing_paths.append(
                image_path
            )

    if missing_paths:

        print(
            f"\nERROR: "
            f"{len(missing_paths)} missing "
            f"image files in {dataset_name}."
        )

        for path in missing_paths[:10]:
            print(f"  {path}")

        if len(missing_paths) > 10:
            print(
                f"  ... and "
                f"{len(missing_paths) - 10} more"
            )

        raise FileNotFoundError(
            f"Missing image files detected "
            f"in {dataset_name}."
        )

    print(
        f"All {len(df)} image paths exist."
    )


# --------------------------------------------------
# Validate manifest
# --------------------------------------------------

def validate_manifest(
    df: pd.DataFrame,
    dataset_name: str,
):

    print(
        f"\nValidating {dataset_name}..."
    )

    # Check duplicate paths.

    duplicate_paths = (
        df["path"].duplicated().sum()
    )

    if duplicate_paths > 0:

        raise ValueError(
            f"{dataset_name} contains "
            f"{duplicate_paths} duplicate paths."
        )

    # Check labels.

    valid_labels = {
        "NON_PNEUMONIA",
        "PNEUMONIA",
    }

    invalid_labels = (
        set(df["label"].unique())
        - valid_labels
    )

    if invalid_labels:

        raise ValueError(
            f"Invalid labels in "
            f"{dataset_name}: "
            f"{sorted(invalid_labels)}"
        )

    # Check source values.

    valid_sources = {
        "KERMANY",
        "RSNA",
    }

    invalid_sources = (
        set(df["source"].unique())
        - valid_sources
    )

    if invalid_sources:

        raise ValueError(
            f"Invalid sources in "
            f"{dataset_name}: "
            f"{sorted(invalid_sources)}"
        )

    # Check image paths.

    validate_image_paths(
        df,
        dataset_name,
    )

    print(
        f"{dataset_name} manifest validation passed."
    )


# --------------------------------------------------
# Create stratified split
# --------------------------------------------------

def create_splits(
    df: pd.DataFrame,
):

    # First split:
    #
    # 70% train
    # 30% temporary
    #
    # The temporary set will later become:
    # 15% validation
    # 15% test.

    train_df, temporary_df = (
        train_test_split(
            df,
            test_size=(
                VAL_SIZE + TEST_SIZE
            ),
            stratify=df["label"],
            random_state=RANDOM_SEED,
        )
    )

    # Split temporary data equally.
    #
    # 15 / 30 = 0.5

    val_df, test_df = (
        train_test_split(
            temporary_df,
            test_size=0.5,
            stratify=temporary_df["label"],
            random_state=RANDOM_SEED,
        )
    )

    # Reset indexes.

    train_df = (
        train_df
        .reset_index(drop=True)
    )

    val_df = (
        val_df
        .reset_index(drop=True)
    )

    test_df = (
        test_df
        .reset_index(drop=True)
    )

    return (
        train_df,
        val_df,
        test_df,
    )


# --------------------------------------------------
# Check split leakage
# --------------------------------------------------

def check_split_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
):

    train_paths = set(
        train_df["path"]
    )

    val_paths = set(
        val_df["path"]
    )

    test_paths = set(
        test_df["path"]
    )

    train_val = (
        train_paths & val_paths
    )

    train_test = (
        train_paths & test_paths
    )

    val_test = (
        val_paths & test_paths
    )

    if train_val:
        raise ValueError(
            f"Train/validation leakage: "
            f"{len(train_val)} paths."
        )

    if train_test:
        raise ValueError(
            f"Train/test leakage: "
            f"{len(train_test)} paths."
        )

    if val_test:
        raise ValueError(
            f"Validation/test leakage: "
            f"{len(val_test)} paths."
        )

    print(
        "Split leakage check: PASSED"
    )


# --------------------------------------------------
# Save splits
# --------------------------------------------------

def save_splits(
    dataset_dir: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
):

    split_dir = (
        dataset_dir / "splits"
    )

    split_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_file = (
        split_dir / "train.csv"
    )

    val_file = (
        split_dir / "val.csv"
    )

    test_file = (
        split_dir / "test.csv"
    )

    train_df.to_csv(
        train_file,
        index=False,
    )

    val_df.to_csv(
        val_file,
        index=False,
    )

    test_df.to_csv(
        test_file,
        index=False,
    )

    print("\nSplit files saved:")

    print(
        f"  TRAIN: {train_file}"
    )

    print(
        f"  VAL:   {val_file}"
    )

    print(
        f"  TEST:  {test_file}"
    )


# --------------------------------------------------
# Print split summary
# --------------------------------------------------

def print_split_summary(
    dataset_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
):

    print("\n" + "=" * 60)
    print(
        f"{dataset_name} SPLIT SUMMARY"
    )
    print("=" * 60)

    splits = {
        "TRAIN": train_df,
        "VAL": val_df,
        "TEST": test_df,
    }

    for split_name, split_df in splits.items():

        print(
            f"\n{split_name}"
        )

        print(
            f"  Total: {len(split_df)}"
        )

        print("  Classes:")

        class_counts = (
            split_df["label"]
            .value_counts()
        )

        for label in [
            "NON_PNEUMONIA",
            "PNEUMONIA",
        ]:

            print(
                f"    {label}: "
                f"{class_counts.get(label, 0)}"
            )

        print("  Sources:")

        source_counts = (
            split_df["source"]
            .value_counts()
        )

        for source, count in (
            source_counts.items()
        ):

            print(
                f"    {source}: {count}"
            )


# --------------------------------------------------
# Process one dataset
# --------------------------------------------------

def process_dataset(
    dataset_dir: Path,
    dataset_name: str,
):

    print("\n" + "=" * 60)
    print(
        f"PROCESSING {dataset_name}"
    )
    print("=" * 60)

    # Load manifest.

    df = load_manifest(
        dataset_dir
    )

    print(
        f"\nTotal images: {len(df)}"
    )

    # Validate.

    validate_manifest(
        df,
        dataset_name,
    )

    # Create splits.

    (
        train_df,
        val_df,
        test_df,
    ) = create_splits(df)

    # Check leakage.

    check_split_leakage(
        train_df,
        val_df,
        test_df,
    )

    # Verify total.

    total_split_images = (
        len(train_df)
        + len(val_df)
        + len(test_df)
    )

    if total_split_images != len(df):

        raise ValueError(
            f"Split count mismatch for "
            f"{dataset_name}: "
            f"{total_split_images} != {len(df)}"
        )

    # Save.

    save_splits(
        dataset_dir,
        train_df,
        val_df,
        test_df,
    )

    # Summary.

    print_split_summary(
        dataset_name,
        train_df,
        val_df,
        test_df,
    )

    return (
        train_df,
        val_df,
        test_df,
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def prepare_datasets():

    print("=" * 60)
    print("DATA PREPROCESSING - DATASET SPLITTING")
    print("=" * 60)

    validate_configuration()

    # Dataset A

    dataset_a = process_dataset(
        DATASET_A_DIR,
        "DATASET A",
    )

    # Dataset B

    dataset_b = process_dataset(
        DATASET_B_DIR,
        "DATASET B",
    )

    print("\n" + "=" * 60)
    print(
        "DATASET SPLITTING COMPLETE"
    )
    print("=" * 60)

    print(
        "\nRandom seed:"
        f" {RANDOM_SEED}"
    )

    print(
        "\nSplit ratio:"
        f" {TRAIN_SIZE:.0%} train /"
        f" {VAL_SIZE:.0%} val /"
        f" {TEST_SIZE:.0%} test"
    )

    return dataset_a, dataset_b


# --------------------------------------------------
# Command-line execution
# --------------------------------------------------

if __name__ == "__main__":

    prepare_datasets()