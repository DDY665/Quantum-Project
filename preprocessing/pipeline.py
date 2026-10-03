from pathlib import Path

from preprocessing.clean_images import (
    validate_dataset,
)

from preprocessing.check_duplicates import (
    check_duplicates,
)

from preprocessing.select_rsna import (
    select_rsna_subset,
)

from preprocessing.rsna_converter import (
    convert_selected_rsna,
)

from preprocessing.build_dataset import (
    build_datasets,
)

from preprocessing.dataset_report import (
    verify_datasets,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data"

RAW_ROOT = DATA_ROOT / "raw"

RESULTS_ROOT = PROJECT_ROOT / "results"

LOG_ROOT = RESULTS_ROOT / "logs"


# --------------------------------------------------
# Validate Kermany
# --------------------------------------------------

def validate_kermany():

    print("\n" + "=" * 60)
    print("STEP 1 — VALIDATE KERMANY")
    print("=" * 60)

    kermany_dir = (
        RAW_ROOT / "kermany"
    )

    report = (
        LOG_ROOT /
        "kermany_corrupted.txt"
    )

    return validate_dataset(
        kermany_dir,
        report,
    )


# --------------------------------------------------
# Check Kermany duplicates
# --------------------------------------------------

def check_kermany_duplicates():

    print("\n" + "=" * 60)
    print("STEP 2 — CHECK KERMANY DUPLICATES")
    print("=" * 60)

    kermany_dir = (
        RAW_ROOT / "kermany"
    )

    report = (
        LOG_ROOT /
        "kermany_duplicates.txt"
    )

    return check_duplicates(
        kermany_dir,
        report,
    )


# --------------------------------------------------
# Select RSNA
# --------------------------------------------------

def select_rsna():

    print("\n" + "=" * 60)
    print("STEP 3 — SELECT RSNA SUBSET")
    print("=" * 60)

    return select_rsna_subset()


# --------------------------------------------------
# Convert RSNA
# --------------------------------------------------

def convert_rsna():

    print("\n" + "=" * 60)
    print("STEP 4 — CONVERT RSNA DICOM")
    print("=" * 60)

    return convert_selected_rsna()


# --------------------------------------------------
# Build Dataset A and B
# --------------------------------------------------

def build():

    print("\n" + "=" * 60)
    print("STEP 5 — BUILD DATASETS")
    print("=" * 60)

    return build_datasets()


# --------------------------------------------------
# Verify
# --------------------------------------------------

def verify():

    print("\n" + "=" * 60)
    print("STEP 6 — VERIFY DATASETS")
    print("=" * 60)

    return verify_datasets()


# --------------------------------------------------
# Complete preprocessing pipeline
# --------------------------------------------------

def run_complete_pipeline():

    print("\n")
    print("=" * 70)
    print("       PNEUMONIA DATA PREPARATION PIPELINE")
    print("=" * 70)

    # Create log directory
    LOG_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------
    # 1. Validate Kermany
    # ----------------------------------------------

    validate_kermany()

    # ----------------------------------------------
    # 2. Check duplicates
    # ----------------------------------------------

    check_kermany_duplicates()

    # ----------------------------------------------
    # 3. Select RSNA
    # ----------------------------------------------

    select_rsna()

    # ----------------------------------------------
    # 4. Convert selected RSNA images
    # ----------------------------------------------

    convert_rsna()

    # ----------------------------------------------
    # 5. Build Dataset A/B
    # ----------------------------------------------

    build()

    # ----------------------------------------------
    # 6. Final verification
    # ----------------------------------------------

    verify()

    print("\n")
    print("=" * 70)
    print("       DATA PREPARATION COMPLETE")
    print("=" * 70)

    print(
        "\nDataset A → data/processed/dataset_A/"
    )

    print(
        "Dataset B → data/processed/dataset_B/"
    )

    print(
        "\nLabels:"
    )

    print(
        "  0 → NON_PNEUMONIA"
    )

    print(
        "  1 → PNEUMONIA"
    )

    print(
        "\nThe datasets are ready for the CNN stage."
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    run_complete_pipeline()


if __name__ == "__main__":
    main()