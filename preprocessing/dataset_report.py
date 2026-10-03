from pathlib import Path
import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"

DATASET_A = PROCESSED_ROOT / "dataset_A" / "manifest.csv"
DATASET_B = PROCESSED_ROOT / "dataset_B" / "manifest.csv"

RESULTS_DIR = PROJECT_ROOT / "results" / "logs"
REPORT_FILE = RESULTS_DIR / "dataset_report.txt"


# --------------------------------------------------
# Load manifest
# --------------------------------------------------

def load_manifest(path: Path) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{path}"
        )

    df = pd.read_csv(path)

    required_columns = {
        "path",
        "label",
        "source",
        "dataset",
    }

    missing = (
        required_columns - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing columns in {path}: "
            f"{sorted(missing)}"
        )

    return df


# --------------------------------------------------
# Verify files
# --------------------------------------------------

def check_files_exist(
    df: pd.DataFrame,
):

    missing = []

    for path in df["path"]:

        if not Path(path).exists():
            missing.append(path)

    return missing


# --------------------------------------------------
# Check duplicate paths
# --------------------------------------------------

def check_duplicate_paths(
    df: pd.DataFrame,
):

    return (
        df["path"]
        .duplicated()
        .sum()
    )


# --------------------------------------------------
# Dataset summary
# --------------------------------------------------

def summarize_dataset(
    name: str,
    df: pd.DataFrame,
):

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(
        f"Total images: {len(df)}"
    )

    print("\nLabels:")
    print(
        df["label"]
        .value_counts()
        .to_string()
    )

    print("\nSources:")
    print(
        df["source"]
        .value_counts()
        .to_string()
    )

    print("\nDataset field:")
    print(
        df["dataset"]
        .value_counts()
        .to_string()
    )


# --------------------------------------------------
# Compare Dataset A and B
# --------------------------------------------------

def compare_datasets(
    dataset_a: pd.DataFrame,
    dataset_b: pd.DataFrame,
):

    print("\n" + "=" * 60)
    print("DATASET COMPARISON")
    print("=" * 60)

    print(
        f"Dataset A images: "
        f"{len(dataset_a)}"
    )

    print(
        f"Dataset B images: "
        f"{len(dataset_b)}"
    )

    print(
        f"\nAdditional images in B: "
        f"{len(dataset_b) - len(dataset_a)}"
    )

    print("\nAdditional source:")
    print(
        dataset_b[
            dataset_b["source"] == "RSNA"
        ]["label"]
        .value_counts()
        .to_string()
    )


# --------------------------------------------------
# Generate report
# --------------------------------------------------

def generate_report(
    dataset_a: pd.DataFrame,
    dataset_b: pd.DataFrame,
    missing_a: list,
    missing_b: list,
):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "PNEUMONIA DATASET VERIFICATION REPORT\n"
        )

        file.write("=" * 60 + "\n\n")

        # ------------------------------------------
        # Dataset A
        # ------------------------------------------

        file.write("DATASET A\n")
        file.write("-" * 40 + "\n")

        file.write(
            f"Total images: "
            f"{len(dataset_a)}\n"
        )

        file.write("\nLabels:\n")

        for label, count in (
            dataset_a["label"]
            .value_counts()
            .items()
        ):

            file.write(
                f"  {label}: {count}\n"
            )

        file.write("\nSources:\n")

        for source, count in (
            dataset_a["source"]
            .value_counts()
            .items()
        ):

            file.write(
                f"  {source}: {count}\n"
            )

        file.write(
            f"\nMissing files: "
            f"{len(missing_a)}\n"
        )

        # ------------------------------------------
        # Dataset B
        # ------------------------------------------

        file.write("\n\nDATASET B\n")
        file.write("-" * 40 + "\n")

        file.write(
            f"Total images: "
            f"{len(dataset_b)}\n"
        )

        file.write("\nLabels:\n")

        for label, count in (
            dataset_b["label"]
            .value_counts()
            .items()
        ):

            file.write(
                f"  {label}: {count}\n"
            )

        file.write("\nSources:\n")

        for source, count in (
            dataset_b["source"]
            .value_counts()
            .items()
        ):

            file.write(
                f"  {source}: {count}\n"
            )

        file.write(
            f"\nMissing files: "
            f"{len(missing_b)}\n"
        )

        # ------------------------------------------
        # Final status
        # ------------------------------------------

        file.write(
            "\n\nFINAL STATUS\n"
        )

        if not missing_a and not missing_b:

            file.write(
                "All manifest files exist.\n"
            )

        else:

            file.write(
                "Some manifest files are missing.\n"
            )


# --------------------------------------------------
# Main verification
# --------------------------------------------------

def verify_datasets():

    print("=" * 60)
    print("DATASET VERIFICATION")
    print("=" * 60)

    dataset_a = load_manifest(
        DATASET_A
    )

    dataset_b = load_manifest(
        DATASET_B
    )

    # ------------------------------------------
    # Summaries
    # ------------------------------------------

    summarize_dataset(
        "DATASET A",
        dataset_a,
    )

    summarize_dataset(
        "DATASET B",
        dataset_b,
    )

    # ------------------------------------------
    # File existence
    # ------------------------------------------

    missing_a = check_files_exist(
        dataset_a
    )

    missing_b = check_files_exist(
        dataset_b
    )

    print(
        f"\nDataset A missing files: "
        f"{len(missing_a)}"
    )

    print(
        f"Dataset B missing files: "
        f"{len(missing_b)}"
    )

    # ------------------------------------------
    # Duplicate path check
    # ------------------------------------------

    duplicate_a = (
        check_duplicate_paths(dataset_a)
    )

    duplicate_b = (
        check_duplicate_paths(dataset_b)
    )

    print(
        f"\nDataset A duplicate paths: "
        f"{duplicate_a}"
    )

    print(
        f"Dataset B duplicate paths: "
        f"{duplicate_b}"
    )

    # ------------------------------------------
    # Comparison
    # ------------------------------------------

    compare_datasets(
        dataset_a,
        dataset_b,
    )

    # ------------------------------------------
    # Report
    # ------------------------------------------

    generate_report(
        dataset_a,
        dataset_b,
        missing_a,
        missing_b,
    )

    print(
        f"\nReport saved to:\n"
        f"{REPORT_FILE}"
    )

    print("\nVerification complete.")


if __name__ == "__main__":
    verify_datasets()