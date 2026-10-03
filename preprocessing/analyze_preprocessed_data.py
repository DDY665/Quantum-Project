from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageStat, ImageDraw

import matplotlib.pyplot as plt


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_ROOT = (
    PROJECT_ROOT / "data" / "processed"
)

RESULTS_ROOT = (
    PROJECT_ROOT / "results"
)

EDA_DIR = (
    RESULTS_ROOT / "eda"
)

LOG_DIR = (
    RESULTS_ROOT / "logs"
)


# ============================================================
# Configuration
# ============================================================

DATASET_NAME = "dataset_B"

SAMPLE_COUNT_PER_CLASS = 8

BLANK_STD_THRESHOLD = 5.0

NEAR_BLANK_MEAN_THRESHOLD = 5.0

RANDOM_SEED = 42


# ============================================================
# Supported image extensions
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# Load split
# ============================================================

def load_split(
    split: str,
) -> pd.DataFrame:

    split_file = (
        PROCESSED_ROOT
        / DATASET_NAME
        / "splits"
        / f"{split}.csv"
    )

    if not split_file.exists():

        raise FileNotFoundError(
            f"Split file not found:\n"
            f"{split_file}\n\n"
            "Run prepare_dataset.py first."
        )

    return pd.read_csv(
        split_file
    )


# ============================================================
# Basic manifest validation
# ============================================================

def validate_manifest(
    dataframe: pd.DataFrame,
    split: str,
):

    required_columns = {
        "path",
        "label",
        "source",
    }

    missing = (
        required_columns
        - set(dataframe.columns)
    )

    if missing:

        raise ValueError(
            f"{split}: missing columns: "
            f"{sorted(missing)}"
        )

    if dataframe.empty:

        raise ValueError(
            f"{split} manifest is empty."
        )

    missing_paths = []

    for path in dataframe["path"]:

        if not Path(path).exists():

            missing_paths.append(
                path
            )

    if missing_paths:

        raise FileNotFoundError(
            f"{split}: "
            f"{len(missing_paths)} "
            f"missing image paths.\n"
            f"Example: {missing_paths[0]}"
        )


# ============================================================
# Image inspection
# ============================================================

def inspect_image(
    image_path: Path,
) -> dict:

    try:

        with Image.open(image_path) as image:

            image.load()

            array = np.asarray(
                image.convert("L"),
                dtype=np.float32,
            )

            width, height = image.size

            mean_value = float(
                array.mean()
            )

            std_value = float(
                array.std()
            )

            minimum = float(
                array.min()
            )

            maximum = float(
                array.max()
            )

            zero_fraction = float(
                np.mean(array <= 1)
            )

            high_fraction = float(
                np.mean(array >= 254)
            )

            return {
                "valid": True,
                "width": width,
                "height": height,
                "mean": mean_value,
                "std": std_value,
                "minimum": minimum,
                "maximum": maximum,
                "zero_fraction": zero_fraction,
                "high_fraction": high_fraction,
                "blank": (
                    std_value
                    <= BLANK_STD_THRESHOLD
                ),
                "near_blank": (
                    mean_value
                    <= NEAR_BLANK_MEAN_THRESHOLD
                ),
                "error": "",
            }

    except Exception as error:

        return {
            "valid": False,
            "width": None,
            "height": None,
            "mean": None,
            "std": None,
            "minimum": None,
            "maximum": None,
            "zero_fraction": None,
            "high_fraction": None,
            "blank": False,
            "near_blank": False,
            "error": str(error),
        }


# ============================================================
# Inspect split images
# ============================================================

def inspect_split(
    dataframe: pd.DataFrame,
    split: str,
) -> pd.DataFrame:

    print(
        f"\nInspecting {split.upper()} images..."
    )

    records = []

    for index, row in dataframe.iterrows():

        image_path = Path(
            row["path"]
        )

        result = inspect_image(
            image_path
        )

        result.update(
            {
                "path": str(image_path),
                "label": row["label"],
                "source": row["source"],
                "split": split,
            }
        )

        records.append(
            result
        )

        completed = index + 1

        if (
            completed % 1000 == 0
            or completed == len(dataframe)
        ):

            print(
                f"  Inspected "
                f"{completed}/{len(dataframe)}"
            )

    return pd.DataFrame(
        records
    )


# ============================================================
# Print dataset summary
# ============================================================

def print_dataset_summary(
    dataframe: pd.DataFrame,
):

    print("\n" + "=" * 60)
    print(
        f"{DATASET_NAME.upper()} DATASET SUMMARY"
    )
    print("=" * 60)

    print(
        f"\nTotal images: "
        f"{len(dataframe)}"
    )

    print("\nSplit distribution:")

    print(
        dataframe["split"]
        .value_counts()
        .sort_index()
        .to_string()
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

    print(
        "\nSplit / class distribution:"
    )

    split_class = pd.crosstab(
        dataframe["split"],
        dataframe["label"],
    )

    print(
        split_class.to_string()
    )

    print(
        "\nSplit / source distribution:"
    )

    split_source = pd.crosstab(
        dataframe["split"],
        dataframe["source"],
    )

    print(
        split_source.to_string()
    )


# ============================================================
# Print image statistics
# ============================================================

def print_image_statistics(
    inspection_df: pd.DataFrame,
):

    print("\n" + "=" * 60)
    print("IMAGE QUALITY ANALYSIS")
    print("=" * 60)

    invalid = (
        ~inspection_df["valid"]
    ).sum()

    blank = (
        inspection_df["blank"]
    ).sum()

    near_blank = (
        inspection_df["near_blank"]
    ).sum()

    print(
        f"\nInvalid/unreadable images: "
        f"{invalid}"
    )

    print(
        f"Blank images: "
        f"{blank}"
    )

    print(
        f"Near-blank images: "
        f"{near_blank}"
    )

    valid_df = inspection_df[
        inspection_df["valid"]
    ]

    print("\nOriginal image dimensions:")

    dimensions = (
        valid_df
        .groupby(
            ["width", "height"]
        )
        .size()
        .sort_values(
            ascending=False
        )
        .head(15)
    )

    for (
        (width, height),
        count,
    ) in dimensions.items():

        print(
            f"  {width} x {height}: "
            f"{count}"
        )

    print("\nPixel statistics:")

    print(
        valid_df[
            [
                "mean",
                "std",
                "minimum",
                "maximum",
            ]
        ]
        .describe()
        .round(3)
        .to_string()
    )


# ============================================================
# Class/source quality analysis
# ============================================================

def print_group_statistics(
    inspection_df: pd.DataFrame,
):

    valid_df = inspection_df[
        inspection_df["valid"]
    ]

    print(
        "\nImage statistics by class:"
    )

    class_stats = (
        valid_df
        .groupby("label")
        [
            [
                "mean",
                "std",
            ]
        ]
        .mean()
        .round(3)
    )

    print(
        class_stats.to_string()
    )

    print(
        "\nImage statistics by source:"
    )

    source_stats = (
        valid_df
        .groupby("source")
        [
            [
                "mean",
                "std",
            ]
        ]
        .mean()
        .round(3)
    )

    print(
        source_stats.to_string()
    )


# ============================================================
# Save inspection report
# ============================================================

def save_report(
    inspection_df: pd.DataFrame,
):

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_file = (
        LOG_DIR
        / f"{DATASET_NAME}_image_quality_report.csv"
    )

    inspection_df.to_csv(
        report_file,
        index=False,
    )

    print(
        f"\nDetailed image report saved to:\n"
        f"{report_file}"
    )


# ============================================================
# Create intensity histogram
# ============================================================

def create_intensity_histogram(
    inspection_df: pd.DataFrame,
):

    valid_df = inspection_df[
        inspection_df["valid"]
    ]

    EDA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        valid_df["mean"],
        bins=50,
    )

    plt.xlabel(
        "Mean grayscale intensity"
    )

    plt.ylabel(
        "Number of images"
    )

    plt.title(
        f"{DATASET_NAME} - "
        "Image Mean Intensity Distribution"
    )

    plt.tight_layout()

    output_file = (
        EDA_DIR
        / f"{DATASET_NAME}_intensity_distribution.png"
    )

    plt.savefig(
        output_file,
        dpi=150,
    )

    plt.close()

    print(
        f"Intensity histogram saved to:\n"
        f"{output_file}"
    )


# ============================================================
# Create class distribution plot
# ============================================================

def create_class_distribution_plot(
    dataframe: pd.DataFrame,
):

    EDA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    counts = (
        dataframe["label"]
        .value_counts()
    )

    plt.figure(
        figsize=(8, 6)
    )

    counts.plot(
        kind="bar"
    )

    plt.xlabel(
        "Class"
    )

    plt.ylabel(
        "Number of images"
    )

    plt.title(
        f"{DATASET_NAME} - "
        "Class Distribution"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    output_file = (
        EDA_DIR
        / f"{DATASET_NAME}_class_distribution.png"
    )

    plt.savefig(
        output_file,
        dpi=150,
    )

    plt.close()

    print(
        f"Class distribution plot saved to:\n"
        f"{output_file}"
    )


# ============================================================
# Create sample contact sheet
# ============================================================

def create_contact_sheet(
    dataframe: pd.DataFrame,
    label: str,
):

    samples = dataframe[
        dataframe["label"] == label
    ].sample(
        n=min(
            SAMPLE_COUNT_PER_CLASS,
            len(
                dataframe[
                    dataframe["label"]
                    == label
                ]
            ),
        ),
        random_state=RANDOM_SEED,
    )

    if samples.empty:

        return

    thumbnail_size = 180

    columns = 4

    rows = int(
        np.ceil(
            len(samples)
            / columns
        )
    )

    sheet = Image.new(
        "RGB",
        (
            columns
            * thumbnail_size,
            rows
            * thumbnail_size,
        ),
        "white",
    )

    draw = ImageDraw.Draw(
        sheet
    )

    for index, (_, row) in enumerate(
        samples.iterrows()
    ):

        image_path = Path(
            row["path"]
        )

        try:

            with Image.open(
                image_path
            ) as image:

                image = image.convert(
                    "L"
                )

                image.thumbnail(
                    (
                        thumbnail_size - 10,
                        thumbnail_size - 30,
                    )
                )

                x = (
                    index % columns
                ) * thumbnail_size

                y = (
                    index // columns
                ) * thumbnail_size

                offset_x = (
                    thumbnail_size
                    - image.width
                ) // 2

                offset_y = 5

                sheet.paste(
                    image.convert("RGB"),
                    (
                        x + offset_x,
                        y + offset_y,
                    ),
                )

                filename = (
                    image_path.name[
                        :18
                    ]
                )

                draw.text(
                    (
                        x + 5,
                        y + thumbnail_size - 22,
                    ),
                    filename,
                    fill="black",
                )

        except Exception:

            continue

    EDA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_label = label.lower()

    output_file = (
        EDA_DIR
        / f"{DATASET_NAME}_{safe_label}_samples.png"
    )

    sheet.save(
        output_file
    )

    print(
        f"Sample images saved to:\n"
        f"{output_file}"
    )


# ============================================================
# Main
# ============================================================

def analyze_dataset():

    print("=" * 60)
    print(
        "PREPROCESSING / EDA "
        "QUALITY ANALYSIS"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Load manifests
    # --------------------------------------------------------

    train_df = load_split(
        "train"
    )

    val_df = load_split(
        "val"
    )

    test_df = load_split(
        "test"
    )

    validate_manifest(
        train_df,
        "TRAIN",
    )

    validate_manifest(
        val_df,
        "VAL",
    )

    validate_manifest(
        test_df,
        "TEST",
    )

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    combined_df = pd.concat(
        [
            train_df,
            val_df,
            test_df,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Dataset summary
    # --------------------------------------------------------

    print_dataset_summary(
        combined_df
    )

    # --------------------------------------------------------
    # Inspect images
    # --------------------------------------------------------

    inspection_frames = []

    for split_df, split_name in [
        (train_df, "train"),
        (val_df, "val"),
        (test_df, "test"),
    ]:

        result = inspect_split(
            split_df,
            split_name,
        )

        inspection_frames.append(
            result
        )

    inspection_df = pd.concat(
        inspection_frames,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Image quality
    # --------------------------------------------------------

    print_image_statistics(
        inspection_df
    )

    print_group_statistics(
        inspection_df
    )

    # --------------------------------------------------------
    # Save detailed report
    # --------------------------------------------------------

    save_report(
        inspection_df
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    create_intensity_histogram(
        inspection_df
    )

    create_class_distribution_plot(
        combined_df
    )

    # --------------------------------------------------------
    # Sample images
    # --------------------------------------------------------

    for label in [
        "NON_PNEUMONIA",
        "PNEUMONIA",
    ]:

        create_contact_sheet(
            combined_df,
            label,
        )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print(
        "PREPROCESSING QUALITY ANALYSIS COMPLETE"
    )
    print("=" * 60)

    print(
        f"\nEDA output directory:\n"
        f"{EDA_DIR}"
    )

    print(
        f"\nDetailed report:\n"
        f"{LOG_DIR / f'{DATASET_NAME}_image_quality_report.csv'}"
    )


if __name__ == "__main__":

    analyze_dataset()