from pathlib import Path

import pandas as pd
import pydicom
import numpy as np
from PIL import Image


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_RSNA = PROJECT_ROOT / "data" / "raw" / "rsna"

IMAGE_DIR = RAW_RSNA / "stage_2_train_images"

SELECTION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rsna_selection"
    / "selected_rsna.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rsna_selected"
)


# --------------------------------------------------
# Read selected patients
# --------------------------------------------------

def load_selected_images():
    if not SELECTION_FILE.exists():
        raise FileNotFoundError(
            f"Selection file not found:\n"
            f"{SELECTION_FILE}\n\n"
            "Run select_rsna.py first."
        )

    df = pd.read_csv(SELECTION_FILE)

    required_columns = {
        "patientId",
        "label",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    return df


# --------------------------------------------------
# DICOM → PIL image
# --------------------------------------------------

def read_dicom(dicom_path: Path) -> Image.Image:

    dicom = pydicom.dcmread(dicom_path)

    pixels = dicom.pixel_array.astype(
        np.float32
    )

    # Handle MONOCHROME1 images.
    if getattr(
        dicom,
        "PhotometricInterpretation",
        ""
    ) == "MONOCHROME1":

        pixels = pixels.max() - pixels

    # Normalize pixel intensity to 0–255.
    minimum = pixels.min()
    maximum = pixels.max()

    if maximum > minimum:

        pixels = (
            (pixels - minimum)
            / (maximum - minimum)
            * 255.0
        )

    else:

        pixels[:] = 0

    pixels = np.clip(
        pixels,
        0,
        255,
    ).astype(np.uint8)

    return Image.fromarray(
        pixels,
        mode="L",
    )


# --------------------------------------------------
# Convert one patient
# --------------------------------------------------

def convert_image(
    patient_id: str,
    label: str,
):

    dicom_path = (
        IMAGE_DIR
        / f"{patient_id}.dcm"
    )

    if not dicom_path.exists():
        raise FileNotFoundError(
            f"DICOM not found: {dicom_path}"
        )

    image = read_dicom(
        dicom_path
    )

    output_dir = (
        OUTPUT_DIR / label
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"rsna_{patient_id}.png"
    )
    if output_path.exists():
        return output_path

    image.save(
        output_path,
        format="PNG",
    )

    return output_path


# --------------------------------------------------
# Convert selected dataset
# --------------------------------------------------

def convert_selected_rsna():

    print("=" * 60)
    print("RSNA DICOM → PNG CONVERSION")
    print("=" * 60)

    selected = load_selected_images()

    print(
        f"\nSelected images: "
        f"{len(selected)}"
    )

    successful = []
    failed = []

    for index, row in selected.iterrows():

        patient_id = str(
            row["patientId"]
        )

        label = str(
            row["label"]
        )

        try:

            output_path = convert_image(
                patient_id,
                label,
            )

            successful.append(
                output_path
            )

        except Exception as error:

            failed.append(
                {
                    "patientId": patient_id,
                    "label": label,
                    "error": str(error),
                }
            )

        completed = index + 1

        if (
            completed % 100 == 0
            or completed == len(selected)
        ):

            print(
                f"Processed "
                f"{completed}/{len(selected)}"
            )

    # --------------------------------------------------
    # Save conversion report
    # --------------------------------------------------

    report_dir = (
        PROJECT_ROOT
        / "results"
        / "logs"
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_file = (
        report_dir
        / "rsna_conversion_report.csv"
    )

    if failed:

        pd.DataFrame(
            failed
        ).to_csv(
            report_file,
            index=False,
        )

    else:

        pd.DataFrame(
            columns=[
                "patientId",
                "label",
                "error",
            ]
        ).to_csv(
            report_file,
            index=False,
        )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)

    print(
        f"Successful: {len(successful)}"
    )

    print(
        f"Failed:     {len(failed)}"
    )
    print("\nOutput class distribution:")

    for label in [
        "NON_PNEUMONIA",
        "PNEUMONIA",
    ]:
        label_dir = OUTPUT_DIR / label

        count = (
            len(list(label_dir.glob("*.png")))
            if label_dir.exists()
            else 0
        )

        print(
            f"  {label}: {count}"
        )

    print(
        f"\nOutput directory:\n"
        f"{OUTPUT_DIR}"
    )

    print(
        f"\nFailure report:\n"
        f"{report_file}"
    )

    return {
        "successful": successful,
        "failed": failed,
    }


if __name__ == "__main__":
    convert_selected_rsna()