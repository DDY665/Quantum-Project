from pathlib import Path
from PIL import Image


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


def find_images(dataset_dir: Path) -> list[Path]:
    """Return all supported image files inside a dataset directory."""

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}"
        )

    return sorted(
        path
        for path in dataset_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def is_valid_image(image_path: Path) -> bool:
    """Check whether an image can be opened and fully loaded."""

    try:
        with Image.open(image_path) as image:
            image.verify()

        # Re-open after verify() and force actual loading.
        with Image.open(image_path) as image:
            image.load()

        return True

    except Exception:
        return False


def find_corrupted_images(
    dataset_dir: Path,
) -> tuple[list[Path], list[Path]]:
    """
    Validate all images.

    Returns:
        valid_images
        corrupted_images

    No files are deleted or modified.
    """

    image_files = find_images(dataset_dir)

    valid_images = []
    corrupted_images = []

    for image_path in image_files:

        if is_valid_image(image_path):
            valid_images.append(image_path)
        else:
            corrupted_images.append(image_path)

    return valid_images, corrupted_images


def save_corrupted_report(
    corrupted_images: list[Path],
    dataset_dir: Path,
    report_path: Path,
) -> None:
    """Save corrupted-image information to a text file."""

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write("CORRUPTED IMAGE REPORT\n")
        file.write("=" * 60 + "\n\n")

        file.write(
            f"Dataset: {dataset_dir}\n"
        )

        file.write(
            f"Corrupted images: "
            f"{len(corrupted_images)}\n\n"
        )

        for image_path in corrupted_images:

            try:
                relative_path = image_path.relative_to(
                    dataset_dir
                )
            except ValueError:
                relative_path = image_path

            file.write(
                f"{relative_path}\n"
            )


def validate_dataset(
    dataset_dir: Path,
    report_path: Path | None = None,
) -> dict:
    """
    Validate a dataset and return a summary.

    This function does not modify or delete data.
    """

    valid_images, corrupted_images = (
        find_corrupted_images(dataset_dir)
    )

    if report_path is not None:
        save_corrupted_report(
            corrupted_images,
            dataset_dir,
            report_path,
        )

    return {
        "dataset": str(dataset_dir),
        "total_images": (
            len(valid_images)
            + len(corrupted_images)
        ),
        "valid_images": len(valid_images),
        "corrupted_images": len(corrupted_images),
        "corrupted_files": [
            str(path)
            for path in corrupted_images
        ],
    }