from pathlib import Path
from collections import Counter, defaultdict

import imagehash
from PIL import Image


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


def get_dataset_metadata(image_path: Path) -> tuple[str | None, str | None]:
    """
    Extract split and class from a Kermany image path.
    """

    valid_splits = {"train", "val", "test"}
    valid_classes = {"NORMAL", "PNEUMONIA"}

    split = None
    label = None

    for parent in image_path.parents:

        name = parent.name

        if name.lower() in valid_splits:
            split = name.lower()

        if name.upper() in valid_classes:
            label = name.upper()

    return split, label

def analyze_duplicate_groups(duplicate_groups: list[list[Path]]) -> dict:
    """
    Analyze duplicate groups by split and class.
    """

    cross_split_groups = []
    cross_class_groups = []

    split_distribution = Counter()
    class_distribution = Counter()

    for index, group in enumerate(
        duplicate_groups,
        start=1,
    ):

        splits = set()
        classes = set()

        for image_path in group:

            split, label = get_dataset_metadata(image_path)

            if split is not None:
                splits.add(split)

            if label is not None:
                classes.add(label)

        split_key = tuple(sorted(splits))
        class_key = tuple(sorted(classes))

        split_distribution[split_key] += 1
        class_distribution[class_key] += 1

        if len(splits) > 1:

            cross_split_groups.append(
                {
                    "group_number": index,
                    "paths": group,
                    "splits": splits,
                    "classes": classes,
                }
            )

        if len(classes) > 1:

            cross_class_groups.append(
                {
                    "group_number": index,
                    "paths": group,
                    "splits": splits,
                    "classes": classes,
                }
            )

    return {
        "cross_split_groups": cross_split_groups,
        "cross_class_groups": cross_class_groups,
        "split_distribution": split_distribution,
        "class_distribution": class_distribution,
    }

def find_images(dataset_dir: Path) -> list[Path]:
    """Find valid image files, excluding macOS metadata."""

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}"
        )

    return sorted(
        path
        for path in dataset_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and "__MACOSX" not in path.parts
        and not path.name.startswith("._")
    )


def calculate_hash(
    image_path: Path,
):
    """Calculate perceptual hash for an image."""

    try:
        with Image.open(image_path) as image:
            return imagehash.phash(image)

    except Exception:
        return None


def find_duplicates(
    dataset_dir: Path,
) -> dict:
    """
    Find images with identical perceptual hashes.

    This version detects exact perceptual-hash matches.
    Near-duplicate distance comparison can be added later.
    """

    image_files = find_images(dataset_dir)

    hash_groups = defaultdict(list)

    failed_images = []

    for image_path in image_files:

        image_hash = calculate_hash(image_path)

        if image_hash is None:
            failed_images.append(image_path)
            continue

        hash_groups[str(image_hash)].append(
            image_path
        )

    duplicate_groups = [
        paths
        for paths in hash_groups.values()
        if len(paths) > 1
    ]
    
    duplicate_analysis = analyze_duplicate_groups(
        duplicate_groups
    )
    
    duplicate_group_sizes = Counter(
        len(group)
        for group in duplicate_groups
    )

    return {
        "total_images": len(image_files),
        "hashed_images": len(image_files) - len(failed_images),
        "failed_images": failed_images,
        "duplicate_groups": duplicate_groups,
        "duplicate_group_sizes": duplicate_group_sizes,
        "duplicate_analysis": duplicate_analysis,
    }


def save_duplicate_report(
    result: dict,
    dataset_dir: Path,
    report_path: Path,
) -> None:
    """Save duplicate information to a report."""

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write("DUPLICATE IMAGE REPORT\n")
        file.write("=" * 60 + "\n\n")

        file.write(
            f"Dataset: {dataset_dir}\n"
        )

        file.write(
            f"Total images: "
            f"{result['total_images']}\n"
        )

        file.write(
            f"Duplicate groups: "
            f"{len(result['duplicate_groups'])}\n\n"
        )

        for index, group in enumerate(
            result["duplicate_groups"],
            start=1,
        ):

            file.write(
                f"Duplicate Group {index}\n"
            )

            for image_path in group:

                try:
                    relative_path = (
                        image_path.relative_to(
                            dataset_dir
                        )
                    )
                except ValueError:
                    relative_path = image_path

                file.write(
                    f"  {relative_path}\n"
                )

            file.write("\n")

        if result["failed_images"]:

            file.write(
                "Images that could not be hashed:\n"
            )

            for image_path in result[
                "failed_images"
            ]:

                file.write(
                    f"  {image_path}\n"
                )


def check_duplicates(
    dataset_dir: Path,
    report_path: Path | None = None,
) -> dict:
    """
    Run duplicate detection.

    No files are deleted or modified.
    """

    result = find_duplicates(
        dataset_dir
    )

    if report_path is not None:
        save_duplicate_report(
            result,
            dataset_dir,
            report_path,
        )

    return result


if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    DATASET_DIR = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "kermany"
    )

    REPORT_PATH = (
        PROJECT_ROOT
        / "results"
        / "logs"
        / "duplicate_report.txt"
    )

    result = check_duplicates(
        dataset_dir=DATASET_DIR,
        report_path=REPORT_PATH,
    )

    print("=" * 60)
    print("KERMANY DUPLICATE ANALYSIS")
    print("=" * 60)

    print(
        f"\nTotal valid images: "
        f"{result['total_images']}"
    )

    print(
        f"Images successfully hashed: "
        f"{result['hashed_images']}"
    )

    print(
        f"Images that failed hashing: "
        f"{len(result['failed_images'])}"
    )

    print(
        f"Duplicate groups: "
        f"{len(result['duplicate_groups'])}"
    )
    print("\nDuplicate group sizes:")

    for size, count in sorted(
        result["duplicate_group_sizes"].items()
    ):
        print(
            f"  Groups containing {size} images: {count}"
        )
        
    analysis = result["duplicate_analysis"]

    print(
        "\nDuplicate groups crossing dataset splits: "
        f"{len(analysis['cross_split_groups'])}"
    )

    print(
        "Duplicate groups crossing classes: "
        f"{len(analysis['cross_class_groups'])}"
    )

    print("\nDuplicate groups by split:")

    for key, count in sorted(
        analysis["split_distribution"].items()
    ):

        print(
            f"  {key}: {count}"
        )

    print("\nDuplicate groups by class:")

    for key, count in sorted(
        analysis["class_distribution"].items()
    ):

        print(
            f"  {key}: {count}"
        )

    print(
        f"\nReport saved to:\n"
        f"{REPORT_PATH}"
    )
    
    