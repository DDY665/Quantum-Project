from pathlib import Path

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = 224

BATCH_SIZE = 32

NUM_WORKERS = 0

RANDOM_SEED = 42


# ============================================================
# Label mapping
# ============================================================

LABEL_MAP = {
    "NON_PNEUMONIA": 0,
    "PNEUMONIA": 1,
}


# ============================================================
# Dataset transforms
# ============================================================

# Training:
# - Resize
# - Random horizontal flip
# - Small rotation
# - Convert grayscale -> 3 channels
# - Convert to tensor
# - Normalize

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            degrees=5
        ),

        transforms.Grayscale(
            num_output_channels=3
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


# Validation/Test:
# NO augmentation.

EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.Grayscale(
            num_output_channels=3
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


# ============================================================
# Dataset class
# ============================================================

class PneumoniaDataset(Dataset):
    """
    PyTorch dataset for the prepared pneumonia datasets.

    The manifest contains:
        path
        label
        source
        original_label
        dataset
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform=None,
    ):

        self.dataframe = (
            dataframe.reset_index(
                drop=True
            )
        )

        self.transform = transform

        required_columns = {
            "path",
            "label",
        }

        missing = (
            required_columns
            - set(self.dataframe.columns)
        )

        if missing:
            raise ValueError(
                f"Missing required columns: "
                f"{sorted(missing)}"
            )

    def __len__(self):

        return len(
            self.dataframe
        )

    def __getitem__(
        self,
        index: int,
    ):

        row = self.dataframe.iloc[index]

        image_path = Path(
            row["path"]
        )

        label_name = str(
            row["label"]
        )

        if label_name not in LABEL_MAP:
            raise ValueError(
                f"Unknown label "
                f"'{label_name}' "
                f"in {image_path}"
            )

        label = LABEL_MAP[
            label_name
        ]

        try:

            image = Image.open(
                image_path
            ).convert("L")

        except Exception as error:

            raise RuntimeError(
                f"Failed to load image:\n"
                f"{image_path}"
            ) from error

        if self.transform is not None:

            image = self.transform(
                image
            )

        return image, label


# ============================================================
# Load split
# ============================================================

def load_split(
    dataset_name: str,
    split: str,
) -> pd.DataFrame:

    split_file = (
        PROCESSED_ROOT
        / dataset_name
        / "splits"
        / f"{split}.csv"
    )

    if not split_file.exists():

        raise FileNotFoundError(
            f"Split file not found:\n"
            f"{split_file}\n\n"
            "Run prepare_dataset.py first."
        )

    dataframe = pd.read_csv(
        split_file
    )

    return dataframe


# ============================================================
# Validate split
# ============================================================

def validate_split(
    dataframe: pd.DataFrame,
    split_name: str,
):

    print(
        f"\nValidating {split_name}..."
    )

    if dataframe.empty:

        raise ValueError(
            f"{split_name} is empty."
        )

    # --------------------------------------------------------
    # Check paths
    # --------------------------------------------------------

    missing_paths = []

    for path in dataframe["path"]:

        if not Path(path).exists():

            missing_paths.append(
                path
            )

    if missing_paths:

        print(
            f"Missing images: "
            f"{len(missing_paths)}"
        )

        raise FileNotFoundError(
            f"Missing image example:\n"
            f"{missing_paths[0]}"
        )

    print(
        f"All {len(dataframe)} "
        f"image paths exist."
    )

    # --------------------------------------------------------
    # Check labels
    # --------------------------------------------------------

    unknown_labels = set(
        dataframe["label"]
    ) - set(LABEL_MAP)

    if unknown_labels:

        raise ValueError(
            f"Unknown labels: "
            f"{sorted(unknown_labels)}"
        )

    print(
        "Labels: PASSED"
    )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print(
        "\nClass distribution:"
    )

    print(
        dataframe["label"]
        .value_counts()
        .to_string()
    )


# ============================================================
# Create dataloaders
# ============================================================

def create_dataloaders(
    dataset_name: str,
):

    print("=" * 60)
    print(
        f"CREATING DATALOADERS - "
        f"{dataset_name}"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Load splits
    # --------------------------------------------------------

    train_df = load_split(
        dataset_name,
        "train",
    )

    val_df = load_split(
        dataset_name,
        "val",
    )

    test_df = load_split(
        dataset_name,
        "test",
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_split(
        train_df,
        "TRAIN",
    )

    validate_split(
        val_df,
        "VAL",
    )

    validate_split(
        test_df,
        "TEST",
    )

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    train_dataset = PneumoniaDataset(
        train_df,
        transform=TRAIN_TRANSFORM,
    )

    val_dataset = PneumoniaDataset(
        val_df,
        transform=EVAL_TRANSFORM,
    )

    test_dataset = PneumoniaDataset(
        test_df,
        transform=EVAL_TRANSFORM,
    )

    # --------------------------------------------------------
    # Create dataloaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )


# ============================================================
# Test one batch
# ============================================================

def test_dataloaders(
    dataset_name: str,
):

    (
        train_loader,
        val_loader,
        test_loader,
    ) = create_dataloaders(
        dataset_name
    )

    print("\n" + "=" * 60)
    print("DATALOADER VALIDATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Training batch
    # --------------------------------------------------------

    train_images, train_labels = next(
        iter(train_loader)
    )

    print("\nTRAIN batch:")

    print(
        f"  Images shape: "
        f"{tuple(train_images.shape)}"
    )

    print(
        f"  Labels shape: "
        f"{tuple(train_labels.shape)}"
    )

    print(
        f"  Image dtype: "
        f"{train_images.dtype}"
    )

    print(
        f"  Label dtype: "
        f"{train_labels.dtype}"
    )

    print(
        f"  Image min: "
        f"{train_images.min().item():.4f}"
    )

    print(
        f"  Image max: "
        f"{train_images.max().item():.4f}"
    )

    # --------------------------------------------------------
    # Validation batch
    # --------------------------------------------------------

    val_images, val_labels = next(
        iter(val_loader)
    )

    print("\nVAL batch:")

    print(
        f"  Images shape: "
        f"{tuple(val_images.shape)}"
    )

    print(
        f"  Labels shape: "
        f"{tuple(val_labels.shape)}"
    )

    # --------------------------------------------------------
    # Test batch
    # --------------------------------------------------------

    test_images, test_labels = next(
        iter(test_loader)
    )

    print("\nTEST batch:")

    print(
        f"  Images shape: "
        f"{tuple(test_images.shape)}"
    )

    print(
        f"  Labels shape: "
        f"{tuple(test_labels.shape)}"
    )

    # --------------------------------------------------------
    # Final checks
    # --------------------------------------------------------

    expected_channels = 3

    expected_height = IMAGE_SIZE

    expected_width = IMAGE_SIZE

    if train_images.shape[1] != expected_channels:
        raise ValueError(
            "Unexpected number of image channels."
        )

    if train_images.shape[2] != expected_height:
        raise ValueError(
            "Unexpected image height."
        )

    if train_images.shape[3] != expected_width:
        raise ValueError(
            "Unexpected image width."
        )

    if not torch.isfinite(
        train_images
    ).all():

        raise ValueError(
            "Training batch contains "
            "NaN or infinite values."
        )

    if not torch.isfinite(
        val_images
    ).all():

        raise ValueError(
            "Validation batch contains "
            "NaN or infinite values."
        )

    if not torch.isfinite(
        test_images
    ).all():

        raise ValueError(
            "Test batch contains "
            "NaN or infinite values."
        )

    print(
        "\nImage preprocessing: PASSED"
    )

    print(
        "Tensor validation: PASSED"
    )

    print(
        "\nDataloader validation "
        "completed successfully."
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # Change this to "dataset_A"
    # when testing Dataset A.

    DATASET_NAME = "dataset_B"

    test_dataloaders(
        DATASET_NAME
    )