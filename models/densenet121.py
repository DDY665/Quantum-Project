import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

def build_densenet121(
    pretrained: bool = True,
    num_classes: int = 2,
    dropout_rate: float = 0.5,
) -> nn.Module:
    """
    Returns a DenseNet121 with a custom binary classifier head.
    Classifier: Dropout -> Linear(1024, num_classes)
    """
    weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = densenet121(weights=weights)
    
    # Replace classifier head
    in_features = model.classifier.in_features  # typically 1024 for DenseNet121
    
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes)
    )
    
    return model

if __name__ == "__main__":
    # Quick sanity check
    print("Testing DenseNet121 build...")
    model = build_densenet121()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Output shape: {out.shape}")  # Expected: [2, 2]
    print("Success!")

