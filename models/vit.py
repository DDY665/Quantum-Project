import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

def build_vit(
    variant: str = "vit_b_16",
    pretrained: bool = True,
) -> nn.Module:
    """
    Returns a Vision Transformer backbone without the classification head.
    Outputs the global CLS token feature vector (768-dim for vit_b_16).
    """
    if variant != "vit_b_16":
        raise NotImplementedError(f"Variant {variant} is not supported yet.")
        
    weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    model = vit_b_16(weights=weights)
    
    # We only want the feature extraction part, not the final 1000-class head.
    # By replacing `heads` with Identity, the model returns the 768-dim CLS token.
    model.heads = nn.Identity()
    
    return model

if __name__ == "__main__":
    print("Testing ViT build...")
    model = build_vit()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Output shape: {out.shape}")  # Expected: [2, 768]
    print("Success!")

