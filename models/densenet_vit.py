import torch
import torch.nn as nn
from .densenet121 import build_densenet121
from .vit import build_vit

class DenseNetViT(nn.Module):
    def __init__(
        self, 
        pretrained: bool = True, 
        num_classes: int = 2, 
        dropout_rate: float = 0.5,
        vit_variant: str = "vit_b_16",
        fusion: str = "concat"
    ):
        super().__init__()
        
        # 1. Local Feature Extractor (DenseNet121)
        # Load densenet but replace its custom classifier with Identity 
        # so it just returns the 1024-dim pooled features.
        self.densenet = build_densenet121(pretrained=pretrained, num_classes=num_classes)
        self.densenet.classifier = nn.Identity()
        self.densenet_dim = 1024
        
        # 2. Global Feature Extractor (ViT)
        self.vit = build_vit(variant=vit_variant, pretrained=pretrained)
        self.vit_dim = 768 # Standard for vit_b_16
        
        # 3. Fusion & Classifier
        if fusion != "concat":
            raise NotImplementedError(f"Fusion strategy '{fusion}' not supported yet.")
            
        self.fusion_dim = self.densenet_dim + self.vit_dim # 1792
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.fusion_dim, num_classes)
        )

    def forward(self, x):
        # Extract local features
        local_features = self.densenet(x)
        
        # Extract global features
        global_features = self.vit(x)
        
        # Simple Concatenation
        fused = torch.cat([local_features, global_features], dim=1)
        
        # Classify
        out = self.classifier(fused)
        return out

def build_densenet_vit(
    pretrained: bool = True, 
    num_classes: int = 2, 
    dropout_rate: float = 0.5,
    vit_variant: str = "vit_b_16",
    fusion: str = "concat"
) -> nn.Module:
    """Returns the combined DenseNet + ViT model."""
    return DenseNetViT(pretrained, num_classes, dropout_rate, vit_variant, fusion)

if __name__ == "__main__":
    print("Testing DenseNet+ViT build...")
    model = build_densenet_vit()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Output shape: {out.shape}")  # Expected: [2, 2]
    print("Success!")

