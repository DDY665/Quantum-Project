import torch
import torch.nn as nn
from .densenet_vit import build_densenet_vit
from .compression import FeatureCompressor

class CompressedPneumoniaModel(nn.Module):
    def __init__(
        self,
        output_dim: int = 8,
        hidden_dim: int = 128,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        use_tanh_scaling: bool = True
    ):
        super().__init__()
        base = build_densenet_vit(pretrained=pretrained, num_classes=num_classes)
        self.densenet = base.densenet
        self.vit = base.vit
        self.fusion_dim = 1792
        self.compressor = FeatureCompressor(
            input_dim=self.fusion_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout_rate=dropout_rate,
            use_tanh_scaling=use_tanh_scaling
        )
        self.classifier = nn.Linear(output_dim, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        f_local = self.densenet(x)
        f_global = self.vit(x)
        fused = torch.cat([f_local, f_global], dim=1)
        compact = self.compressor(fused)
        return compact

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compact = self.extract_features(x)
        logits = self.classifier(compact)
        return logits

def build_compressed_model(
    output_dim: int = 8,
    hidden_dim: int = 128,
    num_classes: int = 2,
    pretrained: bool = True,
    dropout_rate: float = 0.2,
    use_tanh_scaling: bool = True
) -> nn.Module:
    return CompressedPneumoniaModel(
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        use_tanh_scaling=use_tanh_scaling
    )
