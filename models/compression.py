import torch
import torch.nn as nn

class FeatureCompressor(nn.Module):
    def __init__(
        self,
        input_dim: int = 1792,
        hidden_dim: int = 128,
        output_dim: int = 8,
        dropout_rate: float = 0.2,
        use_tanh_scaling: bool = True
    ):
        super().__init__()
        self.compressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )
        self.use_tanh_scaling = use_tanh_scaling
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.compressor(x)
        if self.use_tanh_scaling:
            out = self.tanh(out)
        return out
