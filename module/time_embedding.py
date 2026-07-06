import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    """"""

    def __init__(self, te_dim: int):
        super().__init__()
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, te_dim - 1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: [B*N, L_x, 1]
        """
        out1 = self.te_scale(t)
        out2 = torch.sin(self.te_periodic(t))
        return torch.cat([out1, out2], dim=-1)
