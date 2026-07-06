import torch
import torch.nn as nn
import torch.nn.functional as F


class TTCN(nn.Module):

    def __init__(self, input_dim: int, ttcn_dim: int):
        super().__init__()
        self.ttcn_dim = ttcn_dim
        self.filter_generator = nn.Sequential(
            nn.Linear(input_dim, ttcn_dim),
            nn.ReLU(),
            nn.Linear(ttcn_dim, ttcn_dim),
            nn.ReLU(),
            nn.Linear(ttcn_dim, input_dim * ttcn_dim)
        )
        self.bias = nn.Parameter(torch.randn(1, ttcn_dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: [B*N, L_x, F], [B*N, L_x, te_dim + 1]
        mask: [B*N, L_x, 1]
        h: [B*N, ttcn_dim]
        """
        N, L_x, _ = mask.shape  # [B*N, L_x, 1]

        x_expanded = x.unsqueeze(-2).repeat(1, 1, self.ttcn_dim, 1)  # [B*N, L_x, ttcn_dim, F]

        filters = self.filter_generator(x)  # [B*N, L_x, F*ttcn_dim]
        filters_masked = filters * mask + (1 - mask) * (-1e8)  # [B*N, L_x, F*ttcn_dim]

        filters_norm = F.softmax(filters_masked, dim=-2)  # [B*N, L_x, F*ttcn_dim]
        filters_norm = filters_norm.view(N, L_x, self.ttcn_dim, -1)  # [B*N, L_x, ttcn_dim, F]

        out = torch.sum(torch.sum(x_expanded * filters_norm, dim=-3), dim=-1)  # [B*N, ttcn_dim]
        h = torch.relu(out + self.bias)  # [B*N, ttcn_dim]
        return h
