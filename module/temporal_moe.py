import torch
import torch.nn as nn


class MLPExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1, activation="gelu"):
        super().__init__()
        act = nn.GELU() if activation == "gelu" else nn.SiLU()
        # Preserve enough capacity after bottlenecking.
        bottleneck_dim = max(hidden_dim // 2, input_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
            act,
            nn.Linear(bottleneck_dim, 1),
        )

    def forward(self, x):
        """
        x: [B*N, L_y, L_out + te_dim + 1]
        return: [B*N, L_y, 1]
        """
        return self.net(x)


class GatedMLPExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.value = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        x: [B*N, L_y, L_out + te_dim + 1]
        return: [B*N, L_y, 1]
        """
        x = self.norm(x)
        # Gate the value path feature-wise.
        return self.out(self.value(x) * self.gate(x))


class TemporalMoE(nn.Module):
    """
    Mixture of Experts
    """

    def __init__(self, input_dim, hidden_dim, num_experts=4, top_k=2, dropout=0.1):
        super().__init__()
        self.num_experts = max(1, int(num_experts))
        self.top_k = min(max(1, int(top_k)), self.num_experts)
        gate_hidden = max(hidden_dim // 2, input_dim)

        self.router = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, gate_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, self.num_experts),
        )
        self.experts = nn.ModuleList()
        for expert_idx in range(self.num_experts):
            if expert_idx % 3 == 0:
                self.experts.append(MLPExpert(input_dim, hidden_dim, dropout, activation="gelu"))
            elif expert_idx % 3 == 1:
                self.experts.append(MLPExpert(input_dim, hidden_dim, dropout, activation="silu"))
            else:
                self.experts.append(GatedMLPExpert(input_dim, hidden_dim, dropout))

    def forward(self, x):
        """
        x: [B*N, L_y, L_out + te_dim + 1]
        return: [B*N, L_y, 1]
        """
        # Route each horizon step to its best experts.
        logits = self.router(x)  # [B*N, L_y, num_experts]
        if self.top_k < self.num_experts:
            top_logits, top_indices = torch.topk(logits, self.top_k, dim=-1)  # [B*N, L_y, top_k]  [B*N, L_y, top_k]
            sparse_logits = torch.full_like(logits, -float("inf"))  # [B*N, L_y, num_experts]
            logits = sparse_logits.scatter(-1, top_indices, top_logits)  # [B*N, L_y, num_experts]

        weights = torch.softmax(logits, dim=-1)  # [B*N, L_y, num_experts]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)  # [B*N, L_y, 1, num_experts]
        # Weighted sum gives the residual correction.
        return (expert_outputs * weights.unsqueeze(-2)).sum(dim=-1), weights  # [B*N, L_y, 1]  [B*N, L_y, num_experts]
