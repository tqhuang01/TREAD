import torch
import torch.nn as nn
import torch.nn.functional as F


class NConv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, A):
        # x: [B, hidden_dim, N]
        # A: [B, N, N]
        x = torch.einsum('bcn,bnm->bcm', (x, A))  # [B, hidden_dim, N] used

        return x.contiguous()  # [B, hidden_dim, N]


class LinearConv(nn.Module):
    """1x1 convolution as LinearConv layer"""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.mlp = nn.Linear(c_in, c_out)
        # self.mlp_c = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        # x: [B, hidden_dim+hidden_dim+hidden_dim, N]

        return self.mlp(x.permute(0, 2, 1)).permute(0, 2, 1)
        # return self.mlp_c(x.unsqueeze(-2)).squeeze(-2)


class GCN(nn.Module):
    """Graph Neural Networks, GNNs"""

    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super().__init__()
        self.nconv = NConv()
        c_in = (order * support_len + 1) * c_in
        # c_in = (order*support_len)*c_in
        self.mlp = LinearConv(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        # x: [B, hidden_dim, N]
        # support: [[B, N, N]]
        # a: [B, N, N]
        out = [x]  # [[B, hidden_dim, N]]
        for a in support:
            x1 = self.nconv(x, a)  # [B, hidden_dim, N]
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        ### concat x and x_conv ###
        h = torch.cat(out, dim=1)  # [B, hidden_dim+hidden_dim+hidden_dim, N]
        h = self.mlp(h)  # [B, hidden_dim, N]
        return F.relu(h)
