import torch
import torch.nn as nn
import argparse

from lib.utils import zero_padding
from core.icl.moment import load_moment_model


class FM_ZP(nn.Module):
    """
    Time-Regularized Expansion and Adaptive Decoding
    """

    def __init__(self, args: argparse.Namespace, supports=None, dropout=0.1):
        super().__init__()
        self.L_in = args.ctx_len
        self.L_out = args.fcast_len

        self.fm_chunk_size = max(args.batch_size, 128)
        self.use_padding = args.use_padding

        # === FM ===
        self.fm = load_moment_model(
            model_dir=args.model_dir,
            context_length=self.L_in,
            forecast_horizon=self.L_out
        )

    def forward(self, t_pred, x_obs, t_obs, mask):
        """
        t_pred: [B, L_y]
        t_obs: [B, L_x]
        x_obs: [B, L_x, N]
        mask: [B, L_x, N]
        """
        if self.use_padding == "zero":
            x_obs = zero_padding(x_obs, self.L_in)  # [B, L_in, N]

        x_obs = x_obs.permute(0, 2, 1)  # [B, N, L_in]
        B, N, L = x_obs.shape  # [B, N, L_in]
        x = x_obs.reshape(B * N, 1, L)  # [B*N, 1, L_in]
        forecasts = []

        L_y = t_pred.shape[-1]

        # Chunk FM calls to avoid memory spikes.
        for start in range(0, x.size(0), self.fm_chunk_size):
            forecast = self.fm(x_enc=x[start:start + self.fm_chunk_size]).forecast  # [fm_chunk_size, 1, L_out]
            forecasts.append(forecast)
        y_fm = torch.cat(forecasts, dim=0).reshape(B, N, self.L_out)  # [B, N, L_out]

        return y_fm[:, :, :L_y].permute(0, 2, 1)
