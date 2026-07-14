import torch
import torch.nn as nn
from module.RevIN import RevIN
from module.positional_encoding import SinusoidalPositionalEncoding, LearnablePositionalEncoding


class FeatureSpaceAdapter(nn.Module):
    """Deep nonlinear feature-space adapter for input/output representation transformation."""

    def __init__(self, num_features):
        super().__init__()

        hidden_dim = num_features * 4

        self.encoder_layers = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_features),
        )

        self.decoder_layers = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_features),
        )

    def encode(self, x):
        """
        x: [B, L_in, N]
        """
        return self.encoder_layers(x)  # [B, L_in, N]

    def decode(self, x):
        """
        x: [B, L_out, N]
        """
        return self.decoder_layers(x)  # [B, L_out, N]


class TemporalContextAdapter(nn.Module):
    """
    Temporal context learning with Transformer encoders.
    """

    def __init__(self, num_features, nhead=1, hidden_dim=None, num_layers=3, use_revin=False):
        super().__init__()
        self.use_revin = use_revin
        if use_revin:
            self.revin = RevIN(num_features=num_features)

        hidden_dim = hidden_dim or 4 * num_features

        self.add_pe_e = LearnablePositionalEncoding(d_model=num_features)
        self.add_pe_d = LearnablePositionalEncoding(d_model=num_features)

        # Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=num_features,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Decoder
        dec_layer = nn.TransformerEncoderLayer(
            d_model=num_features,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=num_layers)

    def encode(self, x):
        """
        x: [B, L_in, N]
        """
        if self.use_revin:
            x = self.revin(x, mode="norm")  # [B, L_in, N]
        x = self.encoder(self.add_pe_e(x))  # [B, L_in, N]
        return x

    def decode(self, x):
        """
        x: [B, L_out, N]
        """
        if self.use_revin:
            x = self.revin(x, mode="denorm")  # [B, L_out, N]
        x = self.decoder(self.add_pe_d(x))  # [B, L_out, N]
        return x
