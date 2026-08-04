import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

from module import TTCN, GCN, TimeEmbedding, LearnablePositionalEncoding, TemporalContextAdapter, TemporalMoE
from lib.utils import periodic_extension, zero_padding
from core.icl.moment import load_moment_model


class TREAD(nn.Module):
    """
    Time-Regularized Expansion and Adaptive Decoding
    """

    def __init__(self, args: argparse.Namespace, supports=None, dropout=0.1):
        super().__init__()
        self.backbone = args.backbone
        self.L_in = args.ctx_len
        self.L_out = args.fcast_len
        self.use_padding = args.use_padding

        self.num_features = args.num_features

        self.fm_chunk_size = max(args.batch_size, 128)

        self.num_blocks = args.num_blocks
        self.nhead = args.nhead
        self.num_layers = args.num_layers

        self.hidden_dim = args.hidden_dim
        self.te_dim = args.te_dim

        self.device = args.device

        self.supports = supports if supports is not None else []
        self.supports_len = len(self.supports) + 1  # including adaptive

        # === FM ===
        self.fm = load_moment_model(
            model_dir=args.model_dir,
            context_length=self.L_in,
            forecast_horizon=self.L_out
        )

        # === Adapter ===
        self.adapter = TemporalContextAdapter(num_features=self.num_features)
        self.adapter_encode_gate = nn.Parameter(torch.tensor(-3.0))
        self.adapter_decode_gate = nn.Parameter(torch.tensor(-3.0))

        self.horizon_mlp = nn.Sequential(
            nn.LayerNorm(self.L_out),
            nn.Linear(self.L_out, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.L_out),
        )
        self.horizon_residual_gate = nn.Parameter(torch.tensor(-2.0))
        self.time_residual_gate = nn.Parameter(torch.tensor(-3.0))

        # === Time embedding ===
        self.time_embed1 = TimeEmbedding(te_dim=self.te_dim)
        self.time_embed2 = TimeEmbedding(te_dim=self.te_dim)

        # === TTCN ===
        self.ttcn = TTCN(input_dim=self.te_dim + 1, ttcn_dim=self.hidden_dim - 1)

        # === TransformerEncoder ===
        tf_dim = self.hidden_dim
        tf_dim1 = self.L_out
        tf_dim2 = 1 + self.te_dim
        self.add_pe = LearnablePositionalEncoding(tf_dim)
        self.add_pe1 = LearnablePositionalEncoding(tf_dim1)
        self.add_pe2 = LearnablePositionalEncoding(tf_dim2)

        def make_encoder(d_model: int):
            # Keep attention heads compatible with d_model.
            nhead = self.nhead if d_model % self.nhead == 0 else 1
            ff_dim = max(self.hidden_dim, d_model * 4)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            return nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.transformer_encoder = nn.ModuleList()
        for _ in range(self.num_blocks):
            self.transformer_encoder.append(make_encoder(tf_dim))

        self.transformer_encoder1 = nn.ModuleList()
        for _ in range(self.num_blocks):
            self.transformer_encoder1.append(make_encoder(tf_dim1))

        self.transformer_encoder2 = nn.ModuleList()
        for _ in range(self.num_blocks):
            self.transformer_encoder2.append(make_encoder(tf_dim2))

        # === Gating Mechanism ===
        self.gates1 = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(tf_dim1),
                nn.Linear(tf_dim1, tf_dim1),
                nn.Sigmoid(),
            )
            for _ in range(self.num_blocks)
        ])
        self.gates2 = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(tf_dim2),
                nn.Linear(tf_dim2, tf_dim2),
                nn.Sigmoid(),
            )
            for _ in range(self.num_blocks)
        ])
        self.imts_norms = nn.ModuleList([nn.LayerNorm(tf_dim) for _ in range(self.num_blocks)])
        self.gate_norms1 = nn.ModuleList([nn.LayerNorm(tf_dim1) for _ in range(self.num_blocks)])
        self.gate_norms2 = nn.ModuleList([nn.LayerNorm(tf_dim2) for _ in range(self.num_blocks)])
        self.gnn_dropout = nn.Dropout(dropout)

        # === Output ===
        self.out_proj1 = nn.Sequential(
            nn.LayerNorm(tf_dim1),
            nn.Linear(tf_dim1, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1)
        )
        self.out_proj2 = nn.Sequential(
            nn.LayerNorm(tf_dim2),
            nn.Linear(tf_dim2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1)
        )
        self.moe_num_experts = getattr(args, "moe_num_experts", 4)
        self.moe_top_k = getattr(args, "moe_top_k", 2)
        self.moe_head = TemporalMoE(
            input_dim=tf_dim1 + tf_dim2,
            hidden_dim=self.hidden_dim,
            num_experts=self.moe_num_experts,
            top_k=self.moe_top_k,
            dropout=dropout,
        )
        self.moe_residual_gate = nn.Parameter(torch.tensor(-2.5))
        self.last_moe_weights = None

        # === Inter-time series modeling ===
        self.nodevec_dim = args.node_dim
        node_scale = self.nodevec_dim ** -0.5
        self.nodevec1 = nn.Parameter(torch.randn(self.num_features, self.nodevec_dim) * node_scale)
        self.nodevec2 = nn.Parameter(torch.randn(self.nodevec_dim, self.num_features) * node_scale)

        self.nodevec_linear1 = nn.ModuleList()
        self.nodevec_linear2 = nn.ModuleList()
        self.nodevec_gate1 = nn.ModuleList()
        self.nodevec_gate2 = nn.ModuleList()
        for _ in range(self.num_blocks):
            self.nodevec_linear1.append(nn.Linear(self.hidden_dim, self.nodevec_dim))
            self.nodevec_linear2.append(nn.Linear(self.hidden_dim, self.nodevec_dim))
            self.nodevec_gate1.append(nn.Sequential(
                nn.LayerNorm(self.hidden_dim + self.nodevec_dim),
                nn.Linear(self.hidden_dim + self.nodevec_dim, 1),
                nn.Sigmoid()))
            self.nodevec_gate2.append(nn.Sequential(
                nn.LayerNorm(self.hidden_dim + self.nodevec_dim),
                nn.Linear(self.hidden_dim + self.nodevec_dim, 1),
                nn.Sigmoid()))

        self.gconv = nn.ModuleList()  # gragh conv
        for _ in range(self.num_blocks):
            self.gconv.append(
                GCN(self.hidden_dim, self.hidden_dim, dropout, support_len=self.supports_len, order=args.num_hops))

        # === Decoder ===
        self.decoder = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.L_in)
        )

    def _prepare_supports(self, batch_size, num_nodes, dtype, device):
        # Normalize static supports to batched tensors.
        prepared = []
        for support in self.supports:
            if not torch.is_tensor(support):
                support = torch.as_tensor(support, dtype=dtype, device=device)
            else:
                support = support.to(device=device, dtype=dtype)

            if support.dim() == 2:
                support = support.unsqueeze(0).expand(batch_size, -1, -1)
            elif support.dim() == 3 and support.size(0) == 1:
                support = support.expand(batch_size, -1, -1)

            if support.size(-2) != num_nodes or support.size(-1) != num_nodes:
                raise ValueError(f"Support shape {tuple(support.shape)} does not match num_features={num_nodes}.")
            prepared.append(support)
        return prepared

    def gated_transformer_block1(self, x):
        """
        对序列 x 依次经过多层 Transformer + 门控融合
        """
        for layer_idx in range(self.num_blocks):
            x_pe = self.add_pe1(x)
            x_new = self.transformer_encoder1[layer_idx](x_pe)
            g = self.gates1[layer_idx](x)
            x = self.gate_norms1[layer_idx](x + g * (x_new - x))
        return x

    def gated_transformer_block2(self, x):
        """
        对序列 x 依次经过多层 Transformer + 门控融合
        """
        for layer_idx in range(self.num_blocks):
            x_pe = self.add_pe2(x)
            x_new = self.transformer_encoder2[layer_idx](x_pe)
            g = self.gates2[layer_idx](x)
            x = self.gate_norms2[layer_idx](x + g * (x_new - x))
        return x

    def _time_regularized_expansion(self, x_obs, t_obs, mask):
        """
        x_obs: [B, L_in, N]
        t_obs: [B, L_in]
        mask: [B, L_in, N]
        """
        B, L_x, N = x_obs.shape
        t_obs = t_obs.unsqueeze(1).repeat(1, N, 1).reshape(-1, L_x, 1)  # [B*N, L_x, 1]
        mask = mask.permute(0, 2, 1).reshape(-1, L_x, 1)  # [B*N, L_x, 1]
        x_obs = x_obs.permute(0, 2, 1).reshape(-1, L_x, 1)  # [B*N, L_x, 1]

        te_obs = self.time_embed1(t_obs)  # [B*N, L, te_dim]
        z = torch.cat([x_obs, te_obs], dim=-1)  # [B*N, L, te_dim+1]

        # Encode irregular observations into node states.
        h_ttcn = self.ttcn(z, mask)  # [B*N, ttcn_dim] = [B*N, hidden_dim-1]

        mask = (mask.sum(dim=1) > 0).to(dtype=h_ttcn.dtype)  # [B*N, 1] 避免序列中可能不存在观测值
        h = torch.cat([h_ttcn, mask], dim=-1)  # [B*N, hidden_dim]

        h = h.view(-1, self.num_features, self.hidden_dim)  # [B, N, hidden_dim]
        B, N, hidden_dim = h.shape

        for layer in range(self.num_blocks):
            h_last = h

            # === Transformer for temporal modeling ===
            h_pe = self.add_pe(h)  # [B, N, hidden_dim]

            h_tf = self.transformer_encoder[layer](h_pe)  # [B, N, hidden_dim]

            # === GNN for inter-time series modeling ===
            # === time-adaptive graph structure learning ===
            """ 时变自适应图结构学习（Time-Varying Adaptive Graph Structure Learning） """
            # [N, node_dim] -> [1, N, node_dim] -> [B, N, node_dim]
            # [node_dim, N] -> [1, node_dim, N] -> [B, node_dim, N]
            nodevec1 = self.nodevec1.view(1, N, self.nodevec_dim).expand(B, -1, -1)  # [B, N, node_dim]
            nodevec2 = self.nodevec2.view(1, self.nodevec_dim, N).expand(B, -1, -1)  # [B, node_dim, N]

            """ 门控加法操作 """
            # [B, N, hidden_dim+node_dim] -> [B, N, 1]
            # [B, N, hidden_dim+node_dim] -> [B, N, 1]
            x_gate1 = self.nodevec_gate1[layer](torch.cat([h_tf, nodevec1], dim=-1))  # [B, N, 1]
            x_gate2 = self.nodevec_gate2[layer](torch.cat([h_tf, nodevec2.permute(0, 2, 1)], dim=-1))  # [B, N, 1]

            """ 动态补丁嵌入 """
            # [B, N, 1] * [B, N, node_dim] -> [B, N, node_dim]
            # [B, N, 1] * [B, N, node_dim] -> [B, N, node_dim]
            x_p1 = x_gate1 * self.nodevec_linear1[layer](h_tf)  # [B, N, node_dim]
            x_p2 = x_gate2 * self.nodevec_linear2[layer](h_tf)  # [B, N, node_dim]

            nodevec1 = nodevec1 + x_p1  # [B, N, node_dim] + [B, N, node_dim]
            nodevec2 = nodevec2 + x_p2.permute(0, 2, 1)  # [B, node_dim, N] + [B, node_dim, N]

            adp = F.softmax(F.relu(torch.matmul(nodevec1, nodevec2)), dim=-1)  # [B, N, N] used

            """ 使用图神经网络建模跨序列相关性（GNNs to Model Inter-Time Series Correlation） """
            new_supports = self._prepare_supports(B, N, h_tf.dtype, h_tf.device) + [adp]  # [[B, N, N]]

            # [B, hidden_dim, N], [[B, N, N]]
            h_gcn = self.gconv[layer](h_tf.permute(0, 2, 1), new_supports)  # [B, hidden_dim, N]
            h_gcn = h_gcn.permute(0, 2, 1)  # [B, N, hidden_dim]

            h = self.imts_norms[layer](h_last + self.gnn_dropout(h_gcn))

        x_reg = self.decoder(h)  # [B, N, L_in]

        return x_reg

    def _fm_adaptation_and_forecasting(self, x_reg):
        """
        x_reg: [B, N, L_in]
        return: [B, N, L_out]
        """
        x_encoded = self.adapter.encode(x_reg.permute(0, 2, 1)).permute(0, 2, 1)  # [B, N, L_in]
        x_reg = x_reg + torch.sigmoid(self.adapter_encode_gate) * (x_encoded - x_reg)  # [B, N, L_in]

        B, N, L = x_reg.shape  # [B, N, L_in]
        x_reg = x_reg.reshape(B * N, 1, L)  # [B*N, 1, L_in]
        forecasts = []

        # Chunk FM calls to avoid memory spikes.
        for start in range(0, x_reg.size(0), self.fm_chunk_size):
            forecast = self.fm(x_enc=x_reg[start:start + self.fm_chunk_size]).forecast  # [fm_chunk_size, 1, L_out]
            forecasts.append(forecast)
        y_fm = torch.cat(forecasts, dim=0).reshape(B, N, self.L_out)  # [B, N, L_out]

        y_fm_decoded = self.adapter.decode(y_fm.permute(0, 2, 1)).permute(0, 2, 1)  # [B, N, L_out]
        y_fm = y_fm + torch.sigmoid(self.adapter_decode_gate) * (y_fm_decoded - y_fm)  # [B, N, L_out]
        return y_fm  # [B, N, L_out]

    def _adaptive_decoding(self, y_fm, t_pred):
        """
        y_fm: [B, N, L_out]
        t_pred: [B, L_y]
        return: [B, L_y, N]
        """
        B, N, L_out = y_fm.shape
        L_y = t_pred.shape[1]
        query_times = t_pred.unsqueeze(1).expand(B, N, L_y).reshape(B * N, L_y, 1)  # [B*N, L_y, 1]

        # Initialize the base prediction.
        y_base = y_fm[:, :, :L_y].reshape(B * N, L_y, 1)  # [B*N, L_y, 1]

        # 1.Full-Horizon Correction
        horizon_features = y_fm.reshape(B * N, 1, self.L_out).expand(-1, L_y, -1)  # [B*N, L_y, L_out]
        horizon_features = horizon_features + self.horizon_mlp(horizon_features)  # [B*N, L_y, L_out]
        horizon_context = self.gated_transformer_block1(horizon_features)  # [B*N, L_y, L_out]
        horizon_delta = self.out_proj1(horizon_context)  # [B*N, L_y, 1]

        y_h = y_base + torch.sigmoid(self.horizon_residual_gate) * horizon_delta  # [B*N, L_y, 1]

        # 2.Query-Time Correction
        query_times_embedding = self.time_embed2(query_times)  # [B*N, L_y, te_dim]
        time_features = torch.cat([y_h, query_times_embedding], dim=-1)  # [B*N, L_y, te_dim + 1]
        time_context = self.gated_transformer_block2(time_features)  # [B*N, L_y, te_dim + 1]
        time_delta = self.out_proj2(time_context)  # [B*N, L_y, 1]

        # 3.Sparse MoE Correction
        moe_features = torch.cat([horizon_context, time_context], dim=-1)  # [B*N, L_y, L_out + te_dim + 1]
        moe_delta, moe_weights = self.moe_head(moe_features)  # [B*N, L_y, 1]  [B*N, L_y, num_experts]
        self.last_moe_weights = moe_weights.detach()

        y_pred = (
                y_h  # [B*N, L_y, 1]
                + torch.sigmoid(self.time_residual_gate) * time_delta  # [B*N, L_y, 1]
                + torch.sigmoid(self.moe_residual_gate) * moe_delta  # [B*N, L_y, 1]
        )
        return y_pred.reshape(B, N, L_y).permute(0, 2, 1)  # [B, L_y, N]

    def forward(self, t_pred, x_obs, t_obs, mask):
        """
        t_pred: [B, L_y]
        t_obs: [B, L_x]
        x_obs: [B, L_x, N]
        mask: [B, L_x, N]
        """
        if self.use_padding == "periodic":
            x_obs = periodic_extension(x_obs, self.L_in)  # [B, L_in, N]
            t_obs = periodic_extension(t_obs, self.L_in)  # [B, L_in]
            mask = periodic_extension(mask, self.L_in)  # [B, L_in, N]
        elif self.use_padding == "zero":
            x_obs = zero_padding(x_obs, self.L_in)  # [B, L_in, N]
            t_obs = zero_padding(t_obs, self.L_in)  # [B, L_in]
            mask = zero_padding(mask, self.L_in)  # [B, L_in, N]

        # === 1.Time-Regularized Expansion ===
        x_reg = self._time_regularized_expansion(x_obs, t_obs, mask)  # [B, N, L_in]

        # === 2.Foundation Model Adaptation and Forecasting ===
        y_fm = self._fm_adaptation_and_forecasting(x_reg)  # [B, N, L_out]

        # === 3.Adaptive Decoding ===
        return self._adaptive_decoding(y_fm, t_pred)  # [B, L_y, N]
