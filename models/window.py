import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base.utils import MLP, DenseSequential, get_pool


class InsulinAttentionModel(nn.Module):
    def __init__(
        self,
        examination_size: int = 78,
        insulin_size: int = 12,
        sugar_size: int = 7,
        drug_size: int = 28,
        normal_fes_size: int = 46,
        monotonicity_fes_size: int = 64,
        window_size: int = 2,
        mlp_encode_dilation: int = 8,  # 2,3,4,5,6,7,8
        mlp_output_dim: int = 128,  # 64, 128, 256
        mlp_encode_blocks: int = 3,  # 1,2,3
        mlp_encode_dropout: float = 0,  # 0.1-0.6
        dense_hidden_size: int = 128,  # 64, 128, 256
        dense_dropout: float = 0.5,
        pool_name: str = "max",
        activation_name: str = "relu",
        attn_num_heads: int = 8,  # 2,4,8,16
        attn_dropout: float = 0,
    ):
        super().__init__()
        self.examination_size = examination_size
        self.insulin_size = insulin_size
        self.sugar_size = sugar_size
        self.drug_size = drug_size
        self.normal_fes_size = normal_fes_size
        self.monotonicity_fes_size = monotonicity_fes_size

        self.window_size = window_size

        def _get_mlp_layer(size: int, output_size: int) -> MLP:
            return MLP(
                num_inputs=size,
                num_hidden=size * mlp_encode_dilation,
                num_outputs=output_size,
                num_block=mlp_encode_blocks,
                dropout=mlp_encode_dropout,
                activation_name=activation_name,
            )

        day_size = 10
        self.day_mlp = _get_mlp_layer(1, day_size)
        self.examination_mlp = _get_mlp_layer(examination_size, mlp_output_dim)
        self.normal_fes_mlp = _get_mlp_layer(normal_fes_size, mlp_output_dim)
        self.monotonicity_fes_mlp = _get_mlp_layer(
            monotonicity_fes_size, mlp_output_dim
        )

        self.insulin_mlp = _get_mlp_layer(insulin_size, mlp_output_dim - day_size)
        self.sugar_mlp = _get_mlp_layer(sugar_size, mlp_output_dim - day_size)
        self.drug_mlp = _get_mlp_layer(drug_size, mlp_output_dim - day_size)

        self.insulin_attn = nn.TransformerEncoderLayer(
            mlp_output_dim,
            attn_num_heads,
            dim_feedforward=1024,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.sugar_attn = nn.TransformerEncoderLayer(
            mlp_output_dim,
            attn_num_heads,
            dim_feedforward=1024,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.drug_attn = nn.TransformerEncoderLayer(
            mlp_output_dim,
            attn_num_heads,
            dim_feedforward=1024,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.normal_mix_attn = nn.TransformerEncoderLayer(
            mlp_output_dim,
            1,
            dim_feedforward=1024,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.monotonicity_mix_attn = nn.TransformerEncoderLayer(
            mlp_output_dim,
            1,
            dim_feedforward=1024,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.mix_attn = nn.TransformerEncoderLayer(
            mlp_output_dim,
            1,
            dim_feedforward=1024,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.pool = get_pool(pool_name, 1)(self.window_size * 3 + 3)
        self.dropout = nn.Dropout(dense_dropout)

        self.norm1 = nn.LayerNorm(mlp_output_dim)

        self.dense = DenseSequential(
            input_dims=mlp_output_dim,
            hidden_size=dense_hidden_size,
            dropout=dense_dropout,
            activation_name=activation_name,
        )

    def forward(
        self,
        examination: torch.Tensor,
        insulin,
        sugar,
        drug,
        normal_fes,
        monotonicity_fes,
        days,
    ):
        examination_proj = self.examination_mlp(examination)
        normal_fes_proj = self.normal_fes_mlp(normal_fes)
        monotonicity_fes_proj = self.monotonicity_fes_mlp(monotonicity_fes)

        insulin_proj = self.insulin_mlp(insulin)
        sugar_proj = self.sugar_mlp(sugar)
        drug_proj = self.drug_mlp(drug)

        day_embed = self.day_mlp(days)

        insulin_day_attn = self.insulin_attn(
            torch.cat([insulin_proj, day_embed], dim=2)
        )
        sugar_day_attn = self.sugar_attn(torch.cat([sugar_proj, day_embed], dim=2))
        drug_day_attn = self.drug_attn(torch.cat([drug_proj, day_embed], dim=2))

        normal_mix = torch.cat(
            [examination_proj, normal_fes_proj, insulin_day_attn, drug_day_attn], dim=1
        )
        monotonicity_mix = torch.cat([monotonicity_fes_proj, sugar_day_attn], dim=1)
        normal_mix = self.normal_mix_attn(normal_mix)
        monotonicity_mix = self.monotonicity_mix_attn(monotonicity_mix)

        mix_out = torch.cat((normal_mix, monotonicity_mix), dim=1)
        out = self.mix_attn(mix_out)

        out = self.pool(out.permute((0, 2, 1))).squeeze(2)
        out = self.norm1(out)
        out = self.dropout(out)
        out = self.dense(out)
        return out

    def unpack_data(self, data: dict):
        examination = data["f_examination"].unsqueeze(1)
        insulin = data["f_insulin"]
        sugar = data["f_sugar"]
        drug = data["f_drug"]

        days = data["f_day"]
        normal_fes = data["normal_features"].unsqueeze(1)
        monotonicity_fes = data["monotonicity_features"].unsqueeze(1)

        return (
            examination,
            insulin,
            sugar,
            drug,
            normal_fes,
            monotonicity_fes,
            days,
        )
