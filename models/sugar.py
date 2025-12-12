import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from models.base.utils import MLP, DenseSequential, get_pool, get_rnn


class SugarAttentionModel(nn.Module):
    def __init__(
        self,
        examination_size: int = 78,
        insulin_size: int = 12,
        sugar_size: int = 7,
        drug_size: int = 28,
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

        self.pool = get_pool(pool_name, 1)(self.window_size * 3 + 1)
        self.dropout = nn.Dropout(dense_dropout)

        self.norm1 = nn.LayerNorm(mlp_output_dim)

        self.dense = DenseSequential(
            input_dims=mlp_output_dim,
            hidden_size=dense_hidden_size,
            dropout=dense_dropout,
            activation_name=activation_name,
            output_dims=7,
        )

    def forward(
        self,
        examination: torch.Tensor,
        insulin: torch.Tensor,
        sugar: torch.Tensor,
        drug: torch.Tensor,
        days: torch.Tensor,
    ):
        examination_proj = self.examination_mlp(examination)

        insulin_proj = self.insulin_mlp(insulin)
        sugar_proj = self.sugar_mlp(sugar)
        drug_proj = self.drug_mlp(drug)

        day_embed = self.day_mlp(days)

        insulin_day_attn = self.insulin_attn(
            torch.cat([insulin_proj, day_embed], dim=2)
        )
        sugar_day_attn = self.sugar_attn(torch.cat([sugar_proj, day_embed], dim=2))
        drug_day_attn = self.drug_attn(torch.cat([drug_proj, day_embed], dim=2))
        mix = torch.cat(
            [examination_proj, insulin_day_attn, drug_day_attn, sugar_day_attn], dim=1
        )
        out = self.mix_attn(mix)
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

        return (
            examination,
            insulin,
            sugar,
            drug,
            days,
        )


class SugarAttentionFullModel(nn.Module):
    def __init__(
        self,
        examination_size: int = 78,
        insulin_size: int = 12,
        sugar_size: int = 7,
        drug_size: int = 28,
        temp_insulin_size: int = 24 * 3,
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
        self.temp_insulin_size = temp_insulin_size
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

        bf_tf_encode = partial(
            nn.TransformerEncoderLayer,
            dim_feedforward=1024,
            dropout=attn_dropout,
            batch_first=True,
        )
        day_size = 10
        self.day_mlp = _get_mlp_layer(1, day_size)
        self.examination_mlp = _get_mlp_layer(examination_size, mlp_output_dim)

        self.insulin_mlp = _get_mlp_layer(insulin_size, mlp_output_dim - day_size)
        self.sugar_mlp = _get_mlp_layer(sugar_size, mlp_output_dim - day_size)
        self.drug_mlp = _get_mlp_layer(drug_size, mlp_output_dim - day_size)

        self.temp_insulin_mlp = _get_mlp_layer(
            temp_insulin_size, mlp_output_dim - day_size
        )

        self.insulin_attn = bf_tf_encode(mlp_output_dim, attn_num_heads)
        self.temp_insulin_attn = bf_tf_encode(mlp_output_dim, attn_num_heads)
        self.sugar_attn = bf_tf_encode(mlp_output_dim, attn_num_heads)
        self.drug_attn = bf_tf_encode(mlp_output_dim, attn_num_heads)

        self.mix_attn = bf_tf_encode(mlp_output_dim, 1)
        self.pool = get_pool(pool_name, 1)(self.window_size * 4 + 1)
        self.dropout = nn.Dropout(dense_dropout)

        self.norm1 = nn.LayerNorm(mlp_output_dim)

        self.dense = DenseSequential(
            input_dims=mlp_output_dim,
            hidden_size=dense_hidden_size,
            dropout=dense_dropout,
            activation_name=activation_name,
            output_dims=sugar_size,
        )

    def forward(
        self,
        examination: torch.Tensor,
        insulin: torch.Tensor,
        temp_insulin: torch.Tensor,
        sugar: torch.Tensor,
        drug: torch.Tensor,
        days: torch.Tensor,
    ):
        examination_proj = self.examination_mlp(examination)

        insulin_proj = self.insulin_mlp(insulin)
        insulin_temp_proj = self.temp_insulin_mlp(temp_insulin)

        sugar_proj = self.sugar_mlp(sugar)
        drug_proj = self.drug_mlp(drug)

        day_embed = self.day_mlp(days)

        insulin_attn = self.insulin_attn(torch.cat([insulin_proj, day_embed], dim=2))
        temp_insulin_attn = self.temp_insulin_attn(
            torch.cat([insulin_temp_proj, day_embed], dim=2)
        )

        sugar_attn = self.sugar_attn(torch.cat([sugar_proj, day_embed], dim=2))
        drug_attn = self.drug_attn(torch.cat([drug_proj, day_embed], dim=2))
        mix = torch.cat(
            [examination_proj, insulin_attn, temp_insulin_attn, sugar_attn, drug_attn],
            dim=1,
        )
        out = self.mix_attn(mix)
        out = self.pool(out.permute((0, 2, 1))).squeeze(2)
        out = self.norm1(out)
        out = self.dropout(out)
        out = self.dense(out)
        return out

    def unpack_data(self, data: dict):
        days = data["day"]
        examination = data["examination"].unsqueeze(1)
        insulin = data["insulin"]
        # insulin[:, -1] = -1
        sugar = data["sugar"]
        # sugar[:, -1] = -1
        drug = data["drug"]
        # drug[:, -1] = -1

        temp_insulin = torch.reshape(
            data["temp_insulin"], (-1, self.window_size, 24 * 3)
        )

        return (
            examination,
            insulin,
            temp_insulin,
            sugar,
            drug,
            days,
        )


class SugarRnnAttentionFullModel(nn.Module):
    def __init__(
        self,
        examination_size: int = 78,
        insulin_size: int = 12,
        sugar_size: int = 7,
        drug_size: int = 28,
        temp_insulin_size: int = 24 * 3,
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
        self.temp_insulin_size = temp_insulin_size
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

        bf_tf_encode = partial(
            nn.TransformerEncoderLayer,
            dim_feedforward=1024,
            dropout=attn_dropout,
            batch_first=True,
        )
        day_size = 10
        self.day_mlp = _get_mlp_layer(1, day_size)
        self.examination_mlp = _get_mlp_layer(examination_size, mlp_output_dim)

        self.insulin_mlp = _get_mlp_layer(insulin_size, mlp_output_dim)
        self.sugar_mlp = _get_mlp_layer(sugar_size, mlp_output_dim)
        self.drug_mlp = _get_mlp_layer(drug_size, mlp_output_dim)

        self.temp_insulin_mlp = _get_mlp_layer(temp_insulin_size, mlp_output_dim)

        self.rnn = get_rnn("gru")(
            mlp_output_dim * 4,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
        )
        self.mix_attn = bf_tf_encode(mlp_output_dim, 1)
        self.pool = get_pool(pool_name, 1)(self.window_size + 1)
        self.dropout = nn.Dropout(dense_dropout)

        self.norm1 = nn.LayerNorm(mlp_output_dim)

        self.dense = DenseSequential(
            input_dims=mlp_output_dim,
            hidden_size=dense_hidden_size,
            dropout=dense_dropout,
            activation_name=activation_name,
            output_dims=sugar_size,
        )

    def forward(
        self,
        examination: torch.Tensor,
        insulin: torch.Tensor,
        temp_insulin: torch.Tensor,
        sugar: torch.Tensor,
        drug: torch.Tensor,
        days: torch.Tensor,
    ):
        examination_proj = self.examination_mlp(examination)

        insulin_proj = self.insulin_mlp(insulin)
        insulin_temp_proj = self.temp_insulin_mlp(temp_insulin)

        sugar_proj = self.sugar_mlp(sugar)
        drug_proj = self.drug_mlp(drug)
        time_data_cat = torch.cat(
            [
                insulin_proj,
                insulin_temp_proj,
                sugar_proj,
                drug_proj,
            ],
            dim=2,
        )
        out, _ = self.rnn(time_data_cat)

        mix = torch.cat([examination_proj, out], dim=1)
        out = self.mix_attn(mix)
        # print(out.shape)
        out = self.pool(out.permute((0, 2, 1))).squeeze(2)
        out = self.norm1(out)
        out = self.dropout(out)
        out = self.dense(out)
        return out

    def unpack_data(self, data: dict):
        days = data["day"]
        examination = data["examination"].unsqueeze(1)
        insulin = data["insulin"]
        # insulin[:, -1] = -1
        sugar = data["sugar"]
        # sugar[:, -1] = -1
        drug = data["drug"]
        # drug[:, -1] = -1

        temp_insulin = torch.reshape(
            data["temp_insulin"], (-1, self.window_size, 24 * 3)
        )

        return (
            examination,
            insulin,
            temp_insulin,
            sugar,
            drug,
            days,
        )


class SugarRnnAttentionFull24Model(nn.Module):
    def __init__(
        self,
        examination_size: int = 78,
        insulin_size: int = 4 * 9,
        sugar_size: int = 7,
        drug_size: int = 28,
        temp_insulin_size: int = 24 * 9,
        window_size: int = 10,
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
        self.temp_insulin_size = temp_insulin_size
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

        bf_tf_encode = partial(
            nn.TransformerEncoderLayer,
            dim_feedforward=1024,
            dropout=attn_dropout,
            batch_first=True,
        )
        day_size = 10
        self.day_mlp = _get_mlp_layer(1, day_size)

        self.examination_mlp = _get_mlp_layer(examination_size, mlp_output_dim)

        self.insulin_mlp = _get_mlp_layer(insulin_size, mlp_output_dim - day_size)
        self.sugar_mlp = _get_mlp_layer(sugar_size, mlp_output_dim - day_size)
        self.drug_mlp = _get_mlp_layer(drug_size, mlp_output_dim - day_size)
        self.temp_insulin_mlp = _get_mlp_layer(
            temp_insulin_size, mlp_output_dim - day_size
        )

        self.insulin_attn = bf_tf_encode(mlp_output_dim, attn_num_heads)
        self.insulin_temp_attn = bf_tf_encode(mlp_output_dim, attn_num_heads)
        self.sugar_attn = bf_tf_encode(mlp_output_dim, attn_num_heads)
        self.drug_attn = bf_tf_encode(mlp_output_dim, attn_num_heads)

        self.rnn = get_rnn("gru")(
            mlp_output_dim * 4,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
        )

        self.mix_attn = bf_tf_encode(mlp_output_dim, 1)

        self.pool = get_pool(pool_name, 1)(self.window_size * 4 + 1)
        self.dropout = nn.Dropout(dense_dropout)

        self.norm1 = nn.LayerNorm(mlp_output_dim)

        self.dense = DenseSequential(
            input_dims=mlp_output_dim,
            hidden_size=dense_hidden_size,
            dropout=dense_dropout,
            activation_name=activation_name,
            output_dims=1,
        )

    def forward(
        self,
        examination: torch.Tensor,
        insulin: torch.Tensor,
        temp_insulin: torch.Tensor,
        sugar: torch.Tensor,
        drug: torch.Tensor,
        days: torch.Tensor,
    ):
        examination_proj = self.examination_mlp(examination)
        insulin_proj = self.insulin_mlp(insulin)
        insulin_temp_proj = self.temp_insulin_mlp(temp_insulin)
        sugar_proj = self.sugar_mlp(sugar)
        drug_proj = self.drug_mlp(drug)
        day_embed = self.day_mlp(days)

        insulin_day_attn = self.insulin_attn(
            torch.cat([insulin_proj, day_embed], dim=2)
        )
        insulin_temp_day_attn = self.insulin_temp_attn(
            torch.cat([insulin_temp_proj, day_embed], dim=2)
        )
        sugar_day_attn = self.sugar_attn(torch.cat([sugar_proj, day_embed], dim=2))
        drug_day_attn = self.drug_attn(torch.cat([drug_proj, day_embed], dim=2))
        mix = torch.cat(
            [
                examination_proj,
                insulin_day_attn,
                insulin_temp_day_attn,
                drug_day_attn,
                sugar_day_attn,
            ],
            dim=1,
        )
        out = self.mix_attn(mix)
        out = self.pool(out.permute((0, 2, 1))).squeeze(2)
        out = self.norm1(out)
        out = self.dropout(out)
        out = self.dense(out)
        return out

    def unpack_data(self, data: dict):
        days = data["day_vec"]
        examination = data["examination_vec"].unsqueeze(1)
        insulin = data["insulin_vec_4_point"].reshape(-1, self.window_size, 4 * 9)
        sugar = data["sugar_vec_7_point"].reshape(-1, self.window_size, 7 * 1)
        drug = data["drug_vec_1_point"]
        temp_insulin = data["insulin_vec_temp"].reshape(-1, self.window_size, 24 * 9)

        return (
            examination,
            insulin,
            temp_insulin,
            sugar,
            drug,
            days,
        )


class SugarPredict7PointModel(nn.Module):
    def __init__(
        self,
        examination_size: int = 78,
        insulin_size: int = 9,
        sugar_size: int = 1,
        drug_size: int = 28,
        temp_insulin_size: int = 9,
        window_size: int = 10,
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
        self.temp_insulin_size = temp_insulin_size
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

        bf_tf_encode = partial(
            nn.TransformerEncoderLayer,
            dim_feedforward=1024,
            dropout=attn_dropout,
            batch_first=True,
        )
        day_size = 10
        self.day_mlp = _get_mlp_layer(1, day_size)

        self.examination_mlp = _get_mlp_layer(examination_size, mlp_output_dim)

        output_dim = mlp_output_dim - day_size
        self.insulin_mlp = _get_mlp_layer(insulin_size, output_dim)
        self.temp_insulin_mlp = _get_mlp_layer(temp_insulin_size, output_dim)
        self.sugar_mlp = _get_mlp_layer(sugar_size, output_dim)
        self.drug_mlp = _get_mlp_layer(drug_size, output_dim)

        self.insulin_attn = bf_tf_encode(mlp_output_dim, attn_num_heads)
        self.insulin_temp_attn = bf_tf_encode(mlp_output_dim, attn_num_heads)
        self.sugar_attn = bf_tf_encode(mlp_output_dim, attn_num_heads)
        self.drug_attn = bf_tf_encode(mlp_output_dim, attn_num_heads)

        self.rnn = get_rnn("gru")(
            mlp_output_dim * 5,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
        )

        self.mix_attn = bf_tf_encode(mlp_output_dim * 5, 1)

        self.pool = get_pool(pool_name, 1)(self.window_size * 4 + 1)
        self.dropout = nn.Dropout(dense_dropout)

        self.norm1 = nn.LayerNorm(mlp_output_dim)

        self.dense = DenseSequential(
            input_dims=256,
            hidden_size=dense_hidden_size,
            dropout=dense_dropout,
            activation_name=activation_name,
            output_dims=7,
        )

    def forward(
        self,
        examination: torch.Tensor,
        insulin: torch.Tensor,
        temp_insulin: torch.Tensor,
        sugar: torch.Tensor,
        drug: torch.Tensor,
        days: torch.Tensor,
    ):
        examination_proj = self.examination_mlp(examination)
        insulin_proj = self.insulin_mlp(insulin)
        insulin_temp_proj = self.temp_insulin_mlp(temp_insulin)
        sugar_proj = self.sugar_mlp(sugar)
        drug_proj = self.drug_mlp(drug)
        day_embed = self.day_mlp(days)

        insulin_day_attn = self.insulin_attn(
            torch.cat([insulin_proj, day_embed], dim=2)
        )
        insulin_temp_day_attn = self.insulin_temp_attn(
            torch.cat([insulin_temp_proj, day_embed], dim=2)
        )
        sugar_day_attn = self.sugar_attn(torch.cat([sugar_proj, day_embed], dim=2))
        drug_day_attn = self.drug_attn(torch.cat([drug_proj, day_embed], dim=2))
        examination_proj = examination_proj.repeat((1, 7 * self.window_size, 1))

        mix = torch.cat(
            [
                examination_proj,
                insulin_day_attn,
                insulin_temp_day_attn,
                drug_day_attn,
                sugar_day_attn,
            ],
            dim=2,
        )
        rnn_out, _ = self.rnn(mix)
        # out = self.mix_attn(mix)
        # print(rnn_out.shape)
        # print(mix.shape)
        # print(out.shape)
        # out = self.pool(out.permute((0, 2, 1))).squeeze(2)
        # out = self.norm1(out)
        # out = self.dropout(out)
        out = self.dense(rnn_out[:, -1, :])
        return out

    def unpack_data(self, data: dict):
        days = data["day_vec"].repeat((1, 1, 7)).reshape(-1, self.window_size * 7, 1)

        examination = data["examination_vec"].unsqueeze(1)
        insulin = data["insulin_vec_7_point"].reshape(
            -1, self.window_size * 7, self.insulin_size
        )
        sugar = data["sugar_vec_7_point"].reshape(
            -1, self.window_size * 7, self.sugar_size
        )
        drug = data["drug_vec_7_point"].reshape(
            -1, self.window_size * 7, self.drug_size
        )
        temp_insulin = data["insulin_vec_7_point_temp"].reshape(
            -1, self.window_size * 7, self.insulin_size
        )
        return (
            examination,
            insulin,
            temp_insulin,
            sugar,
            drug,
            days,
        )


class SugarPredict7PointAttnModel(nn.Module):
    def __init__(
        self,
        examination_size: int = 78,
        insulin_size: int = 9,
        sugar_size: int = 1,
        drug_size: int = 28,
        temp_insulin_size: int = 9,
        window_size: int = 10,
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
        self.temp_insulin_size = temp_insulin_size
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

        bf_tf_encode = partial(
            nn.TransformerEncoderLayer,
            dim_feedforward=1024,
            dropout=attn_dropout,
            batch_first=True,
        )
        day_size = 10
        self.day_mlp = _get_mlp_layer(1, day_size)

        self.examination_mlp = _get_mlp_layer(examination_size, mlp_output_dim)

        output_dim = mlp_output_dim - day_size
        self.insulin_mlp = _get_mlp_layer(insulin_size, output_dim)
        self.temp_insulin_mlp = _get_mlp_layer(temp_insulin_size, output_dim)
        self.sugar_mlp = _get_mlp_layer(sugar_size, output_dim)
        self.drug_mlp = _get_mlp_layer(drug_size, output_dim)

        self.insulin_attn = bf_tf_encode(mlp_output_dim, attn_num_heads)
        self.insulin_temp_attn = bf_tf_encode(mlp_output_dim, attn_num_heads)
        self.sugar_attn = bf_tf_encode(mlp_output_dim, attn_num_heads)
        self.drug_attn = bf_tf_encode(mlp_output_dim, attn_num_heads)

        self.mix_attn = bf_tf_encode(mlp_output_dim, 1)

        self.pool = get_pool(pool_name, 1)(self.window_size * 7 * 5)
        self.dropout = nn.Dropout(dense_dropout)

        self.norm1 = nn.LayerNorm(mlp_output_dim)

        self.dense = DenseSequential(
            input_dims=mlp_output_dim,
            hidden_size=dense_hidden_size,
            dropout=dense_dropout,
            activation_name=activation_name,
            output_dims=7,
        )

    def forward(
        self,
        examination: torch.Tensor,
        insulin: torch.Tensor,
        temp_insulin: torch.Tensor,
        sugar: torch.Tensor,
        drug: torch.Tensor,
        days: torch.Tensor,
    ):
        examination_proj = self.examination_mlp(examination)
        insulin_proj = self.insulin_mlp(insulin)
        insulin_temp_proj = self.temp_insulin_mlp(temp_insulin)
        sugar_proj = self.sugar_mlp(sugar)
        drug_proj = self.drug_mlp(drug)
        day_embed = self.day_mlp(days)

        insulin_day_attn = self.insulin_attn(
            torch.cat([insulin_proj, day_embed], dim=2)
        )
        insulin_temp_day_attn = self.insulin_temp_attn(
            torch.cat([insulin_temp_proj, day_embed], dim=2)
        )
        sugar_day_attn = self.sugar_attn(torch.cat([sugar_proj, day_embed], dim=2))
        drug_day_attn = self.drug_attn(torch.cat([drug_proj, day_embed], dim=2))
        examination_proj = examination_proj.repeat((1, 7 * self.window_size, 1))

        mix = torch.cat(
            [
                examination_proj,
                insulin_day_attn,
                insulin_temp_day_attn,
                drug_day_attn,
                sugar_day_attn,
            ],
            dim=1,
        )
        out = self.mix_attn(mix)
        # print(mix.shape)
        # print(out.shape)
        out = self.pool(out.permute((0, 2, 1))).squeeze(2)
        # print(out.shape)
        out = self.norm1(out)
        out = self.dropout(out)
        out = self.dense(out)
        return out

    def unpack_data(self, data: dict):
        days = data["day_vec"].repeat((1, 1, 7)).reshape(-1, self.window_size * 7, 1)

        examination = data["examination_vec"].unsqueeze(1)
        insulin = data["insulin_vec_7_point"].reshape(
            -1, self.window_size * 7, self.insulin_size
        )
        sugar = data["sugar_vec_7_point"].reshape(
            -1, self.window_size * 7, self.sugar_size
        )
        drug = data["drug_vec_7_point"].reshape(
            -1, self.window_size * 7, self.drug_size
        )
        temp_insulin = data["insulin_vec_7_point_temp"].reshape(
            -1, self.window_size * 7, self.insulin_size
        )
        return (
            examination,
            insulin,
            temp_insulin,
            sugar,
            drug,
            days,
        )


class SugarPredict7PointAttn2Model(nn.Module):
    def __init__(
        self,
        examination_size: int = 78,
        insulin_size: int = 9,
        sugar_size: int = 1,
        drug_size: int = 28,
        temp_insulin_size: int = 9,
        window_size: int = 10,
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
        self.temp_insulin_size = temp_insulin_size
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

        bf_tf_encode = partial(
            nn.TransformerEncoderLayer,
            dim_feedforward=1024,
            dropout=attn_dropout,
            batch_first=True,
        )
        day_size = 10
        self.day_mlp = _get_mlp_layer(1, day_size)

        self.examination_mlp = _get_mlp_layer(examination_size, mlp_output_dim)

        output_dim = mlp_output_dim - day_size
        self.insulin_mlp = _get_mlp_layer(insulin_size * 7, output_dim)
        self.temp_insulin_mlp = _get_mlp_layer(temp_insulin_size * 7, output_dim)
        self.sugar_mlp = _get_mlp_layer(sugar_size * 7, output_dim)
        self.drug_mlp = _get_mlp_layer(drug_size * 7, output_dim)

        self.insulin_attn = bf_tf_encode(mlp_output_dim, attn_num_heads)
        self.insulin_temp_attn = bf_tf_encode(mlp_output_dim, attn_num_heads)
        self.sugar_attn = bf_tf_encode(mlp_output_dim, attn_num_heads)
        self.drug_attn = bf_tf_encode(mlp_output_dim, attn_num_heads)

        self.mix_attn = bf_tf_encode(mlp_output_dim, 1)

        self.pool = get_pool(pool_name, 1)(self.window_size * 5)
        self.dropout = nn.Dropout(dense_dropout)

        self.norm1 = nn.LayerNorm(mlp_output_dim)

        self.dense = DenseSequential(
            input_dims=mlp_output_dim,
            hidden_size=dense_hidden_size,
            dropout=dense_dropout,
            activation_name=activation_name,
            output_dims=7,
        )

    def forward(
        self,
        examination: torch.Tensor,
        insulin: torch.Tensor,
        temp_insulin: torch.Tensor,
        sugar: torch.Tensor,
        drug: torch.Tensor,
        days: torch.Tensor,
    ):
        examination_proj = self.examination_mlp(examination)
        insulin_proj = self.insulin_mlp(insulin)
        insulin_temp_proj = self.temp_insulin_mlp(temp_insulin)
        sugar_proj = self.sugar_mlp(sugar)
        drug_proj = self.drug_mlp(drug)
        day_embed = self.day_mlp(days)

        insulin_day_attn = self.insulin_attn(
            torch.cat([insulin_proj, day_embed], dim=2)
        )
        insulin_temp_day_attn = self.insulin_temp_attn(
            torch.cat([insulin_temp_proj, day_embed], dim=2)
        )
        sugar_day_attn = self.sugar_attn(torch.cat([sugar_proj, day_embed], dim=2))
        drug_day_attn = self.drug_attn(torch.cat([drug_proj, day_embed], dim=2))
        examination_proj = examination_proj.repeat((1, self.window_size, 1))

        mix = torch.cat(
            [
                examination_proj,
                insulin_day_attn,
                insulin_temp_day_attn,
                drug_day_attn,
                sugar_day_attn,
            ],
            dim=1,
        )
        out = self.mix_attn(mix)
        # print(mix.shape)
        # print(out.shape)
        out = self.pool(out.permute((0, 2, 1))).squeeze(2)
        # print(out.shape)
        out = self.norm1(out)
        out = self.dropout(out)
        out = self.dense(out)
        return out

    def unpack_data(self, data: dict):
        days = data[
            "day_vec"
        ]  # .repeat((1, 1, 7)).reshape(-1, self.window_size * 7, 1)

        examination = data["examination_vec"].unsqueeze(1)

        insulin = data["insulin_vec_7_point"].reshape(
            -1, self.window_size, self.insulin_size * 7
        )
        sugar = data["sugar_vec_7_point"].reshape(
            -1, self.window_size, self.sugar_size * 7
        )
        drug = data["drug_vec_7_point"].reshape(
            -1, self.window_size, self.drug_size * 7
        )
        temp_insulin = data["insulin_vec_7_point_temp"].reshape(
            -1, self.window_size, self.insulin_size * 7
        )
        return (
            examination,
            insulin,
            temp_insulin,
            sugar,
            drug,
            days,
        )


class SugarPredict7PointAttnRnnModel(nn.Module):
    def __init__(
        self,
        examination_size: int = 78,
        insulin_size: int = 9,
        sugar_size: int = 1,
        drug_size: int = 28,
        points: int = 7,
        window_size: int = 10,
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

        self.points = points
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

        bf_tf_encode = partial(
            nn.TransformerEncoderLayer,
            dim_feedforward=1024,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.cat_data_mlp = MLP(
            num_inputs=sum([insulin_size, insulin_size, sugar_size, drug_size, 1]),
            num_hidden=1024,
            num_outputs=mlp_output_dim,
            num_block=mlp_encode_blocks,
            dropout=mlp_encode_dropout,
            activation_name=activation_name,
        )
        self.examination_mlp = _get_mlp_layer(examination_size, mlp_output_dim)

        self.cat_data_attn = bf_tf_encode(mlp_output_dim, attn_num_heads)
        self.cat_data_rnn = get_rnn("gru")(
            mlp_output_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
        )
        self.mix_attn = bf_tf_encode(mlp_output_dim * 3, 1)

        self.pool = get_pool(pool_name, 1)(window_size * points)
        self.dropout = nn.Dropout(dense_dropout)

        self.norm = nn.LayerNorm(mlp_output_dim * 3)

        self.dense = DenseSequential(
            input_dims=mlp_output_dim * 3,
            hidden_size=dense_hidden_size,
            dropout=dense_dropout,
            activation_name=activation_name,
            output_dims=7,
        )

    def forward(
        self,
        examination: torch.Tensor,
        insulin: torch.Tensor,
        temp_insulin: torch.Tensor,
        sugar: torch.Tensor,
        drug: torch.Tensor,
        days: torch.Tensor,
    ):
        examination_proj = self.examination_mlp(examination)
        examination_proj = examination_proj.repeat((1, 70, 1))
        time_data_cat = torch.cat(
            [
                insulin,
                temp_insulin,
                sugar,
                drug,
                days,
            ],
            dim=2,
        )
        # print(examination_proj.shape)
        # print(time_data_cat.shape)
        time_data_out = self.cat_data_mlp(time_data_cat)
        # print(time_data_out.shape)
        time_data_attn_out = self.cat_data_attn(time_data_out)
        # print(time_data_attn_out.shape)
        time_data_rnn_out, _ = self.cat_data_rnn(time_data_out)
        # print(time_data_rnn_out.shape)

        time_data = torch.cat([time_data_attn_out, time_data_rnn_out], dim=2)
        # print(time_data.shape)
        out = torch.cat([time_data, examination_proj], dim=2)
        # print(out.shape)
        out = self.mix_attn(out)
        # print(out.shape)

        out = self.pool(out.permute((0, 2, 1))).squeeze(2)
        # print(out.shape)
        out = self.norm(out)
        out = self.dropout(out)
        out = self.dense(out)
        return out

    def unpack_data(self, data: dict):
        days = data["day_vec"].repeat((1, 1, 7)).reshape(-1, self.window_size * 7, 1)

        examination = data["examination_vec"].unsqueeze(1)

        insulin = data["insulin_vec_7_point"].reshape(
            -1, self.window_size * 7, self.insulin_size
        )
        sugar = data["sugar_vec_7_point"].reshape(
            -1, self.window_size * 7, self.sugar_size
        )
        drug = data["drug_vec_7_point"].reshape(
            -1, self.window_size * 7, self.drug_size
        )
        temp_insulin = data["insulin_vec_7_point_temp"].reshape(
            -1, self.window_size * 7, self.insulin_size
        )
        return (
            examination,
            insulin,
            temp_insulin,
            sugar,
            drug,
            days,
        )


class SugarPredict7PointAttnRnn2Model(nn.Module):
    def __init__(
        self,
        examination_size: int = 78,
        insulin_size: int = 9,
        sugar_size: int = 1,
        drug_size: int = 28,
        points: int = 7,
        window_size: int = 10,
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

        self.points = points
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

        bf_tf_encode = partial(
            nn.TransformerEncoderLayer,
            dim_feedforward=1024,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.cat_data_mlp = MLP(
            num_inputs=sum([insulin_size, insulin_size, sugar_size, drug_size, 1]),
            num_hidden=1024,
            num_outputs=mlp_output_dim,
            num_block=mlp_encode_blocks,
            dropout=mlp_encode_dropout,
            activation_name=activation_name,
        )
        self.examination_mlp = _get_mlp_layer(examination_size, mlp_output_dim)

        self.cat_data_attn = bf_tf_encode(mlp_output_dim, attn_num_heads)
        self.cat_data_rnn = get_rnn("gru")(
            mlp_output_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
        )
        self.mix_attn = bf_tf_encode(mlp_output_dim, 1)

        self.pool = get_pool(pool_name, 1)(window_size * points * 2 + 1)
        self.dropout = nn.Dropout(dense_dropout)

        self.norm = nn.LayerNorm(mlp_output_dim)

        self.dense = DenseSequential(
            input_dims=mlp_output_dim,
            hidden_size=dense_hidden_size,
            dropout=dense_dropout,
            activation_name=activation_name,
            output_dims=7,
        )

    def forward(
        self,
        examination: torch.Tensor,
        insulin: torch.Tensor,
        temp_insulin: torch.Tensor,
        sugar: torch.Tensor,
        drug: torch.Tensor,
        days: torch.Tensor,
    ):
        examination_proj = self.examination_mlp(examination)

        time_data_cat = torch.cat(
            [
                insulin,
                temp_insulin,
                sugar,
                drug,
                days,
            ],
            dim=2,
        )
        # print(examination_proj.shape)
        # print(time_data_cat.shape)
        time_data_out = self.cat_data_mlp(time_data_cat)
        # print(time_data_out.shape)
        time_data_attn_out = self.cat_data_attn(time_data_out)
        # print(time_data_attn_out.shape)
        time_data_rnn_out, _ = self.cat_data_rnn(time_data_out)
        # print(time_data_rnn_out.shape)

        time_data = torch.cat([time_data_attn_out, time_data_rnn_out], dim=1)
        # print(time_data.shape)
        out = torch.cat([time_data, examination_proj], dim=1)
        # print(out.shape)
        out = self.mix_attn(out)
        # print(out.shape)

        out = self.pool(out.permute((0, 2, 1))).squeeze(2)
        # print(out.shape)
        out = self.norm(out)
        out = self.dropout(out)
        out = self.dense(out)
        return out

    def unpack_data(self, data: dict):
        days = data["day_vec"].repeat((1, 1, 7)).reshape(-1, self.window_size * 7, 1)

        examination = data["examination_vec"].unsqueeze(1)

        insulin = data["insulin_vec_7_point"].reshape(
            -1, self.window_size * 7, self.insulin_size
        )
        sugar = data["sugar_vec_7_point"].reshape(
            -1, self.window_size * 7, self.sugar_size
        )
        drug = data["drug_vec_7_point"].reshape(
            -1, self.window_size * 7, self.drug_size
        )
        temp_insulin = data["insulin_vec_7_point_temp"].reshape(
            -1, self.window_size * 7, self.insulin_size
        )
        return (
            examination,
            insulin,
            temp_insulin,
            sugar,
            drug,
            days,
        )


class SugarPredict7Point2Model(nn.Module):
    def __init__(
        self,
        examination_size: int = 78,
        insulin_size: int = 9,
        sugar_size: int = 1,
        drug_size: int = 28,
        points: int = 24,
        window_size: int = 10,
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

        self.points = points
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

        bf_tf_encode = partial(
            nn.TransformerEncoderLayer,
            dim_feedforward=1024,
            dropout=attn_dropout,
            batch_first=True,
        )
        day_size = 10
        self.day_mlp = _get_mlp_layer(1, day_size)

        self.examination_mlp = _get_mlp_layer(examination_size, mlp_output_dim)

        output_dim = mlp_output_dim - day_size
        self.insulin_mlp = _get_mlp_layer(insulin_size, output_dim)
        self.temp_insulin_mlp = _get_mlp_layer(insulin_size, output_dim)
        self.sugar_mlp = _get_mlp_layer(sugar_size, output_dim)
        self.drug_mlp = _get_mlp_layer(drug_size, output_dim)

        self.insulin_attn = bf_tf_encode(mlp_output_dim, attn_num_heads)
        self.insulin_temp_attn = bf_tf_encode(mlp_output_dim, attn_num_heads)
        self.sugar_attn = bf_tf_encode(mlp_output_dim, attn_num_heads)
        self.drug_attn = bf_tf_encode(mlp_output_dim, attn_num_heads)

        self.rnn = get_rnn("gru")(
            mlp_output_dim * 5,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
        )

        self.mix_attn = bf_tf_encode(mlp_output_dim * 5, 1)

        self.pool = get_pool(pool_name, 1)(self.window_size * self.points)
        self.dropout = nn.Dropout(dense_dropout)

        self.norm1 = nn.LayerNorm(mlp_output_dim)

        self.dense = DenseSequential(
            input_dims=256 + mlp_output_dim * 5,
            hidden_size=dense_hidden_size,
            dropout=dense_dropout,
            activation_name=activation_name,
            output_dims=7,
        )

    def forward(
        self,
        examination: torch.Tensor,
        insulin: torch.Tensor,
        temp_insulin: torch.Tensor,
        sugar: torch.Tensor,
        drug: torch.Tensor,
        days: torch.Tensor,
    ):
        examination_proj = self.examination_mlp(examination)
        insulin_proj = self.insulin_mlp(insulin)
        insulin_temp_proj = self.temp_insulin_mlp(temp_insulin)
        sugar_proj = self.sugar_mlp(sugar)
        drug_proj = self.drug_mlp(drug)
        day_embed = self.day_mlp(days)

        insulin_day_attn = self.insulin_attn(
            torch.cat([insulin_proj, day_embed], dim=2)
        )
        insulin_temp_day_attn = self.insulin_temp_attn(
            torch.cat([insulin_temp_proj, day_embed], dim=2)
        )
        sugar_day_attn = self.sugar_attn(torch.cat([sugar_proj, day_embed], dim=2))
        drug_day_attn = self.drug_attn(torch.cat([drug_proj, day_embed], dim=2))
        examination_proj = examination_proj.repeat(
            (1, self.points * self.window_size, 1)
        )

        mix = torch.cat(
            [
                examination_proj,
                insulin_day_attn,
                insulin_temp_day_attn,
                drug_day_attn,
                sugar_day_attn,
            ],
            dim=2,
        )
        # print(mix.shape)
        rnn_out, _ = self.rnn(mix)
        attn_out = self.mix_attn(mix)
        # print(rnn_out.shape)
        # print(attn_out.shape)
        rnn_out = rnn_out[:, -1, :]
        attn_out = self.pool(attn_out.permute((0, 2, 1))).squeeze(2)
        # print(rnn_out.shape)
        # print(attn_out.shape)
        # out = self.norm1(out)
        # out = self.dropout(out)

        out = torch.cat([attn_out, rnn_out], dim=1)
        # print(out.shape)
        out = self.dense(out)
        return out

    def unpack_data(self, data: dict):
        days = (
            data["day_vec"]
            .repeat((1, 1, self.points))
            .reshape(-1, self.window_size * self.points, 1)
        )

        examination = data["examination_vec"].unsqueeze(1)
        insulin = data[f"insulin_vec_{self.points}_point"].reshape(
            -1, self.window_size * self.points, self.insulin_size
        )
        sugar = data[f"sugar_vec_{self.points}_point"].reshape(
            -1, self.window_size * self.points, self.sugar_size
        )
        drug = data[f"drug_vec_{self.points}_point"].reshape(
            -1, self.window_size * self.points, self.drug_size
        )
        temp_insulin = data[f"insulin_{self.points}_point_vec_temp"].reshape(
            -1, self.window_size * self.points, self.insulin_size
        )
        return (
            examination,
            insulin,
            temp_insulin,
            sugar,
            drug,
            days,
        )


class SugarPredict7Point3Model(nn.Module):
    def __init__(
        self,
        examination_size: int = 78,
        insulin_size: int = 9,
        sugar_size: int = 1,
        drug_size: int = 28,
        temp_insulin_size: int = 9,
        window_size: int = 10,
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
        self.temp_insulin_size = temp_insulin_size
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

        bf_tf_encode = partial(
            nn.TransformerEncoderLayer,
            dim_feedforward=1024,
            dropout=attn_dropout,
            batch_first=True,
        )
        day_size = 10
        self.day_mlp = _get_mlp_layer(1, day_size)

        self.examination_mlp = _get_mlp_layer(examination_size, mlp_output_dim)

        output_dim = mlp_output_dim - day_size
        self.insulin_mlp = _get_mlp_layer(insulin_size, output_dim)
        self.temp_insulin_mlp = _get_mlp_layer(temp_insulin_size, output_dim)
        self.sugar_mlp = _get_mlp_layer(sugar_size, output_dim)
        self.drug_mlp = _get_mlp_layer(drug_size, output_dim)

        self.insulin_attn = bf_tf_encode(mlp_output_dim, attn_num_heads)
        self.insulin_temp_attn = bf_tf_encode(mlp_output_dim, attn_num_heads)
        self.sugar_attn = bf_tf_encode(mlp_output_dim, attn_num_heads)
        self.drug_attn = bf_tf_encode(mlp_output_dim, attn_num_heads)

        self.rnn = get_rnn("gru")(
            mlp_output_dim * 5,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
        )

        self.mix_attn = bf_tf_encode(mlp_output_dim * 5, 1)

        self.pool = get_pool(pool_name, 1)(self.window_size * 4 + 1)
        self.dropout = nn.Dropout(dense_dropout)

        self.norm1 = nn.LayerNorm(mlp_output_dim)

        self.dense = DenseSequential(
            input_dims=256,
            hidden_size=dense_hidden_size,
            dropout=dense_dropout,
            activation_name=activation_name,
            output_dims=7,
        )

    def forward(
        self,
        examination: torch.Tensor,
        insulin: torch.Tensor,
        temp_insulin: torch.Tensor,
        sugar: torch.Tensor,
        drug: torch.Tensor,
        days: torch.Tensor,
    ):
        examination_proj = self.examination_mlp(examination)
        insulin_proj = self.insulin_mlp(insulin)
        insulin_temp_proj = self.temp_insulin_mlp(temp_insulin)
        sugar_proj = self.sugar_mlp(sugar)
        drug_proj = self.drug_mlp(drug)
        day_embed = self.day_mlp(days)

        insulin_day_attn = self.insulin_attn(
            torch.cat([insulin_proj, day_embed], dim=2)
        )
        insulin_temp_day_attn = self.insulin_temp_attn(
            torch.cat([insulin_temp_proj, day_embed], dim=2)
        )
        sugar_day_attn = self.sugar_attn(torch.cat([sugar_proj, day_embed], dim=2))
        drug_day_attn = self.drug_attn(torch.cat([drug_proj, day_embed], dim=2))
        examination_proj = examination_proj.repeat((1, 7 * self.window_size, 1))

        mix = torch.cat(
            [
                examination_proj,
                insulin_day_attn,
                insulin_temp_day_attn,
                drug_day_attn,
                sugar_day_attn,
            ],
            dim=2,
        )
        rnn_out, _ = self.rnn(mix)
        out = self.dense(rnn_out[:, -1, :])
        return out

    def unpack_data(self, data: dict):
        days = data["day_vec"].repeat((1, 1, 7)).reshape(-1, self.window_size * 7, 1)

        examination = data["examination_vec"].unsqueeze(1)
        insulin = data["insulin_vec_7_point"].reshape(
            -1, self.window_size * 7, self.insulin_size
        )
        sugar = data["sugar_vec_7_point"].reshape(
            -1, self.window_size * 7, self.sugar_size
        )
        drug = data["drug_vec_7_point"].reshape(
            -1, self.window_size * 7, self.drug_size
        )
        temp_insulin = data["insulin_vec_7_point_temp"].reshape(
            -1, self.window_size * 7, self.insulin_size
        )
        return (
            examination,
            insulin,
            temp_insulin,
            sugar,
            drug,
            days,
        )


class SugarPredict7PointMixModel(nn.Module):
    def __init__(
        self,
        examination_size: int = 78,
        insulin_size: int = 9,
        sugar_size: int = 1,
        drug_size: int = 28,
        points: int = 7,
        window_size: int = 10,
        mlp_encode_dilation: int = 4,  # 2,3,4,5,6,7,8
        mlp_output_dim: int = 128,  # 64, 128, 256
        mlp_encode_blocks: int = 3,  # 1,2,3
        mlp_encode_dropout: float = 0,  # 0.1-0.6
        dense_hidden_size: int = 128,  # 64, 128, 256
        dense_dropout: float = 0.5,  # 0 - 0.5
        pool_name: str = "max",  # max, avg
        activation_name: str = "relu",
        attn_num_heads: int = 8,  # 2,4,8,16
        attn_dropout: float = 0,
        rnn_name: str = "gru",
        rnn_hidden_size: int = 128,
        rnn_num_layers: int = 2,
        target_sugar_dim: int = 7,
    ):
        super().__init__()
        self.examination_size = examination_size
        self.insulin_size = insulin_size
        self.sugar_size = sugar_size
        self.drug_size = drug_size

        self.points = points
        self.window_size = window_size
        self.day_size = 10

        self.rnn_name = rnn_name
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.target_sugar_dim = target_sugar_dim

        def _get_mlp_layer(size: int, output_size: int) -> MLP:
            return MLP(
                num_inputs=size,
                num_hidden=size * mlp_encode_dilation,
                num_outputs=output_size,
                num_block=mlp_encode_blocks,
                dropout=mlp_encode_dropout,
                activation_name=activation_name,
            )

        def _get_attn_layer(dims: int, heads: int):
            return nn.TransformerEncoderLayer(
                dims,
                heads,
                dim_feedforward=2048,
                dropout=attn_dropout,
                batch_first=True,
            )

        self.day_mlp = _get_mlp_layer(1, self.day_size)

        self.examination_mlp = _get_mlp_layer(examination_size, mlp_output_dim)

        output_dim = mlp_output_dim - self.day_size
        self.insulin_mlp = _get_mlp_layer(insulin_size, output_dim)
        self.temp_insulin_mlp = _get_mlp_layer(insulin_size, output_dim)
        self.sugar_mlp = _get_mlp_layer(sugar_size, output_dim)
        self.drug_mlp = _get_mlp_layer(drug_size, output_dim)
        self.cat_data_mlp = _get_mlp_layer(
            sum(
                [examination_size, insulin_size, insulin_size, sugar_size, drug_size, 1]
            ),
            mlp_output_dim,
        )

        self.insulin_attn = _get_attn_layer(mlp_output_dim, attn_num_heads)
        self.insulin_temp_attn = _get_attn_layer(mlp_output_dim, attn_num_heads)
        self.sugar_attn = _get_attn_layer(mlp_output_dim, attn_num_heads)
        self.drug_attn = _get_attn_layer(mlp_output_dim, attn_num_heads)
        self.cat_data_attn = _get_attn_layer(mlp_output_dim, attn_num_heads)

        self.rnn = get_rnn(rnn_name)(
            mlp_output_dim * 5,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
        )
        self.cat_data_rnn = get_rnn(rnn_name)(
            mlp_output_dim,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
        )
        self.mix_attn = _get_attn_layer(mlp_output_dim * 5, attn_num_heads)

        self.pool = get_pool(pool_name, 1)(self.window_size * self.points)
        self.dropout = nn.Dropout(dense_dropout)

        self.dense = DenseSequential(
            input_dims=mlp_output_dim * 6 + rnn_hidden_size * 2,
            hidden_size=dense_hidden_size,
            dropout=dense_dropout,
            activation_name=activation_name,
            output_dims=target_sugar_dim,
        )

    def cat_mlp_forward(
        self,
        examination: torch.Tensor,
        insulin: torch.Tensor,
        temp_insulin: torch.Tensor,
        sugar: torch.Tensor,
        drug: torch.Tensor,
        days: torch.Tensor,
    ):
        examination_rps = examination.repeat((1, self.window_size * self.points, 1))
        time_data_cat = torch.cat(
            [
                examination_rps,
                insulin,
                temp_insulin,
                sugar,
                drug,
                days,
            ],
            dim=2,
        )
        time_data_proj = self.cat_data_mlp(time_data_cat)

        time_data_rnn_out, _ = self.cat_data_rnn(time_data_proj)
        time_data_attn_out = self.cat_data_attn(time_data_proj)

        time_data_rnn_out = time_data_rnn_out[:, -1, :]
        time_data_attn_out = self.pool(time_data_attn_out.permute((0, 2, 1))).squeeze(2)

        out = torch.cat([time_data_attn_out, time_data_rnn_out], dim=1)
        return out

    def forward(
        self,
        examination: torch.Tensor,
        insulin: torch.Tensor,
        temp_insulin: torch.Tensor,
        sugar: torch.Tensor,
        drug: torch.Tensor,
        days: torch.Tensor,
    ):

        day_embed = self.day_mlp(days)

        examination_proj = self.examination_mlp(examination)
        insulin_proj = self.insulin_mlp(insulin)
        insulin_temp_proj = self.temp_insulin_mlp(temp_insulin)
        sugar_proj = self.sugar_mlp(sugar)
        drug_proj = self.drug_mlp(drug)

        insulin_day_attn = self.insulin_attn(
            torch.cat([insulin_proj, day_embed], dim=2)
        )
        insulin_temp_day_attn = self.insulin_temp_attn(
            torch.cat([insulin_temp_proj, day_embed], dim=2)
        )
        sugar_day_attn = self.sugar_attn(torch.cat([sugar_proj, day_embed], dim=2))
        drug_day_attn = self.drug_attn(torch.cat([drug_proj, day_embed], dim=2))

        mix = torch.cat(
            [
                examination_proj.repeat((1, self.points * self.window_size, 1)),
                insulin_day_attn,
                insulin_temp_day_attn,
                drug_day_attn,
                sugar_day_attn,
            ],
            dim=2,
        )

        # rnn_hidden_size
        rnn_out, _ = self.rnn(mix)
        # mlp_output_dim * 5
        attn_out = self.mix_attn(mix)

        attn_out = self.pool(attn_out.permute((0, 2, 1))).squeeze(2)

        # mlp_output_dim * 5 + rnn_hidden_size
        out = torch.cat([attn_out, rnn_out[:, -1, :]], dim=1)

        # mlp_output_dim + rnn_hidden_size
        out2 = self.cat_mlp_forward(
            examination,
            insulin,
            temp_insulin,
            sugar,
            drug,
            days,
        )

        mix_out = torch.cat([out, out2], dim=1)
        out = self.dense(mix_out)
        return out

    def unpack_data(self, data: dict):
        days = (
            data["day_vec"]
            .repeat((1, 1, self.points))
            .reshape(-1, self.window_size * self.points, 1)
        )

        examination = data["examination_vec"].unsqueeze(1)
        insulin = data[f"insulin_vec_{self.points}_point"].reshape(
            -1, self.window_size * self.points, self.insulin_size
        )
        sugar = data[f"sugar_vec_{self.points}_point"].reshape(
            -1, self.window_size * self.points, self.sugar_size
        )
        drug = data[f"drug_vec_{self.points}_point"].reshape(
            -1, self.window_size * self.points, self.drug_size
        )
        temp_insulin = data[f"insulin_{self.points}_point_vec_temp"].reshape(
            -1, self.window_size * self.points, self.insulin_size
        )
        return (
            examination,
            insulin,
            temp_insulin,
            sugar,
            drug,
            days,
        )
