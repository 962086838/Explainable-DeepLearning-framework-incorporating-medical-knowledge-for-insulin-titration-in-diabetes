import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from models.base.utils import MLP, DenseSequential, get_pool, get_rnn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, MultiheadAttention
from torch.nn import Transformer
from torch import Tensor


class MixPredictMaskModel(nn.Module):
    def __init__(
        self,
        examination_size: int = 78,
        insulin_size: int = 9,
        sugar_size: int = 1,
        drug_size: int = 28,
        points: int = 7,
        window_size: int = 10,
        mlp_encode_dilation: int = 4,  # 2,3,4,5,6,7,8
        mlp_output_dim: int = 256,  # 64, 128, 256
        mlp_encode_blocks: int = 5,  # 1,2,3
        mlp_encode_dropout: float = 0,  # 0.1-0.6
        dense_hidden_size: int = 128,  # 64, 128, 256
        dense_dropout: float = 0.5,  # 0 - 0.5
        pool_name: str = "max",  # max, avg
        activation_name: str = "relu",
        attn_num_heads: int = 8,  # 2,4,8,16
        attn_dropout: float = 0,
        attn_dim: int = 1024,  # 256, 512, 1024,
        rnn_name: str = "gru",
        rnn_hidden_size: int = 256,
        rnn_num_layers: int = 2,
        # use_weather: bool = False,
    ):
        super().__init__()
        # self.use_weather = use_weather
        day_size = 1 + 1
        # if use_weather:
        # day_size += 4

        self.examination_size = examination_size
        self.insulin_size = insulin_size
        self.sugar_size = sugar_size
        self.drug_size = drug_size
        self.day_size = day_size
        n_layers = 4
        self.points = points
        self.window_size = window_size
        self.total_length = points * window_size

        self.rnn_name = rnn_name
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers

        def _get_mlp_layer(size: int, output_size: int) -> MLP:
            return MLP(
                num_inputs=size,
                num_hidden=size * mlp_encode_dilation,
                num_outputs=output_size,
                num_block=mlp_encode_blocks,
                dropout=mlp_encode_dropout,
                activation_name=activation_name,
            )

        def _get_mlp_fix_output_size_layer(size: int) -> MLP:
            return _get_mlp_layer(size, mlp_output_dim)

        # def _get_transformer_encoder(dims: int, heads: int):
        #     return nn.Transformer(
        #         dims,
        #         heads,
        #         # dim_feedforward=2048,
        #         # dropout=attn_dropout,
        #         batch_first=True,
        #     )

        def _get_transformer_encoder(dims: int, heads: int):
            encoder_layers = TransformerEncoderLayer(
                dims,
                heads,
                dim_feedforward=attn_dim,
                dropout=attn_dropout,
                batch_first=True,
            )
            norm = nn.LayerNorm(dims)
            transformer_encoder = TransformerEncoder(encoder_layers, n_layers, norm)
            return transformer_encoder

        def _get_transformer(dims: int, heads: int):

            return nn.Transformer(
                dims,
                heads,
                dim_feedforward=attn_dim,
                dropout=attn_dropout,
                # dropout=0,
                batch_first=True,
            )

        def _get_transformer_encoder_fix_params():
            return _get_transformer_encoder(mlp_output_dim, attn_num_heads)

        self.day_encoder = _get_mlp_fix_output_size_layer(day_size)

        self.examination_encoder = _get_mlp_fix_output_size_layer(examination_size)
        self.insulin_encoder = _get_mlp_fix_output_size_layer(insulin_size)
        self.sugar_encoder = _get_mlp_fix_output_size_layer(sugar_size)
        self.drug_encoder = _get_mlp_fix_output_size_layer(drug_size)

        self.insulin_transformer_encoder = _get_transformer_encoder_fix_params()
        self.insulin_temp_transformer_encoder = _get_transformer_encoder_fix_params()
        self.sugar_transformer_encoder = _get_transformer_encoder_fix_params()
        self.drug_transformer_encoder = _get_transformer_encoder_fix_params()

        self.data_cat_mlp = _get_mlp_layer(
            sum(
                [
                    examination_size,
                    insulin_size,
                    insulin_size,
                    sugar_size,
                    drug_size,
                    day_size,
                ]
            ),
            mlp_output_dim,
        )
        self.cat_mlp_rnn = get_rnn(rnn_name)(
            mlp_output_dim,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
        )
        # self.cat_mlp_attn = _get_transformer(mlp_output_dim, 8)
        self.cat_mlp_attn = _get_transformer_encoder(mlp_output_dim, 1)
        self.mlp_cat_rnn = get_rnn(rnn_name)(
            mlp_output_dim * 6,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
        )
        # self.mlp_cat_attn = _get_transformer(mlp_output_dim * 6, 8)
        self.mlp_cat_attn = _get_transformer_encoder(mlp_output_dim * 6, 1)
        self.pool = get_pool(pool_name, 1)(self.total_length)

        # self.pos_encoder = PositionalEncoding(mlp_output_dim * 6, 0, 7 * 10)

        self.sugar_predict_dense = DenseSequential(
            input_dims=mlp_output_dim * 7 + rnn_hidden_size * 2,
            hidden_size=dense_hidden_size,
            dropout=dense_dropout,
            activation_name=activation_name,
            output_dims=7,
        )
        self.insulin_predict_dense = DenseSequential(
            input_dims=mlp_output_dim * 7 + rnn_hidden_size * 2,
            hidden_size=dense_hidden_size,
            dropout=dense_dropout,
            activation_name=activation_name,
            output_dims=7,
        )

    def forward(
        self,
        examination: Tensor,
        insulin: Tensor,
        temp_insulin: Tensor,
        sugar: Tensor,
        drug: Tensor,
        days: Tensor,
        mask: Tensor = None,
    ) -> Tensor:
        return self.sugar_forward(
            examination, insulin, temp_insulin, sugar, drug, days, mask
        )

    def sugar_forward(
        self,
        examination: Tensor,
        insulin: Tensor,
        temp_insulin: Tensor,
        sugar: Tensor,
        drug: Tensor,
        days: Tensor,
        mask: Tensor = None,
    ) -> Tensor:
        out1 = self._mlp_cat_mix_forward(
            examination, insulin, temp_insulin, sugar, drug, days, mask
        )
        out2 = self._cat_mlp_mix_forward(
            examination, insulin, temp_insulin, sugar, drug, days, mask
        )
        out = torch.cat([out1, out2], dim=1)
        out = self.sugar_predict_dense(out)
        return out

    def insulin_forward(
        self,
        examination: Tensor,
        insulin: Tensor,
        temp_insulin: Tensor,
        sugar: Tensor,
        drug: Tensor,
        days: Tensor,
        mask: Tensor = None,
    ) -> Tensor:
        out1 = self._mlp_cat_mix_forward(
            examination, insulin, temp_insulin, sugar, drug, days, mask
        )
        out2 = self._cat_mlp_mix_forward(
            examination, insulin, temp_insulin, sugar, drug, days, mask
        )
        out = torch.cat([out1, out2], dim=1)
        out = self.insulin_predict_dense(out)
        return out

    def _mlp_cat_mix_forward(
        self,
        examination: Tensor,
        insulin: Tensor,
        temp_insulin: Tensor,
        sugar: Tensor,
        drug: Tensor,
        days: Tensor,
        mask: Tensor,
    ) -> Tensor:
        (
            examination_proj,
            insulin_proj,
            insulin_temp_proj,
            sugar_proj,
            drug_proj,
            day_proj,
        ) = self._base_info_mlp_encode(
            examination,
            insulin,
            temp_insulin,
            sugar,
            drug,
            days,
        )
        examination_proj_repeat = examination_proj.repeat((1, self.total_length, 1))
        # print(examination_proj_repeat.shape)  # torch.Size([1, 21, 256])
        # print(insulin_proj.shape)  # torch.Size([1, 21, 256])
        # print(sugar_proj.shape)  # torch.Size([1, 21, 256])
        proj_cat = torch.cat(
            [
                examination_proj_repeat,
                insulin_proj,
                insulin_temp_proj,
                sugar_proj,
                drug_proj,
                day_proj,
            ],
            dim=2,
        )
        # print(proj_cat.shape)  # torch.Size([1, 21, 1536])
        # assert 1==0

        rnn_out, _ = self.mlp_cat_rnn(proj_cat)
        attn_out = self.mlp_cat_attn(proj_cat)
        # attn_out = self.mlp_cat_attn(self.pos_encoder(proj_cat))

        rnn_out = rnn_out[:, -1, :]
        attn_out = self.pool(attn_out.permute((0, 2, 1))).squeeze(2)

        out = torch.cat([rnn_out, attn_out], dim=1)
        return out

    def _cat_mlp_mix_forward(
        self,
        examination: Tensor,
        insulin: Tensor,
        temp_insulin: Tensor,
        sugar: Tensor,
        drug: Tensor,
        days: Tensor,
        mask: Tensor,
    ) -> Tensor:
        # 将所有基础信息 拼接-mlp映射-分别做rnn、attn
        examination_rps = examination.repeat((1, self.total_length, 1))
        # print(examination_rps.shape)
        # print(insulin.shape)
        # print(sugar.shape)
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
        # print(time_data_cat.shape)
        # assert 1==0
        time_data_proj = self.data_cat_mlp(time_data_cat)

        rnn_out, _ = self.cat_mlp_rnn(time_data_proj)
        # attn_out = self.cat_mlp_attn(time_data_proj, time_data_proj)
        attn_out = self.cat_mlp_attn(time_data_proj)

        rnn_out = rnn_out[:, -1, :]
        attn_out = self.pool(attn_out.permute((0, 2, 1))).squeeze(2)

        out = torch.cat([rnn_out, attn_out], dim=1)
        return out

    def _base_info_mlp_encode(
        self,
        examination: Tensor,
        insulin: Tensor,
        temp_insulin: Tensor,
        sugar: Tensor,
        drug: Tensor,
        days: Tensor,
    ) -> Tensor:
        examination_proj = self.examination_encoder(examination)
        insulin_proj = self.insulin_encoder(insulin)
        insulin_temp_proj = self.insulin_encoder(temp_insulin)
        sugar_proj = self.sugar_encoder(sugar)
        drug_proj = self.drug_encoder(drug)
        day_proj = self.day_encoder(days)
        return (
            examination_proj,
            insulin_proj,
            insulin_temp_proj,
            sugar_proj,
            drug_proj,
            day_proj,
        )

    def unpack_data(self, data: dict):
        return self.unpack_sugar_data(data)

    def _base_unpack_data(self, data: dict):
        data = copy.deepcopy(data)
        length = self.window_size * self.points

        days = data[f"day_vec_{self.points}_point"].reshape(-1, length, 1)
        point_vec = data[f"point_vec_{self.points}_point"].reshape(-1, length, 1)

        days = torch.cat([days, point_vec], dim=2)

        examination = data["examination_vec"].unsqueeze(1)
        insulin = data[f"insulin_vec_{self.points}_point"].reshape(
            -1, length, self.insulin_size
        )
        sugar = data[f"sugar_vec_{self.points}_point"].reshape(
            -1, length, self.sugar_size
        )
        drug = data[f"drug_vec_{self.points}_point"].reshape(-1, length, self.drug_size)
        temp_insulin = data[f"insulin_vec_{self.points}_point_temp"].reshape(
            -1, length, self.insulin_size
        )

        mask = []

        return (examination, insulin, temp_insulin, sugar, drug, days, mask)

    def unpack_sugar_data(self, data: dict):
        (
            examination,
            insulin,
            temp_insulin,
            sugar,
            drug,
            days,
            mask,
        ) = self._base_unpack_data(data)
        sugar[:, -self.points :, :] = -1
        temp_insulin[:, -self.points :, :] = -1
        return (examination, insulin, temp_insulin, sugar, drug, days, mask)

    def unpack_insulin_data(self, data: dict):
        (
            examination,
            insulin,
            temp_insulin,
            sugar,
            drug,
            days,
            mask,
        ) = self._base_unpack_data(data)

        insulin[:, -self.points :, -1] = -1
        sugar[:, -self.points :, :] = -1
        # drug[:, -self.points:, :] = -1
        temp_insulin[:, -self.points :, :] = -1

        return (examination, insulin, temp_insulin, sugar, drug, days, mask)
